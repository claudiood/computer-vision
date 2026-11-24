#!/usr/bin/env python3
"""Etapa 2 — Preparação do dataset (cópia limpa no formato YOLO + divisão sem vazamento).

O dataset original (dataset/detection) NÃO é alterado. Este script gera:

  data/<nome>/images/*.jpg          imagens com a orientação EXIF aplicada, HEIC→JPEG,
                                    reduzidas para no máx. `max_side` px (rótulos normalizados
                                    não mudam com o redimensionamento)
  data/<nome>/labels/*.txt          rótulos limpos (caixas degeneradas/duplicadas removidas)
  data/<nome>/split_*.txt           listas de imagens por partição (train/val de cada fold, test)
  data/<nome>/data_fold<k>.yaml     data.yaml para a Ultralytics (fold 0 = padrão)
  data/<nome>/data_trainval.yaml    treino em treino+validação (só para o treino definitivo opcional)
  data/<nome>/split_report.md       distribuição por partição
  splits/<nome>.csv                 MANIFESTO da divisão (versionado no git): garante que todo o
                                    grupo usa exatamente a mesma divisão

Divisão: (1) teste fixo, (2) K folds sobre o restante. A unidade da divisão é o GRUPO de
fotos da mesma cena (ver configs/dataset.yaml → grouping), estratificado por classe e anotador.

Uso:
  python scripts/02_prepare_dataset.py                 # usa o manifesto se existir; senão cria
  python scripts/02_prepare_dataset.py --resplit       # refaz a divisão (cria manifesto novo)
  python scripts/02_prepare_dataset.py --preprocess clahe --clahe-clip 1.0   # cria pavement_v3_clahe1 (MESMA divisão)
  python scripts/02_prepare_dataset.py --preprocess clahe --clahe-clip 2.0   # cria pavement_v3_clahe2
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pavedet.config import DATA_DIR, SPLITS_DIR, save_yaml  # noqa: E402
from pavedet.dataset import load_dataset_cfg, load_records, read_class_names  # noqa: E402
from pavedet import preprocess  # noqa: E402
from pavedet.preprocess import PREPROCESSORS  # noqa: E402
from pavedet.io_utils import clean_boxes, dhash, image_meta, load_oriented, write_labels  # noqa: E402
from pavedet.splitting import Item, build_groups, stratified_group_assign  # noqa: E402


def safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="configs/dataset.yaml")
    ap.add_argument("--name", default=None, help="nome da versão (padrão: prepare.name do config)")
    ap.add_argument("--max-side", type=int, default=None)
    ap.add_argument("--resplit", action="store_true", help="ignora o manifesto existente e refaz a divisão")
    ap.add_argument("--split-only", action="store_true", help="só calcula a divisão e o manifesto (não copia imagens)")
    ap.add_argument("--preprocess", choices=PREPROCESSORS, default="none",
                    help="pré-processamento determinístico gravado nas imagens (ex.: clahe)")
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-tile", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_dataset_cfg(args.config)
    pc = cfg["prepare"]
    split_name = args.name or pc["name"]  # a divisão (manifesto) é a da versão-base
    # o nome da versão com CLAHE inclui o clipLimit: pavement_v3_clahe1, pavement_v3_clahe2, ...
    suffix = f"clahe{args.clahe_clip:g}".replace(".", "p") if args.preprocess == "clahe" else args.preprocess
    name = split_name if args.preprocess == "none" else f"{split_name}_{suffix}"
    pre_spec = None if args.preprocess == "none" else {"method": args.preprocess, "clip_limit": args.clahe_clip,
                                                         "tile": args.clahe_tile}
    max_side = args.max_side or pc["max_side"]
    names = read_class_names(cfg)
    nc = len(names)
    out = DATA_DIR / name
    manifest_path = SPLITS_DIR / f"{split_name}.csv"
    if pre_spec and not manifest_path.exists():
        raise SystemExit(f"Prepare primeiro a versão sem pré-processamento ({split_name}): a versão "
                         f"'{name}' precisa usar exatamente a mesma divisão.")

    records, warnings = load_records(cfg)
    for w in warnings:
        print("AVISO:", w)
    excluded = set(cfg.get("exclude") or [])
    background = set(cfg.get("background_images") or [])
    policy = cfg.get("unlabeled_policy", "exclude")

    # ---- seleção e limpeza
    kept, log = [], collections.Counter()
    for r in records:
        if r.name in excluded:
            log["excluída (config)"] += 1
            continue
        boxes, notes = clean_boxes(r.labels.boxes, cfg["clean"]["min_wh"])
        log["caixas removidas na limpeza"] += len(r.labels.boxes) - len(boxes)
        if not boxes:
            is_bg = r.name in background or (r.label is None and policy == "background")
            if not is_bg:
                log["sem rótulo (fora)" if r.label is None else "rótulo vazio (fora)"] += 1
                continue
            log["fundo (sem objetos)"] += 1
        kept.append((r, boxes))
    stems = collections.Counter(safe_stem(r.name) for r, _ in kept)
    if any(c > 1 for c in stems.values()):
        raise SystemExit(f"Nomes repetidos após normalização: {[s for s, c in stems.items() if c > 1]}")
    print(f"{len(kept)} imagens selecionadas de {len(records)}: {dict(log)}")

    # ---- metadados e hashes para agrupar a mesma cena
    items = []
    for r, boxes in kept:
        meta = image_meta(r.image)
        items.append(Item(r.name, r.annotator, np.bincount([b.cls for b in boxes], minlength=nc),
                          dhash(load_oriented(r.image, 256)), meta.datetime, meta.camera))
    g = pc["grouping"]
    group_ids, links = build_groups(items, g["hash_threshold"], g["time_window_s"], g.get("seq_window", 0))

    # ---- divisão: reaproveita o manifesto (reprodutível) ou cria um novo
    split_of, fold_of = {}, {}
    if manifest_path.exists() and not args.resplit:
        with open(manifest_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        prev = {r["image"]: r for r in rows}
        missing = [it.name for it in items if it.name not in prev]
        if missing:
            raise SystemExit(f"{len(missing)} imagens não estão no manifesto {manifest_path.name} "
                             f"(ex.: {missing[:3]}). O dataset mudou: rode com --resplit e um --name novo "
                             "(ex.: pavement_v4) para não misturar resultados de divisões diferentes.")
        for it in items:
            split_of[it.name] = prev[it.name]["split"]
            fold_of[it.name] = int(prev[it.name]["fold"]) if prev[it.name]["fold"] not in ("", "-1") else -1
        print(f"Divisão lida de {manifest_path} ({len(rows)} linhas)")
    else:
        seed, restarts = pc["seed"], pc.get("restarts", 300)
        tv_test = stratified_group_assign(items, group_ids, [1 - pc["test_fraction"], pc["test_fraction"]],
                                          seed=seed, restarts=restarts)
        tv_idx = [i for i, s in enumerate(tv_test) if s == 0]
        sub_items = [items[i] for i in tv_idx]
        sub_groups = [group_ids[i] for i in tv_idx]
        remap = {gid: k for k, gid in enumerate(sorted(set(sub_groups)))}
        folds = stratified_group_assign(sub_items, [remap[x] for x in sub_groups], [1] * pc["folds"],
                                        seed=seed + 1, restarts=restarts)
        for it, s in zip(items, tv_test):
            split_of[it.name] = "test" if s == 1 else "trainval"
            fold_of[it.name] = -1
        for it, f in zip(sub_items, folds):
            fold_of[it.name] = int(f)
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "file", "annotator", "group", "split", "fold"] + [f"n_{n}" for n in names])
            for it, gid in zip(items, group_ids):
                w.writerow([it.name, safe_stem(it.name) + ".jpg", it.annotator, gid, split_of[it.name],
                            fold_of[it.name]] + [int(x) for x in it.class_counts])
        print(f"Manifesto salvo em {manifest_path}")

    # ---- checagem de vazamento: um grupo nunca pode estar em duas partições
    part = lambda n: "test" if split_of[n] == "test" else f"fold{fold_of[n]}"
    by_group = collections.defaultdict(set)
    for it, gid in zip(items, group_ids):
        by_group[gid].add(part(it.name))
    leaks = {gid: p for gid, p in by_group.items() if len(p) > 1}
    if leaks:
        print(f"AVISO: {len(leaks)} grupos atravessam partições (o agrupamento mudou desde o manifesto?)")

    if args.split_only:
        print(f"OK (--split-only) em {time.time() - t0:.0f}s")
        return

    # ---- cópia das imagens e rótulos
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for k, (r, boxes) in enumerate(kept, 1):
        stem = safe_stem(r.name)
        dst = out / "images" / f"{stem}.jpg"
        if not dst.exists():
            preprocess.apply(load_oriented(r.image, max_side), pre_spec).save(dst, quality=pc.get("jpeg_quality", 95))
        write_labels(out / "labels" / f"{stem}.txt", boxes)
        if k % 25 == 0:
            print(f"  {k}/{len(kept)} imagens ({time.time() - t0:.0f}s)")

    # ---- listas e data.yaml
    def write_list(fname, members):
        # entradas "./" são relativas à pasta da lista (convenção da Ultralytics)
        (out / fname).write_text("".join(f"./images/{safe_stem(n)}.jpg\n" for n in sorted(members)),
                                           encoding="utf-8")
        return fname

    all_names = [it.name for it in items]
    test = [n for n in all_names if split_of[n] == "test"]
    trainval = [n for n in all_names if split_of[n] != "test"]
    K = max(fold_of.values()) + 1
    yaml_names = {i: n for i, n in enumerate(names)}
    write_list("split_test.txt", test)
    write_list("split_trainval.txt", trainval)
    for k in range(K):
        tr = [n for n in trainval if fold_of[n] != k]
        va = [n for n in trainval if fold_of[n] == k]
        save_yaml({"path": str(out.resolve()), "train": write_list(f"split_fold{k}_train.txt", tr),
                   "val": write_list(f"split_fold{k}_val.txt", va), "test": "split_test.txt", "names": yaml_names},
                  out / f"data_fold{k}.yaml")
    save_yaml({"path": str(out.resolve()), "train": "split_trainval.txt",
               # Atenção: aqui a 'val' está DENTRO do treino. Serve só para acompanhar o treino
               # definitivo em treino+val; não use essas métricas para escolher nada.
               "val": "split_fold0_val.txt", "test": "split_test.txt", "names": yaml_names},
              out / "data_trainval.yaml")

    # ---- relatório da divisão
    cnt = {n: it.class_counts for n, it in zip(all_names, items)}
    parts = [("test", test)] + [(f"fold{k}", [n for n in trainval if fold_of[n] == k]) for k in range(K)]
    L = [f"# Divisão do dataset `{name}`", "",
         f"Manifesto: `splits/{manifest_path.name}`. Grupos (mesma cena): {len(set(group_ids))}. "
         f"Ligações entre fotos: {len(links)}.", "",
         "| partição | imagens | grupos | " + " | ".join(names) + " | anotadores |", "|---" * (4 + nc) + "|"]
    for pname, members in parts:
        c = sum((cnt[n] for n in members), np.zeros(nc, dtype=int))
        anns = collections.Counter(it.annotator for it in items if it.name in set(members))
        L.append(f"| {pname} | {len(members)} | {len({group_ids[all_names.index(n)] for n in members})} | "
                 + " | ".join(str(int(x)) for x in c) + " | "
                 + ", ".join(f"{a} {v}" for a, v in sorted(anns.items())) + " |")
    L += ["", f"Validação padrão = fold0; treino = demais folds. Teste usado UMA vez (scripts/final_test.py).",
          f"Grupos que atravessam partições: {len(leaks)} (deve ser 0)."]
    (out / "split_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    info = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "max_side": max_side,
            "n_images": len(items), "classes": names, "folds": K, "selection_log": dict(log),
            "grouping": g, "manifest": str(manifest_path), "preprocess": pre_spec}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))
    print(f"\nPronto em {time.time() - t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
