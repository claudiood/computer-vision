#!/usr/bin/env python3
"""Etapa 3 — Validação do dataset preparado (antes de gastar GPU).

Confere, para cada data_*.yaml de data/<nome>/:
  * toda imagem listada existe e tem rótulo; rótulos no formato certo (5 colunas,
    classe em names, valores em [0,1], caixas com área > 0);
  * nenhuma imagem (nem cópia idêntica, por md5) aparece em duas partições;
  * nenhum GRUPO de mesma cena (manifesto) atravessa partições;
  * todas as classes aparecem em treino e validação de cada fold;
  * opcional (--ultralytics): a própria Ultralytics consegue ler o dataset.

Sai com código 1 se encontrar erro (útil para automatizar).

Uso:
  python scripts/03_validate_dataset.py
  python scripts/03_validate_dataset.py --name pavement_v1_clahe --ultralytics
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pavedet.config import DATA_DIR, SPLITS_DIR, load_yaml  # noqa: E402
from pavedet.dataset import load_dataset_cfg  # noqa: E402
from pavedet.io_utils import file_md5, read_labels  # noqa: E402
from pavedet.metrics import label_path_for, read_image_list  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None, help="versão do dataset (padrão: prepare.name)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ultralytics", action="store_true", help="também carrega com a Ultralytics (precisa dela instalada)")
    args = ap.parse_args()

    cfg = load_dataset_cfg(args.config)
    name = args.name or cfg["prepare"]["name"]
    root = DATA_DIR / name
    if not root.exists():
        raise SystemExit(f"{root} não existe: rode scripts/02_prepare_dataset.py")
    errors, warns = [], []
    yamls = sorted(root.glob("data_*.yaml"))
    split_name = name.split("_clahe")[0]
    manifest = SPLITS_DIR / f"{split_name}.csv"
    group_of = {}
    if manifest.exists():
        with open(manifest, encoding="utf-8") as f:
            group_of = {r["file"]: r["group"] for r in csv.DictReader(f)}
    else:
        warns.append(f"manifesto {manifest} não encontrado: checagem de grupos pulada")

    checked_labels = set()
    for y in yamls:
        d = load_yaml(y)
        nc = len(d["names"])
        parts = {}
        for split in ("train", "val", "test"):
            if split not in d:
                continue
            lst = Path(d[split]) if Path(d[split]).is_absolute() else Path(d["path"]) / d[split]
            imgs = read_image_list(lst)
            parts[split] = imgs
            counts = collections.Counter()
            for im in imgs:
                if not im.exists():
                    errors.append(f"{y.name}/{split}: imagem não existe {im}")
                    continue
                lbl = label_path_for(im)
                if not lbl.exists():
                    warns.append(f"{y.name}/{split}: {im.name} sem rótulo (será tratada como fundo)")
                    continue
                lf = read_labels(lbl, nc)
                if lbl not in checked_labels:
                    checked_labels.add(lbl)
                    for p in lf.problems:
                        errors.append(f"{lbl.name}: {p}")
                    for b in lf.boxes:
                        if not all(0 <= v <= 1 for v in (b.x, b.y, b.w, b.h)) or b.w <= 0 or b.h <= 0:
                            errors.append(f"{lbl.name}: caixa inválida {b.line()}")
                counts.update(b.cls for b in lf.boxes)
            if split in ("train", "val"):
                for c in range(nc):
                    if counts[c] == 0:
                        errors.append(f"{y.name}/{split}: classe {c} ({d['names'][c]}) sem nenhuma instância")
            print(f"{y.name:22s} {split:5s} {len(imgs):4d} imagens  "
                  + "  ".join(f"{d['names'][c]}={counts[c]}" for c in range(nc)))
        # vazamento entre partições (mesmo arquivo, mesmo conteúdo ou mesmo grupo)
        if y.name == "data_trainval.yaml":
            continue  # aqui a 'val' está dentro do treino de propósito (só monitoramento)
        names = {s: {p.name for p in v} for s, v in parts.items()}
        for a in names:
            for b in names:
                if a < b and names[a] & names[b]:
                    errors.append(f"{y.name}: {len(names[a] & names[b])} imagens em '{a}' e '{b}' ao mesmo tempo")
        if group_of:
            gs = {s: {group_of.get(n) for n in v} for s, v in names.items()}
            for a in gs:
                for b in gs:
                    if a < b and gs[a] & gs[b]:
                        errors.append(f"{y.name}: grupos de mesma cena em '{a}' e '{b}': {sorted(gs[a] & gs[b])[:5]}")
        md5 = collections.defaultdict(set)
        for s, v in parts.items():
            for p in v:
                if p.exists():
                    md5[file_md5(p)].add(s)
        dup = [h for h, s in md5.items() if len(s) > 1]
        if dup:
            errors.append(f"{y.name}: {len(dup)} imagens com conteúdo idêntico em partições diferentes")

    if args.ultralytics:
        from ultralytics.data.utils import check_det_dataset

        for y in yamls:
            try:
                check_det_dataset(str(y))
                print(f"Ultralytics leu {y.name} sem erro")
            except Exception as e:
                errors.append(f"Ultralytics não leu {y.name}: {e}")

    print()
    for w in sorted(set(warns)):
        print("AVISO:", w)
    for e in errors:
        print("ERRO:", e)
    print(f"\n{len(yamls)} arquivos data.yaml conferidos: {len(errors)} erros, {len(set(warns))} avisos.")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
