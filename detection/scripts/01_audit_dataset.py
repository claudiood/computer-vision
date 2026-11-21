#!/usr/bin/env python3
"""Etapa 1 — Auditoria do dataset de origem (não altera nenhum arquivo do dataset).

Gera em results/audit/:
  report.md              resumo, tabelas e lista de problemas encontrados
  images.csv / boxes.csv dados por imagem e por caixa
  near_duplicates.csv    pares de fotos da mesma cena (risco de vazamento)
  plots/*.png            distribuição de classes, tamanhos de caixa, resoluções...
  crops/<classe>.jpg     recortes por classe e anotador (conferir se a classe 0/1/2
                         significa a mesma coisa para todo mundo)
  overlays/*.jpg         imagens com as caixas desenhadas (amostra + casos suspeitos)
  unlabeled.jpg          miniaturas das imagens sem .txt
  near_duplicates.jpg    pares de quase-duplicatas lado a lado

Uso:
  python scripts/01_audit_dataset.py
  python scripts/01_audit_dataset.py --config configs/dataset.yaml --out results/audit --max-side 1280
"""
from __future__ import annotations

import argparse
import collections
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from pavedet.config import RESULTS_DIR  # noqa: E402
from pavedet.dataset import load_dataset_cfg, load_records, read_class_names  # noqa: E402
from pavedet.io_utils import (HEIF_OK, YOLO_NATIVE_EXTS, clean_boxes, dhash, file_md5,  # noqa: E402
                              image_meta, load_oriented)
from pavedet.splitting import Item, build_groups  # noqa: E402

COLORS = [(230, 57, 70), (42, 157, 143), (244, 162, 97), (69, 123, 157), (131, 56, 236)]


def draw_boxes(im: Image.Image, boxes, names) -> Image.Image:
    im = im.copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    lw = max(2, W // 300)
    for b in boxes:
        x1, y1 = (b.x - b.w / 2) * W, (b.y - b.h / 2) * H
        x2, y2 = (b.x + b.w / 2) * W, (b.y + b.h / 2) * H
        c = COLORS[b.cls % len(COLORS)]
        d.rectangle([x1, y1, x2, y2], outline=c, width=lw)
        d.text((x1 + lw, max(0, y1 - 12)), f"{b.cls}:{names[b.cls]}", fill=c)
    return im


def grid(images: list[Image.Image], cols: int, cell: int, captions: list[str] | None = None) -> Image.Image:
    rows = max(1, (len(images) + cols - 1) // cols)
    cap_h = 14 if captions else 0
    out = Image.new("RGB", (cols * cell, rows * (cell + cap_h)), (255, 255, 255))
    d = ImageDraw.Draw(out)
    for i, im in enumerate(images):
        t = im.copy()
        t.thumbnail((cell - 4, cell - 4))
        x, y = (i % cols) * cell, (i // cols) * (cell + cap_h)
        out.paste(t, (x + (cell - t.width) // 2, y + (cell - t.height) // 2))
        if captions:
            d.text((x + 3, y + cell), captions[i][: cell // 6], fill=(0, 0, 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="configs/dataset.yaml")
    ap.add_argument("--out", default=str(RESULTS_DIR / "audit"))
    ap.add_argument("--max-side", type=int, default=1280, help="resolução usada para hashes/recortes")
    ap.add_argument("--crops-per-cell", type=int, default=6, help="recortes por classe×anotador")
    ap.add_argument("--overlay-samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_dataset_cfg(args.config)
    names = read_class_names(cfg)
    nc = len(names)
    out = Path(args.out)
    for sub in ("plots", "crops", "overlays"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    records, warnings = load_records(cfg)
    rng = random.Random(args.seed)
    issues: list[str] = list(warnings)
    if not HEIF_OK and any(r.image.suffix.lower() in (".heic", ".heif") for r in records):
        issues.append("pillow-heif não instalado: imagens HEIC não puderam ser lidas (pip install pillow-heif)")

    img_rows, box_rows, items, thumbs = [], [], [], {}
    crops = collections.defaultdict(list)  # (cls, annotator) -> [crop]
    md5s = collections.defaultdict(list)
    print(f"Lendo {len(records)} imagens...")
    for k, r in enumerate(records, 1):
        row = {"image": r.name, "annotator": r.annotator, "ext": r.image.suffix.lower(),
               "has_label": r.label is not None, "label_file": r.label.name if r.label else "",
               "label_problems": "; ".join(r.labels.problems)}
        try:
            meta = image_meta(r.image)
            im = load_oriented(r.image, args.max_side)
        except Exception as e:  # imagem corrompida ou formato sem suporte
            issues.append(f"{r.name}: não foi possível abrir ({e})")
            row["error"] = str(e)
            img_rows.append(row)
            continue
        md5s[file_md5(r.image)].append(r.name)
        cleaned, notes = clean_boxes(r.labels.boxes, cfg["clean"]["min_wh"])
        counts = np.bincount([b.cls for b in r.labels.boxes], minlength=nc)
        row.update(width=meta.width, height=meta.height, megapixels=round(meta.width * meta.height / 1e6, 2),
                   aspect=round(meta.width / meta.height, 3), exif_orientation=meta.orientation,
                   datetime=meta.datetime.isoformat() if meta.datetime else "", camera=meta.camera or "",
                   n_boxes=len(r.labels.boxes), n_boxes_after_clean=len(cleaned), clean_notes="; ".join(notes),
                   **{f"n_{n}": int(c) for n, c in zip(names, counts)})
        img_rows.append(row)
        for n in notes:
            issues.append(f"{r.name}: {n}")
        for p in r.labels.problems:
            issues.append(f"{r.label.name}: {p}")
        items.append(Item(r.name, r.annotator, counts, dhash(im), meta.datetime, meta.camera))
        thumbs[r.name] = im.copy()
        thumbs[r.name].thumbnail((320, 320))
        W, H = im.size
        for b in r.labels.boxes:
            box_rows.append({"image": r.name, "annotator": r.annotator, "cls": b.cls, "class": names[b.cls],
                             "x": b.x, "y": b.y, "w": b.w, "h": b.h,
                             "w_px": round(b.w * meta.width), "h_px": round(b.h * meta.height),
                             "img_long": max(meta.width, meta.height),
                             "area_frac": b.w * b.h, "aspect_wh": (b.w * meta.width) / max(b.h * meta.height, 1e-9)})
            if b.w * b.h > 0 and len(crops[(b.cls, r.annotator)]) < 40:
                x1, y1 = max(0, (b.x - b.w / 2) * W), max(0, (b.y - b.h / 2) * H)
                x2, y2 = min(W, (b.x + b.w / 2) * W), min(H, (b.y + b.h / 2) * H)
                if x2 - x1 >= 4 and y2 - y1 >= 4:
                    crops[(b.cls, r.annotator)].append((im.crop((int(x1), int(y1), int(x2), int(y2))), r.name))
        suspicious = meta.orientation != 1 or notes or r.name in (cfg.get("label_aliases") or {})
        if suspicious and r.labels.boxes:
            draw_boxes(im, r.labels.boxes, names).save(out / "overlays" / f"check_{Path(r.name).stem}.jpg", quality=85)
        if k % 20 == 0:
            print(f"  {k}/{len(records)} ({time.time() - t0:.0f}s)")

    # amostra aleatória de overlays
    labeled = [r for r in records if r.labels.boxes and r.name in thumbs]
    for r in rng.sample(labeled, min(args.overlay_samples, len(labeled))):
        im = load_oriented(r.image, args.max_side)
        draw_boxes(im, r.labels.boxes, names).save(out / "overlays" / f"sample_{Path(r.name).stem}.jpg", quality=85)

    # ---- duplicatas e quase-duplicatas
    for h, files in md5s.items():
        if len(files) > 1:
            issues.append(f"arquivos idênticos (mesmo md5): {', '.join(files)}")
    g = cfg["prepare"]["grouping"]
    group_ids, links = build_groups(items, g["hash_threshold"], g["time_window_s"], g.get("seq_window", 0))
    nn = []
    for i, a in enumerate(items):
        d = min((int(np.count_nonzero(a.hash != b.hash)) for j, b in enumerate(items) if j != i), default=256)
        nn.append(d)
    with open(out / "near_duplicates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b", "reason"])
        w.writerows(links)
    if links:
        pairs = []
        for a, b, why in links[:30]:
            pairs += [thumbs[a], thumbs[b]]
        caps = [x for a, b, why in links[:30] for x in (a, f"{b} ({why})")]
        grid(pairs, 6, 260, caps).save(out / "near_duplicates.jpg", quality=85)
    sizes = collections.Counter(group_ids)
    multi = sum(1 for c in sizes.values() if c > 1)

    # ---- imagens sem rótulo
    unl = [r.name for r in records if r.label is None and r.name in thumbs]
    if unl:
        grid([thumbs[n] for n in unl], 6, 260, unl).save(out / "unlabeled.jpg", quality=85)

    # ---- recortes por classe x anotador (consistência do significado de cada classe)
    annotators = sorted({r.annotator for r in records})
    for c, n in enumerate(names):
        cells, caps = [], []
        for a in annotators:
            pool = crops.get((c, a), [])
            pick = rng.sample(pool, min(args.crops_per_cell, len(pool)))
            for i in range(args.crops_per_cell):
                if i < len(pick):
                    cells.append(pick[i][0])
                    caps.append(f"{a}: {pick[i][1]}")
                else:
                    cells.append(Image.new("RGB", (8, 8), (240, 240, 240)))
                    caps.append("")
        grid(cells, args.crops_per_cell, 240, caps).save(out / "crops" / f"{c}_{n}.jpg", quality=85)

    # ---- CSVs
    def write_csv(path, rows):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(out / "images.csv", img_rows)
    write_csv(out / "boxes.csv", box_rows)

    # ---- gráficos
    ok_rows = [r for r in img_rows if "width" in r]
    cls_tot = collections.Counter(b["cls"] for b in box_rows)
    img_with = collections.Counter(c for r in ok_rows for c in range(nc) if r.get(f"n_{names[c]}", 0) > 0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].bar(names, [cls_tot[c] for c in range(nc)], color=["#e63946", "#2a9d8f", "#f4a261"])
    ax[0].set_title("Instâncias por classe")
    ax[1].bar(names, [img_with[c] for c in range(nc)], color=["#e63946", "#2a9d8f", "#f4a261"])
    ax[1].set_title("Imagens que contêm a classe")
    for a in ax:
        for p in a.patches:
            a.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "plots" / "class_counts.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bottom = np.zeros(len(annotators))
    for c, n in enumerate(names):
        v = np.array([sum(1 for b in box_rows if b["cls"] == c and b["annotator"] == a) for a in annotators])
        ax.bar(annotators, v, bottom=bottom, label=n, color=["#e63946", "#2a9d8f", "#f4a261"][c % 3])
        bottom += v
    ax.set_title("Instâncias por anotador e classe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plots" / "class_by_annotator.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for c, n in enumerate(names):
        a = [b["area_frac"] for b in box_rows if b["cls"] == c and b["area_frac"] > 0]
        if a:
            ax[0].hist(np.log10(a), bins=25, alpha=0.6, label=n)
        r = [b["aspect_wh"] for b in box_rows if b["cls"] == c and b["area_frac"] > 0]
        if r:
            ax[1].hist(np.log2(r), bins=25, alpha=0.6, label=n)
    ax[0].set_title("log10(área da caixa / área da imagem)")
    ax[1].set_title("log2(largura/altura) da caixa")
    ax[0].legend()
    ax[2].hist([r["n_boxes"] for r in ok_rows], bins=range(0, max(r["n_boxes"] for r in ok_rows) + 2))
    ax[2].set_title("Caixas por imagem")
    fig.tight_layout()
    fig.savefig(out / "plots" / "box_stats.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter([r["width"] for r in ok_rows], [r["height"] for r in ok_rows], s=12)
    ax[0].set_xlabel("largura (px)")
    ax[0].set_ylabel("altura (px)")
    ax[0].set_title("Resolução (com orientação EXIF)")
    ax[1].hist(nn, bins=30)
    ax[1].axvline(g["hash_threshold"], color="r", ls="--", label="limiar")
    ax[1].set_title("Distância dHash ao vizinho mais próximo")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "plots" / "resolution_and_similarity.png", dpi=120)
    plt.close(fig)

    # tamanho efetivo do menor lado das caixas em cada imgsz candidato
    imgszs = [640, 960, 1280]
    small_tbl = []
    for c, n in enumerate(names):
        rows = [b for b in box_rows if b["cls"] == c]
        line = [n]
        for s in imgszs:
            # letterbox: o lado maior da imagem vira `s` px
            ms = [min(b["w_px"], b["h_px"]) * s / b["img_long"] for b in rows]
            line.append(f"{np.median(ms):.0f} px / {100 * np.mean(np.array(ms) < 8):.0f}% < 8 px" if ms else "-")
        small_tbl.append(line)

    # ---- relatório
    n_img = len(records)
    n_lab = sum(1 for r in records if r.label is not None)
    ext = collections.Counter(r.image.suffix.lower() for r in records)
    non_native = [r.name for r in records if r.image.suffix.lower() not in YOLO_NATIVE_EXTS]
    rot = [r["image"] for r in ok_rows if r.get("exif_orientation", 1) != 1]
    per_ann = collections.defaultdict(lambda: [0, 0] + [0] * nc)
    for r in ok_rows:
        p = per_ann[r["annotator"]]
        p[0] += 1
        p[1] += int(r["has_label"])
        for c in range(nc):
            p[2 + c] += r.get(f"n_{names[c]}", 0)
    total_inst = sum(cls_tot.values())
    L = ["# Auditoria do dataset de detecção", "",
         f"Gerado por `scripts/01_audit_dataset.py` em {time.strftime('%Y-%m-%d %H:%M')} ({time.time() - t0:.0f}s).", "",
         "## Resumo", "",
         f"- Imagens: **{n_img}** ({n_lab} com .txt, {n_img - n_lab} sem .txt)",
         f"- Instâncias (caixas): **{total_inst}**; média {total_inst / max(n_lab, 1):.1f} por imagem anotada",
         f"- Classes (ordem do labels.txt): " + ", ".join(f"`{i}` = {n}" for i, n in enumerate(names)),
         f"- Formatos: " + ", ".join(f"{k} ({v})" for k, v in ext.most_common()),
         f"- Grupos de quase-duplicatas (mesma cena): {multi} grupos com mais de 1 imagem; "
         f"{len(set(group_ids))} grupos no total", "",
         "## Instâncias por classe", "", "| classe | instâncias | % | imagens com a classe |", "|---|---|---|---|"]
    for c, n in enumerate(names):
        L.append(f"| {c} {n} | {cls_tot[c]} | {100 * cls_tot[c] / max(total_inst, 1):.1f}% | {img_with[c]} |")
    L += ["", "## Por anotador", "", "| anotador | imagens | com .txt | " + " | ".join(names) + " |",
          "|---" * (3 + nc) + "|"]
    for a in annotators:
        p = per_ann[a]
        L.append(f"| {a} | {p[0]} | {p[1]} | " + " | ".join(str(x) for x in p[2:]) + " |")
    L += ["", "## Resoluções", ""]
    if ok_rows:
        mp = [r["megapixels"] for r in ok_rows]
        L.append(f"- Megapixels: mín {min(mp)}, mediana {np.median(mp):.1f}, máx {max(mp)}")
        L.append(f"- Resoluções distintas: {len({(r['width'], r['height']) for r in ok_rows})}")
        L.append(f"- Imagens com orientação EXIF ≠ 1 (giradas pela câmera): {len(rot)}")
    L += ["", "## Tamanho das caixas na entrada da rede", "",
          "Menor lado da caixa (mediana) e % de caixas com menos de 8 px, por `imgsz`:", "",
          "| classe | " + " | ".join(f"imgsz={s}" for s in imgszs) + " |", "|---" * (1 + len(imgszs)) + "|"]
    L += ["| " + " | ".join(r) + " |" for r in small_tbl]
    L += ["", "## Problemas encontrados", ""]
    L += [f"- {i}" for i in issues] or ["- nenhum"]
    if non_native:
        L.append(f"- {len(non_native)} imagens em formato que a YOLO não lê diretamente "
                 f"(convertidas para JPEG no preparo): {', '.join(non_native)}")
    if n_img - n_lab:
        L.append(f"- {n_img - n_lab} imagens sem .txt (ver `unlabeled.jpg`): decidir se são fundo (sem dano) "
                 "ou se faltou anotar. Por padrão ficam FORA do dataset (`unlabeled_policy: exclude`).")
    L += ["", "## Arquivos para conferência visual", "",
          "- `crops/<classe>.jpg`: uma linha por anotador. Se uma linha mostrar outro tipo de dano, "
          "aquele anotador usou outra ordem de classes no Make Sense.",
          "- `overlays/check_*.jpg`: imagens com EXIF girado, rótulos limpos ou nomes corrigidos (alias).",
          "- `near_duplicates.jpg`: pares agrupados (ficam sempre na mesma partição).",
          "- `unlabeled.jpg`: imagens sem anotação.", ""]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))
    print(f"\nRelatório: {out / 'report.md'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
