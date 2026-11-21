#!/usr/bin/env python3
"""Gera um dataset SINTÉTICO pequeno, com a mesma estrutura do real, para testar
os scripts sem GPU e sem tocar nos dados verdadeiros.

Estrutura criada em <out>/:
  dataset/detection/images/*.jpg, labels/*.txt, labels/labels.txt
  imagens/detection/<anotador>/<imagem>.jpg   (para a auditoria achar o anotador)
  dataset.yaml                                 (config apontando para cá)

Inclui de propósito: uma imagem sem rótulo, uma caixa degenerada, uma foto com
EXIF girado e pares de fotos "da mesma cena" (EXIF com segundos de diferença).

Uso: python tests/make_synthetic_dataset.py --out /tmp/pavedet_test --n 48
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pavedet.config import CONFIG_DIR, load_yaml, save_yaml  # noqa: E402

NAMES = ["rachadura", "remendo", "buraco"]


def asphalt(rng, w, h):
    base = rng.integers(95, 140)
    a = rng.normal(base, 18, (h, w, 1)).clip(0, 255)
    return np.repeat(a, 3, 2).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    img_dir, lbl_dir = out / "dataset/detection/images", out / "dataset/detection/labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (lbl_dir / "labels.txt").write_text("\n".join(NAMES) + "\n")
    annotators = ["ana", "bia", "caio"]
    t0 = datetime(2026, 9, 20, 15, 0, 0)
    for i in range(args.n):
        w, h = (640, 480) if i % 4 else (480, 640)
        im = Image.fromarray(asphalt(rng, w, h))
        d = ImageDraw.Draw(im)
        lines = []
        # classe rara: remendo em ~1/3 das imagens
        k_classes = [0, 2] + ([1] if i % 3 == 0 else [])
        for c in k_classes:
            for _ in range(rng.integers(1, 3)):
                bw, bh = rng.uniform(0.1, 0.3), rng.uniform(0.08, 0.25)
                x, y = rng.uniform(bw / 2, 1 - bw / 2), rng.uniform(bh / 2, 1 - bh / 2)
                x1, y1, x2, y2 = (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h
                if c == 0:  # rachadura: linha escura em zigue-zague
                    pts = [(x1 + (x2 - x1) * t, y1 + (y2 - y1) * (0.5 + 0.4 * np.sin(9 * t))) for t in np.linspace(0, 1, 12)]
                    d.line(pts, fill=(35, 35, 35), width=3)
                elif c == 1:  # remendo: retângulo mais escuro e liso
                    d.rectangle([x1, y1, x2, y2], fill=(60, 60, 64))
                else:  # buraco: elipse escura com borda clara
                    d.ellipse([x1, y1, x2, y2], fill=(30, 28, 25), outline=(190, 185, 175), width=4)
                lines.append(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
        name = f"IMG_{1000 + i}"
        ann = annotators[i % 3]
        exif = Image.Exif()
        exif[0x0110] = "FakePhone"
        ifd = exif.get_ifd(0x8769)
        # pares da mesma cena: i e i+1 (i par) com 10 s de diferença; demais, 10 min
        ts = t0 + timedelta(minutes=10 * (i // 2), seconds=10 * (i % 2))
        ifd[0x9003] = ts.strftime("%Y:%m:%d %H:%M:%S")
        if i == 5:  # foto "deitada" com orientação EXIF 6: salva girada, rótulo no sistema já orientado
            exif[0x0112] = 6
            im = im.rotate(90, expand=True)
        im.save(img_dir / f"{name}.jpg", quality=90, exif=exif)
        (out / "imagens/detection" / ann).mkdir(parents=True, exist_ok=True)
        (out / "imagens/detection" / ann / f"{name}.jpg").write_bytes((img_dir / f"{name}.jpg").read_bytes())
        if i == args.n - 1:
            continue  # imagem sem rótulo
        if i == 1:
            lines.append("0 0.5 0.5 0.0 0.001")  # caixa degenerada
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")

    cfg = load_yaml(CONFIG_DIR / "dataset.yaml")
    cfg["source"] = {"images": str(img_dir), "labels": str(lbl_dir), "class_names_file": str(lbl_dir / "labels.txt"),
                     "annotator_dirs": str(out / "imagens/detection")}
    cfg["label_aliases"] = {}
    cfg["prepare"]["name"] = "pavement_v1"  # mesmo nome do real: os YAMLs de experimento funcionam sem mudança
    cfg["prepare"]["max_side"] = 640
    save_yaml(cfg, out / "dataset.yaml")
    print(f"Dataset sintético: {args.n} imagens em {out}")


if __name__ == "__main__":
    main()
