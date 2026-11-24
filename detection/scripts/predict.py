#!/usr/bin/env python3
"""Usa um modelo treinado em imagens novas, aplicando o MESMO preparo do treino.

O preparo (orientação EXIF, HEIC→RGB e, se o modelo foi treinado com ele, o
pré-processamento como CLAHE) é lido do dataset_info.json do dataset usado no
treino. Sem isso, o modelo veria no uso imagens diferentes das do treino.

Uso:
  python scripts/predict.py --run results/experiments/08_final/final_trainval/s0_ftrainval --source pasta_ou_imagem --conf 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pavedet import preprocess  # noqa: E402
from pavedet.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from pavedet.env import resolve_device  # noqa: E402
from pavedet.io_utils import IMG_EXTS, load_oriented  # noqa: E402
from pavedet.tracking import read_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="pasta da execução (run_meta.json)")
    ap.add_argument("--source", required=True, help="imagem ou pasta")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    meta = read_meta(Path(args.run))
    cfg = meta["config"]
    info_f = DATA_DIR / cfg["dataset"] / "dataset_info.json"
    info = json.loads(info_f.read_text(encoding="utf-8")) if info_f.exists() else {}
    pre = info.get("preprocess")
    max_side = info.get("max_side")
    src = Path(args.source)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    out = Path(args.out) if args.out else RESULTS_DIR / "predictions" / Path(args.run).parent.name
    out.mkdir(parents=True, exist_ok=True)
    # treino em treino+validação: vale a última época (o best.pt foi escolhido por uma validação contida no treino)
    wkey = "last" if str(cfg.get("data")) == "trainval" else "best"
    model = YOLO(meta["weights"][wkey])
    imgsz = cfg["train"].get("imgsz", 640)
    device = resolve_device(args.device or meta.get("device"))
    print(f"Pré-processamento: {pre or 'nenhum'}; max_side={max_side}; imgsz={imgsz}; device={device}")
    for f in files:
        im = preprocess.apply(load_oriented(f, max_side), pre)
        r = model.predict(im, conf=args.conf, imgsz=imgsz, device=device, verbose=False)[0]
        r.save(filename=str(out / f"{f.stem}.jpg"))
        found = [f"{r.names[int(c)]} {float(s):.2f}" for c, s in zip(r.boxes.cls, r.boxes.conf)]
        print(f"{f.name}: {len(found)} danos  {found}")
    print(f"Imagens com as caixas em {out}")


if __name__ == "__main__":
    main()
