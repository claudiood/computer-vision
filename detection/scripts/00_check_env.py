#!/usr/bin/env python3
"""Etapa 0 — Confere o ambiente antes de qualquer treino.

Mostra versões (Python, PyTorch, Ultralytics, Optuna), onde o treino vai rodar
(CUDA, MPS do Mac ou CPU), memória e disco, e se o dataset de origem está no lugar.
Com --smoke, faz o teste rápido do manual: 1 época no COCO8 (baixa ~1 MB).

Uso:
  python scripts/00_check_env.py
  python scripts/00_check_env.py --smoke
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pavedet.config import PROJECT_ROOT, REPO_ROOT  # noqa: E402
from pavedet.env import auto_device, env_info  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="treino de 1 época no COCO8")
    args = ap.parse_args()

    ok = True
    info = env_info()
    for k, v in info.items():
        print(f"{k:14s} {v}")
    print(f"{'device':14s} {auto_device()}  (auto: cuda > mps > cpu)")
    for mod, need in [("ultralytics", True), ("torch", True), ("optuna", True), ("pillow_heif", False),
                      ("cv2", True), ("pandas", True), ("matplotlib", True), ("albumentations", False)]:
        try:
            m = importlib.import_module(mod)
            print(f"  [ok] {mod} {getattr(m, '__version__', '')}")
            if mod == "albumentations":
                print("       ATENÇÃO: com albumentations instalado, a Ultralytics aplica blur/CLAHE/cinza "
                      "aleatórios no treino. Mantenha o mesmo ambiente em todos os experimentos.")
        except ImportError:
            print(f"  [{'FALTA' if need else 'opcional'}] {mod}")
            ok &= not need
    du = shutil.disk_usage(PROJECT_ROOT)
    print(f"disco livre    {du.free / 1e9:.1f} GB")
    src = REPO_ROOT / "dataset" / "detection"
    print(f"dataset        {'ok' if (src / 'images').exists() else 'NÃO ENCONTRADO'}: {src}")
    if auto_device() == "cpu":
        print("AVISO: sem GPU. Treinar detecção na CPU é bem lento; considere o Google Colab (manual, seção 9).")
    if args.smoke:
        from ultralytics import YOLO

        YOLO("yolo26n.pt").train(data="coco8.yaml", epochs=1, imgsz=320, device=auto_device(),
                                 project=str(PROJECT_ROOT / "results" / "smoke"), name="coco8", exist_ok=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
