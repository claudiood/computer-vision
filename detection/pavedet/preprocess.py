"""Pré-processamento determinístico (aula 10, parte 1).

Diferente do aumento de dados, o pré-processamento é aplicado do mesmo jeito em
treino, validação, teste E no uso (predict). Por isso ele fica registrado no
dataset_info.json do dataset preparado e o scripts/predict.py o reaplica.

A Ultralytics já faz letterbox, BGR→RGB e /255; aqui ficam só as técnicas
opcionais que precisam ser testadas contra a linha de base.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

PREPROCESSORS = ("none", "clahe")


def clahe(im: Image.Image, clip_limit: float = 2.0, tile: int = 8) -> Image.Image:
    """CLAHE só na luminância (canal L do LAB), para não distorcer as cores."""
    import cv2

    lab = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB))


def apply(im: Image.Image, spec: dict | None) -> Image.Image:
    """spec: {"method": "clahe", "clip_limit": 2.0, "tile": 8} ou None/"none"."""
    if not spec or spec.get("method", "none") == "none":
        return im
    if spec["method"] == "clahe":
        return clahe(im, spec.get("clip_limit", 2.0), spec.get("tile", 8))
    raise ValueError(f"pré-processamento desconhecido: {spec['method']}")
