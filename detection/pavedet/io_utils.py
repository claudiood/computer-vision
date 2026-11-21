"""Leitura de imagens e rótulos YOLO, metadados EXIF e hashes perceptuais.

Só depende de Pillow e NumPy (pillow-heif é opcional, para .heic).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # suporte a HEIC/HEIF (fotos de iPhone)
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:  # pragma: no cover
    HEIF_OK = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
# formatos que a Ultralytics lê diretamente (HEIC é convertido no preparo)
YOLO_NATIVE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

EXIF_ORIENTATION = 0x0112
EXIF_DATETIME_ORIGINAL = 0x9003
EXIF_DATETIME = 0x0132
EXIF_MODEL = 0x0110


@dataclass
class Box:
    cls: int
    x: float
    y: float
    w: float
    h: float

    def line(self) -> str:
        return f"{self.cls} {self.x:.6f} {self.y:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class LabelFile:
    boxes: list[Box] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # linhas malformadas etc.


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_labels(path: Path, num_classes: int | None = None) -> LabelFile:
    """Lê um .txt YOLO. Linhas inválidas vão para `problems` (não para `boxes`)."""
    lf = LabelFile()
    if not path.exists():
        return lf
    for i, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 5:
            lf.problems.append(f"linha {i}: {len(parts)} colunas (esperado 5)")
            continue
        try:
            c = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            lf.problems.append(f"linha {i}: valor não numérico")
            continue
        if num_classes is not None and not 0 <= c < num_classes:
            lf.problems.append(f"linha {i}: classe {c} fora de 0..{num_classes - 1}")
            continue
        lf.boxes.append(Box(c, x, y, w, h))
    return lf


def write_labels(path: Path, boxes: list[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(b.line() + "\n" for b in boxes), encoding="utf-8")


def clean_boxes(boxes: list[Box], min_wh: float = 1e-3) -> tuple[list[Box], list[str]]:
    """Recorta caixas para [0,1], remove degeneradas e duplicadas exatas.

    min_wh: largura/altura mínima normalizada (1e-3 ≈ 4 px numa foto de 4000 px).
    """
    notes, out, seen = [], [], set()
    for b in boxes:
        x1, y1 = b.x - b.w / 2, b.y - b.h / 2
        x2, y2 = b.x + b.w / 2, b.y + b.h / 2
        cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            notes.append(f"caixa da classe {b.cls} recortada para dentro da imagem")
        w, h = cx2 - cx1, cy2 - cy1
        if w < min_wh or h < min_wh:
            notes.append(f"caixa degenerada removida (classe {b.cls}, w={b.w:.6f}, h={b.h:.6f})")
            continue
        nb = Box(b.cls, (cx1 + cx2) / 2, (cy1 + cy2) / 2, w, h)
        key = (nb.cls, round(nb.x, 4), round(nb.y, 4), round(nb.w, 4), round(nb.h, 4))
        if key in seen:
            notes.append(f"caixa duplicada removida (classe {b.cls})")
            continue
        seen.add(key)
        out.append(nb)
    return out, notes


@dataclass
class ImageMeta:
    width: int  # já com a orientação EXIF aplicada (como o usuário vê)
    height: int
    raw_width: int
    raw_height: int
    orientation: int
    datetime: datetime | None
    camera: str | None
    fmt: str
    mode: str


def _exif_datetime(exif) -> datetime | None:
    for tag in (EXIF_DATETIME_ORIGINAL, EXIF_DATETIME):
        val = None
        try:
            val = exif.get_ifd(0x8769).get(tag) if tag == EXIF_DATETIME_ORIGINAL else exif.get(tag)
        except Exception:
            val = exif.get(tag)
        if val:
            try:
                return datetime.strptime(str(val).strip("\x00 ")[:19], "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
    return None


def image_meta(path: Path) -> ImageMeta:
    with Image.open(path) as im:
        exif = im.getexif()
        orient = int(exif.get(EXIF_ORIENTATION, 1) or 1)
        rw, rh = im.size
        w, h = (rh, rw) if orient in (5, 6, 7, 8) else (rw, rh)
        cam = exif.get(EXIF_MODEL)
        return ImageMeta(w, h, rw, rh, orient, _exif_datetime(exif), str(cam).strip("\x00 ") if cam else None,
                         im.format or path.suffix.lstrip(".").upper(), im.mode)


def load_oriented(path: Path, max_side: int | None = None) -> Image.Image:
    """Abre a imagem com a orientação EXIF aplicada, em RGB, opcionalmente reduzida.

    Com max_side, JPEGs são decodificados já em escala reduzida (draft), o que
    é muito mais rápido para fotos de celular de 12+ MP.
    """
    with Image.open(path) as im:
        if max_side and im.format == "JPEG":
            im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return im.copy()


def dhash(im: Image.Image, size: int = 16) -> np.ndarray:
    """Difference hash (size*size bits). Robusto a redimensionamento/compressão."""
    g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
