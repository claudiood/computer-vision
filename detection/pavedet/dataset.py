"""Leitura do dataset de origem (pasta plana images/ + labels/ do repositório)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT, load_yaml, resolve_path
from .io_utils import IMG_EXTS, LabelFile, list_images, read_labels

DEFAULT_DATASET_CFG = PROJECT_ROOT / "configs" / "dataset.yaml"


@dataclass
class Record:
    name: str            # nome do arquivo de imagem
    image: Path
    label: Path | None   # None = imagem sem .txt
    annotator: str
    labels: LabelFile


def load_dataset_cfg(path: str | Path | None = None) -> dict:
    cfg = load_yaml(path or DEFAULT_DATASET_CFG)
    src = cfg["source"]
    for k in ("images", "labels", "class_names_file", "annotator_dirs"):
        if src.get(k):
            src[k] = resolve_path(src[k])
    return cfg


def read_class_names(cfg: dict) -> list[str]:
    """Classes do config, conferidas contra o labels.txt do Make Sense."""
    names = list(cfg["classes"])
    f = cfg["source"].get("class_names_file")
    if f and Path(f).exists():
        file_names = [l.strip() for l in Path(f).read_text(encoding="utf-8").splitlines() if l.strip()]
        if file_names != names:
            raise ValueError(
                f"Ordem das classes diverge!\n  configs/dataset.yaml: {names}\n  {f}: {file_names}\n"
                "A ordem define o número de cada classe nos .txt: corrija antes de seguir.")
    return names


def annotator_index(cfg: dict) -> dict[str, str]:
    """Mapeia nome de arquivo (sem extensão) -> anotador, pelas pastas imagens/detection/<nome>."""
    root = cfg["source"].get("annotator_dirs")
    idx: dict[str, str] = {}
    if not root or not Path(root).exists():
        return idx
    for person in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for f in person.rglob("*"):
            if f.is_file() and (f.suffix.lower() in IMG_EXTS or f.suffix.lower() == ".txt"):
                idx.setdefault(f.stem, person.name)
    return idx


def load_records(cfg: dict) -> tuple[list[Record], list[str]]:
    """Retorna os registros (uma imagem cada) e avisos (rótulos órfãos etc.)."""
    src = cfg["source"]
    nc = len(cfg["classes"])
    img_dir, lbl_dir = Path(src["images"]), Path(src["labels"])
    aliases = cfg.get("label_aliases") or {}
    who = annotator_index(cfg)
    ignore = {Path(src["class_names_file"]).name} if src.get("class_names_file") else set()
    records, warnings, used = [], [], set()
    for img in list_images(img_dir):
        lbl = lbl_dir / aliases.get(img.name, img.stem + ".txt")
        has = lbl.exists()
        if has:
            used.add(lbl.name)
        records.append(Record(img.name, img, lbl if has else None,
                              who.get(img.stem, who.get(Path(aliases.get(img.name, "")).stem, "desconhecido")),
                              read_labels(lbl, nc) if has else LabelFile()))
    for t in sorted(lbl_dir.glob("*.txt")):
        if t.name not in used and t.name not in ignore:
            warnings.append(f"rótulo sem imagem correspondente: {t.name}")
    return records, warnings
