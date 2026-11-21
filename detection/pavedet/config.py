"""Configuração: caminhos do projeto, leitura de YAML, herança e overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

# detection/ (raiz deste projeto) e raiz do repositório
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
# PAVEDET_WORKDIR permite isolar data/, results/ e splits/ (usado pelos testes automáticos)
_WORK = Path(os.environ["PAVEDET_WORKDIR"]).resolve() if os.environ.get("PAVEDET_WORKDIR") else PROJECT_ROOT
RESULTS_DIR = _WORK / "results"
DATA_DIR = _WORK / "data"
SPLITS_DIR = _WORK / "splits"
REGISTRY_FILE = RESULTS_DIR / "registry.jsonl"


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} não contém um dicionário YAML")
    return data


def save_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def deep_merge(base: dict, override: dict) -> dict:
    """Mescla recursiva: valores de `override` vencem. Não altera as entradas."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_overrides(items: list[str] | None) -> dict:
    """Converte ["epochs=50", "oversample.method=rfs"] em dicionário aninhado.

    Os valores são interpretados como YAML (números, bool, listas, null).
    """
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override inválido '{item}'. Use chave=valor")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw) if raw != "" else None
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def resolve_path(p: str | Path, base: Path = PROJECT_ROOT) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def load_experiment(path: str | Path) -> dict:
    """Lê um YAML de experimento resolvendo a chave `extends` (herança)."""
    path = resolve_path(path, CONFIG_DIR / "experiments") if not Path(path).exists() else Path(path)
    cfg = load_yaml(path)
    own_name = cfg.get("name", path.stem)
    parent = cfg.pop("extends", None)
    if parent:
        parent_cfg = load_experiment((path.parent / parent).resolve())
        # variantes não são herdadas: cada experimento define as suas
        parent_cfg.pop("variants", None)
        cfg = deep_merge(parent_cfg, cfg)
    cfg["name"] = own_name
    cfg["_source"] = str(path)
    return cfg
