#!/usr/bin/env python3
"""Treino avulso (uma execução), a partir da config-base + overrides.

Útil para testes rápidos ou para treinar uma configuração fora dos experimentos
planejados. Também fica registrado em results/registry.jsonl.

Exemplos:
  python scripts/train.py --name teste_rapido --set model=yolo26n.pt train.epochs=10 train.fraction=0.5
  python scripts/train.py --name meu_teste --config configs/experiments/03_imgsz.yaml --set train.imgsz=960
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pavedet.config import CONFIG_DIR, load_experiment, parse_overrides  # noqa: E402
from pavedet.training import resolve_config, train_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="nome da variante (pasta em results/experiments/adhoc/)")
    ap.add_argument("--experiment", default="adhoc", help="nome do grupo no registro (padrão: adhoc)")
    ap.add_argument("--config", default=str(CONFIG_DIR / "base_train.yaml"),
                    help="YAML de base (base_train.yaml ou um experimento; as variantes são ignoradas)")
    ap.add_argument("--set", nargs="*", default=[], metavar="CHAVE=VALOR")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp = load_experiment(args.config)
    cfg = resolve_config(exp, None, parse_overrides(args.set))
    train_run(cfg, args.experiment, args.name, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
