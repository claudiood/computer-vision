#!/usr/bin/env python3
"""Roda as variantes de um experimento (configs/experiments/<nome>.yaml).

Cada variante × seed × fold vira uma execução em
results/experiments/<experimento>/<variante>/s<seed>_f<fold>/ e uma linha em
results/registry.jsonl. Execuções já concluídas são puladas, então dá para
interromper e continuar depois.

Exemplos:
  python scripts/run_experiment.py 01_baseline
  python scripts/run_experiment.py 02_models --variants yolo11n yolo26n
  python scripts/run_experiment.py 03_imgsz --seeds 0 1 2            # repetir para medir a variância
  python scripts/run_experiment.py 02_models --folds 0 1 2           # validação cruzada (mais robusto)
  python scripts/run_experiment.py 01_baseline --set train.epochs=5 --dry-run
  python scripts/run_experiment.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pavedet.config import CONFIG_DIR, deep_merge, load_experiment, parse_overrides  # noqa: E402
from pavedet.training import resolve_config, train_run  # noqa: E402


def find_experiment(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    exp_dir = CONFIG_DIR / "experiments"
    for cand in [exp_dir / name, exp_dir / f"{name}.yaml"]:
        if cand.exists():
            return cand
    matches = sorted(exp_dir.glob(f"{name}*.yaml"))
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"Experimento '{name}' não encontrado em {exp_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", help="nome do YAML em configs/experiments (ex.: 02_models)")
    ap.add_argument("--variants", nargs="*", help="rodar só estas variantes")
    ap.add_argument("--seeds", nargs="*", type=int, help="seeds (padrão: as do YAML ou [0])")
    ap.add_argument("--folds", nargs="*", type=int, help="folds de validação (padrão: os do YAML ou [0])")
    ap.add_argument("--set", nargs="*", default=[], metavar="CHAVE=VALOR",
                    help="sobrescreve a config em todas as variantes, ex.: train.epochs=50 device=mps")
    ap.add_argument("--force", action="store_true", help="roda de novo execuções concluídas (as antigas são preservadas)")
    ap.add_argument("--dry-run", action="store_true", help="só mostra o que seria rodado")
    ap.add_argument("--list", action="store_true", help="lista os experimentos disponíveis")
    args = ap.parse_args()

    if args.list or not args.experiment:
        for p in sorted((CONFIG_DIR / "experiments").glob("*.yaml")):
            e = load_experiment(p)
            print(f"{p.stem:24s} {len(e.get('variants') or {}):2d} variantes  {e.get('description', '').strip()[:80]}")
        return

    exp = load_experiment(find_experiment(args.experiment))
    name = exp["name"]
    variants = exp.get("variants") or {"default": {}}
    if args.variants:
        unknown = set(args.variants) - set(variants)
        if unknown:
            raise SystemExit(f"Variantes inexistentes: {unknown}. Disponíveis: {list(variants)}")
        variants = {k: v for k, v in variants.items() if k in args.variants}
    seeds = args.seeds if args.seeds else exp.get("seeds", [0])
    folds = args.folds if args.folds is not None and len(args.folds) else exp.get("folds", [0])
    cli = parse_overrides(args.set)

    print(f"== {name}: {len(variants)} variantes × {len(seeds)} seeds × {len(folds)} folds")
    if exp.get("description"):
        print("   " + exp["description"].strip().replace("\n", "\n   "))
    done, failed = 0, []
    for vname, vcfg in variants.items():
        for fold in folds:
            for seed in seeds:
                per_run = deep_merge(cli, {"train": {"seed": seed}})
                if str(exp.get("data", "fold0")).startswith("fold") and "data" not in (vcfg or {}) and "data" not in cli:
                    per_run["data"] = f"fold{fold}"
                cfg = resolve_config(exp, vcfg, per_run, allow_missing=args.dry_run)
                try:
                    train_run(cfg, name, vname, force=args.force, dry_run=args.dry_run)
                    done += 1
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # uma variante com erro não derruba as outras
                    print(f"[ERRO] {vname} s{seed} f{fold}: {type(e).__name__}: {e}")
                    failed.append(f"{vname}/s{seed}/f{fold}")
    print(f"\nConcluídas/puladas: {done}. Falhas: {len(failed)} {failed if failed else ''}")
    if not args.dry_run:
        print(f"Compare com: python scripts/compare.py --experiments {name}")


if __name__ == "__main__":
    main()
