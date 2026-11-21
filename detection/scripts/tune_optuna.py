#!/usr/bin/env python3
"""Etapa 6 — Otimização de hiperparâmetros com Optuna (só na validação).

O estudo fica salvo em results/optuna/<study_name>.db (SQLite): pode ser
interrompido e continuado, e todos os trials ficam registrados. Cada trial
também é uma execução normal em results/experiments/06_optuna/trial_XXXX/.

Uso:
  python scripts/tune_optuna.py                          # roda/continua o estudo de configs/optuna.yaml
  python scripts/tune_optuna.py --n-trials 10            # mais 10 trials
  python scripts/tune_optuna.py --report                 # só relatório (importâncias, histórico)
  python scripts/tune_optuna.py --confirm-top 3 --seeds 0 1   # treina as 3 melhores por completo (06b)
  python scripts/tune_optuna.py --dry-run                # mostra um trial sorteado, sem treinar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pavedet.config import CONFIG_DIR, RESULTS_DIR, deep_merge, load_experiment, load_yaml, parse_overrides, save_yaml  # noqa: E402
from pavedet.training import resolve_config, train_run  # noqa: E402

METRIC_KEYS = {"mAP50_95": "metrics/mAP50-95(B)", "mAP50": "metrics/mAP50(B)",
               "P": "metrics/precision(B)", "R": "metrics/recall(B)"}


def suggest(trial, space: dict) -> dict:
    """Sorteia os valores do espaço de busca e devolve como overrides aninhados."""
    flat = {}
    for key, s in space.items():
        t = s["type"]
        if t == "float":
            flat[key] = trial.suggest_float(key, s["low"], s["high"], log=s.get("log", False))
        elif t == "int":
            flat[key] = trial.suggest_int(key, s["low"], s["high"], log=s.get("log", False))
        elif t == "categorical":
            flat[key] = trial.suggest_categorical(key, s["choices"])
        else:
            raise ValueError(f"tipo desconhecido em {key}: {t}")
    return parse_overrides([f"{k}={json.dumps(v)}" for k, v in flat.items()])


def make_sampler(optuna, spec):
    if spec.get("type", "tpe") == "tpe":
        return optuna.samplers.TPESampler(seed=spec.get("seed"), n_startup_trials=spec.get("n_startup_trials", 10))
    if spec["type"] == "random":
        return optuna.samplers.RandomSampler(seed=spec.get("seed"))
    raise ValueError(spec)


def make_pruner(optuna, spec):
    if not spec or spec.get("type") == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(n_startup_trials=spec.get("n_startup_trials", 8),
                                       n_warmup_steps=spec.get("n_warmup_steps", 20))


def base_config(oc: dict, cli: dict, allow_missing: bool = False) -> dict:
    exp = load_experiment(CONFIG_DIR / "base_train.yaml")
    if oc.get("inherit_best"):
        exp["inherit_best"] = oc["inherit_best"]
    return resolve_config(exp, None, cli, allow_missing=allow_missing)


def report(study, out_dir: Path) -> None:
    import optuna

    out_dir.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(out_dir / "trials.csv", index=False)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    states = df["state"].value_counts().to_dict() if len(df) else {}
    print(f"Trials: {len(study.trials)} {states}")
    if not done:
        return
    print(f"Melhor trial #{study.best_trial.number}: {study.best_value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"   {k} = {v}")
    save_yaml({"best_value": study.best_value, "best_trial": study.best_trial.number,
               "params": study.best_trial.params}, out_dir / "best_params.yaml")
    try:
        imp = optuna.importance.get_param_importances(study)
        (out_dir / "importances.json").write_text(json.dumps(imp, indent=2), encoding="utf-8")
        print("Importância dos hiperparâmetros (fANOVA):")
        for k, v in imp.items():
            print(f"   {k:22s} {v:.3f}")
    except Exception as e:
        print(f"(importâncias indisponíveis: {e})")
    try:
        import matplotlib

        matplotlib.use("Agg")
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

        for fn, name in [(plot_optimization_history, "history.png"), (plot_param_importances, "importances.png")]:
            ax = fn(study)
            ax.figure.set_size_inches(9, 5)
            ax.figure.tight_layout()
            ax.figure.savefig(out_dir / name, dpi=110)
    except Exception as e:
        print(f"(gráficos indisponíveis: {e})")
    print(f"Relatório em {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_DIR / "optuna.yaml"))
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="CHAVE=VALOR", help="overrides na config-base")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--confirm-top", type=int, default=0, help="treina as K melhores configs por completo (06b)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1], help="seeds da confirmação")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import optuna

    oc = load_yaml(args.config)
    cli = parse_overrides(args.set)
    out_dir = RESULTS_DIR / "optuna" / oc["study_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(RESULTS_DIR / 'optuna' / (oc['study_name'] + '.db')).resolve()}"
    base = base_config(oc, cli, allow_missing=args.dry_run)
    metric = base.get("selection_metric", "mAP50_95")

    if args.dry_run:
        t = optuna.trial.FixedTrial({k: (v["choices"][0] if v["type"] == "categorical" else v["low"])
                                     for k, v in oc["search_space"].items()})
        cfg = deep_merge(deep_merge(base, oc.get("trial_overrides") or {}), suggest(t, oc["search_space"]))
        train_run(cfg, oc["experiment"], "trial_XXXX", dry_run=True)
        return

    study = optuna.create_study(study_name=oc["study_name"], storage=storage, load_if_exists=True,
                                direction="maximize", sampler=make_sampler(optuna, oc.get("sampler", {})),
                                pruner=make_pruner(optuna, oc.get("pruner")))
    study.set_user_attr("metric", metric)
    study.set_user_attr("base_inherited", base.get("_inherited"))

    if args.report:
        report(study, out_dir)
        return

    if args.confirm_top:
        done = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
                      key=lambda t: t.value, reverse=True)[: args.confirm_top]
        if not done:
            raise SystemExit("Nenhum trial concluído ainda.")
        exp_name = "06b_optuna_confirm"
        exp = load_experiment(CONFIG_DIR / "experiments" / f"{exp_name}.yaml")
        variants = {}
        for t in done:
            variants[f"trial_{t.number:04d}"] = t.user_attrs["overrides"]
        save_yaml({"generated_from": oc["study_name"], "metric": metric, "variants": variants},
                  out_dir / "confirm_variants.yaml")
        for vname, ov in variants.items():
            for seed in args.seeds:
                cfg = deep_merge(deep_merge(base, ov), deep_merge(cli, {"train": {"seed": seed}}))
                cfg["_from_trial"] = vname
                train_run(cfg, exp["name"], vname)
        print(f"\nCompare com: python scripts/compare.py --experiments {exp_name}")
        return

    space = oc["search_space"]
    trial_ov = oc.get("trial_overrides") or {}
    key = METRIC_KEYS.get(metric)

    def objective(trial):
        ov = suggest(trial, space)
        trial.set_user_attr("overrides", ov)
        cfg = deep_merge(deep_merge(base, trial_ov), ov)

        def on_epoch(trainer):
            m = trainer.metrics or {}
            if key and key in m:
                trial.report(float(m[key]), trainer.epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"podado na época {trainer.epoch}")

        meta = train_run(cfg, oc["experiment"], f"trial_{trial.number:04d}",
                         callbacks={"on_fit_epoch_end": on_epoch}, extra_meta={"optuna_trial": trial.number})
        return float(meta["metrics"][metric])

    n = args.n_trials or oc.get("n_trials", 30)
    timeout = oc["timeout_h"] * 3600 if oc.get("timeout_h") else None
    study.optimize(objective, n_trials=n, timeout=timeout, gc_after_trial=True,
                   catch=(RuntimeError, ValueError))
    report(study, out_dir)
    print("\nPróximo passo: python scripts/tune_optuna.py --confirm-top 3")


if __name__ == "__main__":
    main()
