#!/usr/bin/env python3
"""Compara execuções registradas (results/registry.jsonl): tabelas e gráficos.

Para cada experimento: a métrica de cada variante e a diferença para a
referência, calculada fold contra o mesmo fold (as decisões usam o fold 0).

Veredito de cada variante contra a referência (margem = `adopt_margin`, 0,015):
  melhor   superou a referência por mais que a margem
  pior     perdeu por mais que a margem
  empate   diferença menor que a margem: ruído, fica a opção mais simples
Com mais de um fold em comum, "melhor"/"pior" exige o mesmo sinal em todos.

Referências de fora do experimento: a chave `include_best` do YAML traz a
melhor variante de outros experimentos já rodados (ex.: o yolo26n do E1 entra
no E2 sem treinar de novo).

Saída em results/comparisons/<nome>/:
  summary.csv / summary.md         tabela completa (gerais, por classe, tempo)
  <exp>_<métrica>.png              barras com erro
  <exp>_per_class_F1.png           F1 por classe de cada variante
  tradeoff_*.png                   métrica × tempo de inferência / de treino
  all_runs.csv                     uma linha por execução (para análises próprias)

Uso:
  python scripts/compare.py --experiments 02_models
  python scripts/compare.py --experiments 01_baseline 02_models 03_imgsz 04_clahe 06_ablation --out resumo
  python scripts/compare.py --experiments 05_optuna --metric mAP50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pavedet import plots  # noqa: E402
from pavedet.config import CONFIG_DIR, RESULTS_DIR, load_experiment  # noqa: E402
from pavedet.metrics import flat  # noqa: E402
from pavedet.tracking import best_of, read_registry  # noqa: E402


def runs_table(entries: list[dict]) -> pd.DataFrame:
    rows = []
    for e in entries:
        m = e["metrics"]
        rows.append({"experiment": e["experiment"], "variant": e["variant"], "seed": e["seed"], "fold": str(e["fold"]),
                     "model": e["config"].get("model"), "imgsz": e["config"]["train"].get("imgsz"),
                     "dataset": e["config"].get("dataset"), "oversample": str(e["config"].get("oversample")),
                     "epochs_run": e.get("epochs_run"), "best_epoch": e.get("best_epoch"),
                     "train_time_min": round((e.get("train_time_s") or 0) / 60, 2),
                     "gpu_peak_mb": e.get("gpu_peak_mb"), "params_M": e.get("params_M"), "gflops": e.get("gflops"),
                     "device": e.get("device"), "run_dir": e["run_dir"], **flat(m)})
    return pd.DataFrame(rows)


def experiment_yaml(experiment: str) -> dict:
    try:
        return load_experiment(CONFIG_DIR / "experiments" / f"{experiment}.yaml")
    except Exception:
        return {}


def external_refs(experiment: str, metric: str, registry: list[dict]) -> list[dict]:
    """Execuções trazidas de outros experimentos pela chave `include_best` do YAML."""
    out = []
    y = experiment_yaml(experiment)
    for label, sources in (y.get("include_best") or {}).items():
        best = best_of(sources, metric, registry, margin=float(y.get("adopt_margin", 0.0)))
        if best is None:
            print(f"AVISO: include_best '{label}': nenhum de {sources} tem execuções concluídas")
            continue
        _, src_exp, src_var, _, runs = best
        for r in runs:
            out.append({**r, "experiment": experiment, "variant": label, "_source": f"{src_exp}/{src_var}"})
    return out


def reference_of(experiment: str, variants: set) -> str:
    """Variante de referência: chave `reference` do YAML, senão `control`, senão a primeira."""
    ref = experiment_yaml(experiment).get("reference")
    if ref in variants:
        return ref
    return "control" if "control" in variants else sorted(variants)[0]


def paired(df: pd.DataFrame, var: str, ref: str, metric: str, margin: float = 0.015) -> tuple[float, str, str]:
    """Diferença pareada por fold (média das seeds de cada fold). Retorna (Δ médio, veredito, Δ por fold)."""
    a = df[df.variant == var].groupby("fold")[metric].mean()
    b = df[df.variant == ref].groupby("fold")[metric].mean()
    common = sorted(set(a.index) & set(b.index))
    if var == ref:
        return 0.0, "referência", ""
    if not common:
        return float("nan"), "sem folds em comum", ""
    d = np.array([a[f] - b[f] for f in common])
    if d.mean() > margin and (d > 0).all():
        verdict = "melhor"
    elif d.mean() < -margin and (d < 0).all():
        verdict = "pior"
    else:
        verdict = "empate"
    per_fold = ", ".join(f"f{f}: {x:+.3f}" for f, x in zip(common, d))
    return float(d.mean()), verdict, per_fold


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--metric", default="mAP50_95")
    ap.add_argument("--out", default=None, help="nome da pasta em results/comparisons/")
    args = ap.parse_args()

    registry = read_registry()
    metric = args.metric
    entries = registry
    if args.experiments:
        entries = [e for e in registry if e["experiment"] in args.experiments]
        for exp in args.experiments:
            entries += external_refs(exp, metric, registry)
    if not entries:
        raise SystemExit("Nenhuma execução concluída encontrada no registro.")
    df = runs_table(entries)
    tag = args.out or ("_".join(args.experiments) if args.experiments else "all")
    out = RESULTS_DIR / "comparisons" / tag
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_runs.csv", index=False)
    classes = sorted({c.split("__", 1)[1] for c in df.columns if c.startswith("F1__")})

    agg_cols = [c for c in ["mAP50_95", "mAP50", "P", "R", "F1", "P_op", "R_op", "F1_op", "best_conf"]
                + [f"F1__{c}" for c in classes] + [f"mAP50_95__{c}" for c in classes]
                + ["train_time_min", "speed_inference_ms", "params_M", "gflops", "best_epoch", "epochs_run"] if c in df]
    g = df.groupby(["experiment", "variant"], sort=False)
    summ = g[agg_cols].mean()
    summ[f"{metric}_std"] = g[metric].std(ddof=1)
    summ["n_runs"] = g.size()
    summ["folds"] = g["fold"].apply(lambda s: ",".join(sorted(set(s))))
    summ = summ.reset_index()

    lines = [f"# Comparação — {tag}", "",
             f"Métrica de seleção: **{metric}** (validação). Δ ref = diferença contra a referência, fold a fold. "
             "Veredito: **melhor**/**pior** = diferença maior que a margem de adoção (0,015); **empate** = "
             "menor que a margem (ruído).", ""]
    rows_out = []
    for exp, s in summ.groupby("experiment", sort=False):
        s = s.copy()
        dexp = df[df.experiment == exp]
        ref_name = reference_of(exp, set(s.variant))
        margin = float(experiment_yaml(exp).get("adopt_margin", 0.015))
        res = [paired(dexp, v, ref_name, metric, margin) for v in s.variant]
        s["delta_vs_ref"] = [r[0] for r in res]
        s["veredito"] = [r[1] for r in res]
        s["delta_por_fold"] = [r[2] for r in res]
        s = s.sort_values(metric, ascending=False)
        rows_out.append(s)
        ref_val = float(s.loc[s.variant == ref_name, metric].iloc[0])
        lines += [f"## {exp}", "", f"Referência: `{ref_name}`.", "",
                  "| variante | folds | " + metric + " | ± | Δ ref | veredito | Δ por fold | mAP50 | F1 | "
                  + " | ".join(f"F1 {c}" for c in classes) + " | treino (min) |",
                  "|---" * (10 + len(classes)) + "|"]
        for _, r in s.iterrows():
            std = r[f"{metric}_std"]
            lines.append(f"| {r.variant} | {r.folds} | {r[metric]:.4f} | {std if std == std else float('nan'):.4f} | "
                         f"{r.delta_vs_ref:+.4f} | {r.veredito} | {r.delta_por_fold} | {r.get('mAP50', np.nan):.3f} | "
                         f"{r.get('F1', np.nan):.3f} | "
                         + " | ".join(f"{r.get(f'F1__{c}', np.nan):.3f}" for c in classes)
                         + f" | {r.get('train_time_min', np.nan):.1f} |")
        best = s.iloc[0]
        lines += ["", f"Maior valor: **{best.variant}** ({metric} = {best[metric]:.4f}, Δ = {best.delta_vs_ref:+.4f}, "
                  f"veredito: {best.veredito}). Os próximos experimentos só adotam uma variante nova se ela "
                  "superar a configuração vigente por mais que a margem; num empate, fica a vigente.", ""]
        plots.comparison_bars(list(s.variant), list(s[metric]), list(s[f"{metric}_std"].fillna(0)),
                              out / f"{exp}_{metric}.png", f"{exp}: {metric} na validação (média ± desvio entre folds)",
                              metric, ref=ref_val)
        if classes:
            plots.grouped_bars(list(s.variant), {c: list(s.get(f"F1__{c}", np.nan)) for c in classes},
                               out / f"{exp}_per_class_F1.png", f"{exp}: F1 por classe", "F1")
    full = pd.concat(rows_out)
    full.to_csv(out / "summary.csv", index=False)
    if "speed_inference_ms" in full:
        lab = [f"{a}/{b}" for a, b in zip(full.experiment, full.variant)]
        plots.scatter_tradeoff(full["speed_inference_ms"], full[metric], lab, out / "tradeoff_inference.png",
                               f"{metric} × tempo de inferência (ms/img, na validação)", "ms por imagem", metric)
        plots.scatter_tradeoff(full["train_time_min"], full[metric], lab, out / "tradeoff_train_time.png",
                               f"{metric} × tempo de treino", "minutos", metric)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saída: {out}")


if __name__ == "__main__":
    main()
