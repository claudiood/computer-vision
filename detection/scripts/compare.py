#!/usr/bin/env python3
"""Compara execuções registradas (results/registry.jsonl): tabelas e gráficos.

Para cada experimento: média ± desvio (entre seeds/folds) de cada variante,
diferença para a referência (variante `control`, ou a primeira), e se essa
diferença supera o RUÍDO medido no 01_baseline (desvio entre seeds idênticas).

Saída em results/comparisons/<nome>/:
  summary.csv / summary.md         tabela completa (gerais, por classe, tempo, custo)
  <exp>_<métrica>.png              barras com erro e faixa de ruído
  <exp>_per_class_F1.png           F1 por classe de cada variante
  tradeoff_*.png                   métrica × tempo de inferência / de treino
  all_runs.csv                     uma linha por execução (para análises próprias)

Uso:
  python scripts/compare.py                          # todos os experimentos
  python scripts/compare.py --experiments 02_models 03_imgsz --metric mAP50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pavedet import plots  # noqa: E402
from pavedet.config import RESULTS_DIR  # noqa: E402
from pavedet.metrics import flat  # noqa: E402
from pavedet.tracking import read_registry  # noqa: E402

MAIN = ["mAP50_95", "mAP50", "P", "R", "F1", "P_op", "R_op", "F1_op"]


def runs_table(entries: list[dict]) -> pd.DataFrame:
    rows = []
    for e in entries:
        m = e["metrics"]
        rows.append({"experiment": e["experiment"], "variant": e["variant"], "seed": e["seed"], "fold": e["fold"],
                     "model": e["config"].get("model"), "imgsz": e["config"]["train"].get("imgsz"),
                     "dataset": e["config"].get("dataset"), "oversample": str(e["config"].get("oversample")),
                     "epochs_run": e.get("epochs_run"), "best_epoch": e.get("best_epoch"),
                     "train_time_min": round((e.get("train_time_s") or 0) / 60, 2),
                     "gpu_peak_mb": e.get("gpu_peak_mb"), "params_M": e.get("params_M"), "gflops": e.get("gflops"),
                     "device": e.get("device"), "run_dir": e["run_dir"], **flat(m)})
    return pd.DataFrame(rows)


def noise_std(df: pd.DataFrame, metric: str) -> float | None:
    """Desvio entre seeds no baseline; se não houver, média dos desvios de grupos com >1 seed."""
    base = df[df.experiment.str.startswith("01_baseline")]
    if len(base) > 1:
        return float(base[metric].std(ddof=1))
    stds = df.groupby(["experiment", "variant", "fold"])[metric].std(ddof=1).dropna()
    return float(stds.mean()) if len(stds) else None


def reference_of(experiment: str, variants: set) -> str:
    """Variante de referência: chave `reference` do YAML do experimento, senão `control`, senão a primeira."""
    try:
        from pavedet.config import CONFIG_DIR, load_experiment

        ref = load_experiment(CONFIG_DIR / "experiments" / f"{experiment}.yaml").get("reference")
        if ref in variants:
            return ref
    except Exception:
        pass
    return "control" if "control" in variants else sorted(variants)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--metric", default="mAP50_95")
    ap.add_argument("--out", default=None, help="nome da pasta em results/comparisons/")
    args = ap.parse_args()

    entries = read_registry()
    if args.experiments:
        entries = [e for e in entries if e["experiment"] in args.experiments]
    if not entries:
        raise SystemExit("Nenhuma execução concluída encontrada no registro.")
    df = runs_table(entries)
    tag = args.out or ("_".join(args.experiments) if args.experiments else "all")
    out = RESULTS_DIR / "comparisons" / tag
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_runs.csv", index=False)
    metric = args.metric
    noise = noise_std(runs_table(read_registry()), metric)  # ruído do registro inteiro
    classes = sorted({c.split("__", 1)[1] for c in df.columns if c.startswith("F1__")})

    agg_cols = [c for c in MAIN + [f"F1__{c}" for c in classes] + [f"mAP50_95__{c}" for c in classes]
                + ["train_time_min", "speed_inference_ms", "params_M", "gflops", "best_epoch", "epochs_run"] if c in df]
    g = df.groupby(["experiment", "variant"], sort=False)
    summ = g[agg_cols].mean()
    summ.columns = [f"{c}" for c in summ.columns]
    summ[f"{metric}_std"] = g[metric].std(ddof=1)
    summ["n_runs"] = g.size()
    summ = summ.reset_index()

    lines = [f"# Comparação — {tag}", "", f"Métrica de seleção: **{metric}** (validação). "
             f"Ruído (desvio entre seeds do baseline): **{noise:.4f}**" if noise else
             f"Métrica de seleção: **{metric}**. Ruído: indisponível (rode 01_baseline com 3 seeds).", ""]
    rows_out = []
    for exp, s in summ.groupby("experiment", sort=False):
        s = s.copy()
        ref_name = reference_of(exp, set(s.variant))
        ref = float(s.loc[s.variant == ref_name, metric].iloc[0])
        s["delta_vs_ref"] = s[metric] - ref
        s["supera_ruido"] = s["delta_vs_ref"].apply(lambda d: "sim" if noise and abs(d) > noise and d > 0
                                                    else ("pior" if noise and d < -noise else "não"))
        s = s.sort_values(metric, ascending=False)
        rows_out.append(s)
        lines += [f"## {exp}", "", f"Referência: `{ref_name}`.", "",
                  "| variante | n | " + metric + " | ± | Δ ref | supera ruído | mAP50 | F1 | F1_op | "
                  + " | ".join(f"F1 {c}" for c in classes) + " | treino (min) | inferência (ms) |",
                  "|---" * (11 + len(classes)) + "|"]
        for _, r in s.iterrows():
            lines.append(f"| {r.variant} | {int(r.n_runs)} | {r[metric]:.4f} | "
                         f"{r[f'{metric}_std'] if r[f'{metric}_std'] == r[f'{metric}_std'] else float('nan'):.4f} | "
                         f"{r.delta_vs_ref:+.4f} | {r.supera_ruido} | {r.get('mAP50', np.nan):.3f} | "
                         f"{r.get('F1', np.nan):.3f} | {r.get('F1_op', np.nan):.3f} | "
                         + " | ".join(f"{r.get(f'F1__{c}', np.nan):.3f}" for c in classes)
                         + f" | {r.get('train_time_min', np.nan):.1f} | {r.get('speed_inference_ms', np.nan):.1f} |")
        best = s.iloc[0]
        lines += ["", f"Melhor: **{best.variant}** ({metric} = {best[metric]:.4f}, Δ = {best.delta_vs_ref:+.4f}; "
                  f"supera o ruído: {best.supera_ruido}).", ""]
        # gráficos do experimento
        plots.comparison_bars(list(s.variant), list(s[metric]), list(s[f"{metric}_std"].fillna(0)),
                              out / f"{exp}_{metric}.png", f"{exp}: {metric} na validação (média ± desvio)",
                              metric, ref=ref, ref_band=noise)
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
