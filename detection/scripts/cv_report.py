#!/usr/bin/env python3
"""E7 — Relatório da validação cruzada (3 folds) da configuração escolhida.

A configuração vencedora é a mesma que o 07_cv e o 08_final herdam (a vigente ao
fim do `inherit_best` do 07_cv.yaml). O fold 0 vem do experimento de onde ela saiu;
os folds 1 e 2, do 07_cv. O script confere que as configurações são iguais antes
de juntar.

Saída em results/cv/:
  cv_summary.json   média ± desvio (3 folds), mediana da melhor época (usada no E8)
                    e o limiar de confiança que maximiza o F1 (mediana entre folds)
  cv_report.md      tabela por fold e por classe (vai para o relatório)
  cv_runs.csv       uma linha por fold

Uso:
  python scripts/cv_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pavedet.config import CONFIG_DIR, PROJECT_ROOT, RESULTS_DIR, load_experiment  # noqa: E402
from pavedet.metrics import flat  # noqa: E402
from pavedet.tracking import best_of, read_registry  # noqa: E402

KEYS = ("dataset", "model", "oversample", "train")
IGNORE_TRAIN = {"seed", "plots"}


def comparable(cfg: dict) -> str:
    c = {k: cfg.get(k) for k in KEYS}
    c["train"] = {k: v for k, v in (c["train"] or {}).items() if k not in IGNORE_TRAIN}
    return json.dumps(c, sort_keys=True, default=str)


def main() -> None:
    exp_cv = load_experiment(CONFIG_DIR / "experiments" / "07_cv.yaml")
    registry = read_registry()
    metric = exp_cv.get("selection_metric", "mAP50_95")
    sources = exp_cv["inherit_best"]
    best = best_of(sources, metric, [e for e in registry if e["experiment"] != "07_cv"],
                   margin=float(exp_cv.get("adopt_margin", 0.0)))
    if best is None:
        raise SystemExit(f"Nenhuma execução concluída em {sources}.")
    _, src_exp, src_var, src_cfg, src_runs = best
    cv_runs = [e for e in registry if e["experiment"] == "07_cv"]
    if not cv_runs:
        raise SystemExit("Nenhuma execução do 07_cv. Rode: python scripts/run_experiment.py 07_cv")
    ref = comparable(src_cfg)
    bad = [e["run_dir"] for e in cv_runs if comparable(e["config"]) != ref]
    if bad:
        raise SystemExit(f"As execuções do 07_cv não têm a mesma configuração de {src_exp}/{src_var}: {bad}. "
                         "Algum experimento rodou depois do 07_cv e mudou o vencedor? Rode o 07_cv de novo com --force.")
    runs = sorted(src_runs + cv_runs, key=lambda e: str(e["fold"]))
    folds = [str(e["fold"]) for e in runs]
    if len(set(folds)) != len(folds):
        print(f"AVISO: folds repetidos {folds}; vale a média de cada fold.")

    df = pd.DataFrame([{"fold": str(e["fold"]), "experiment": e["experiment"], "variant": e["variant"],
                        "best_epoch": e.get("best_epoch"), "epochs_run": e.get("epochs_run"),
                        "train_time_min": round((e.get("train_time_s") or 0) / 60, 1), **flat(e["metrics"])}
                       for e in runs])
    per_fold = df.groupby("fold").mean(numeric_only=True)
    main_cols = [c for c in ["mAP50_95", "mAP50", "P", "R", "F1", "F1_op", "best_conf"] if c in per_fold]
    cls_cols = sorted(c for c in per_fold.columns if "__" in c and c.split("__")[0] in ("mAP50", "mAP50_95", "F1"))
    summary = {
        "config_from": f"{src_exp}/{src_var}",
        "folds": list(per_fold.index),
        "n_folds": int(len(per_fold)),
        "metrics": {c: {"mean": float(per_fold[c].mean()), "std": float(per_fold[c].std(ddof=1))}
                    for c in main_cols + cls_cols},
        "median_best_epoch": int(np.median(per_fold["best_epoch"].dropna())),
        "best_conf_median": float(np.median(per_fold["best_conf"].dropna())) if "best_conf" in per_fold else None,
        "config": {k: src_cfg.get(k) for k in KEYS},
    }
    out = RESULTS_DIR / "cv"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cv_runs.csv", index=False)
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    L = [f"# Validação cruzada — {summary['n_folds']} folds", "",
         f"Configuração: `{src_exp}/{src_var}` (modelo {src_cfg.get('model')}, imgsz "
         f"{src_cfg['train'].get('imgsz')}, dataset {src_cfg.get('dataset')}, oversample {src_cfg.get('oversample')}).",
         "", "| fold | " + " | ".join(main_cols) + " | melhor época |", "|---" * (len(main_cols) + 2) + "|"]
    for f, r in per_fold.iterrows():
        L.append(f"| {f} | " + " | ".join(f"{r[c]:.3f}" for c in main_cols) + f" | {r['best_epoch']:.0f} |")
    L.append("| **média ± desvio** | " + " | ".join(
        f"**{summary['metrics'][c]['mean']:.3f} ± {summary['metrics'][c]['std']:.3f}**" for c in main_cols) + " | |")
    L += ["", "## Por classe (média ± desvio entre folds)", "", "| métrica | média | desvio |", "|---|---|---|"]
    for c in cls_cols:
        L.append(f"| {c} | {summary['metrics'][c]['mean']:.3f} | {summary['metrics'][c]['std']:.3f} |")
    L += ["", f"Mediana da melhor época: **{summary['median_best_epoch']}** (épocas do treino definitivo, E8).",
          f"Limiar de confiança que maximiza o F1 (mediana): **{summary['best_conf_median']}** "
          "(use no teste com `final_test.py --conf cv`).", "",
          "Observação: o fold 0 foi usado em todas as decisões (modelo, tamanho, resolução, CLAHE, aumento "
          "e Optuna); os folds 1 e 2 não participaram de nenhuma escolha. Se a média dos folds 1 e 2 ficar bem "
          "abaixo do fold 0, as escolhas se ajustaram demais ao fold 0."]
    (out / "cv_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nSaída: {out.relative_to(PROJECT_ROOT) if out.is_relative_to(PROJECT_ROOT) else out}")


if __name__ == "__main__":
    main()
