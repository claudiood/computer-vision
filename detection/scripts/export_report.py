#!/usr/bin/env python3
"""Junta numa pasta só os resultados que vão para o relatório.

Rode na máquina de treino, no fim da rodada, e copie a pasta gerada para o seu
computador (a pasta results/ fica fora do git e pode ser grande demais para copiar inteira).

Conteúdo de relatorio_<data>/:
  00_config/              configs usados (dataset, base, optuna, experimentos) e o registry.jsonl
  01_comparacoes/         tabelas (summary.md/csv) e gráficos de cada etapa, com o veredito
  02_optuna/              histórico, importâncias e melhores parâmetros
  03_validacao_cruzada/   cv_report.md (métricas por fold, por classe e overfitting treino × validação)
                          e o results.png / results.csv de cada fold (curvas de loss e mAP por época)
  04_modelo_final/        results.png, labels.jpg, curvas PR/F1/P/R, matrizes de confusão,
                          val_batch real × previsto, run_meta.json e weights/best.pt
  05_teste/               relatório do teste, curvas e matrizes no conjunto de teste e as piores imagens
  LEIAME.md               índice com os números principais

Uso:
  python scripts/export_report.py
  python scripts/export_report.py --out ~/relatorio_final
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pavedet.config import CONFIG_DIR, PROJECT_ROOT, REGISTRY_FILE, RESULTS_DIR, load_yaml  # noqa: E402
from pavedet.tracking import read_registry  # noqa: E402

EXPERIMENTS = ["01_baseline", "02_models", "03_imgsz", "04_clahe", "05_augmentation", "05b_freeze",
               "06b_optuna_best"]
# arquivos que a Ultralytics gera em cada treino (os nomes das curvas variam entre versões)
RUN_FILES = ["results.png", "results.csv", "args.yaml", "run_meta.json", "labels.jpg", "labels_correlogram.jpg",
             "*curve*.png", "confusion_matrix*.png", "val_batch*_labels.jpg", "val_batch*_pred.jpg"]

missing: list[str] = []


def copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        missing.append(str(src))
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.npy", "*.cache"))
    else:
        shutil.copy2(src, dst)
    return True


def copy_run(run_dir: Path, dst: Path, with_weights: bool = False) -> None:
    for pattern in RUN_FILES:
        for f in run_dir.glob(pattern):
            copy(f, dst / f.name)
    if with_weights:
        copy(run_dir / "weights" / "best.pt", dst / "weights" / "best.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="pasta de saída (padrão: detection/relatorio_<data>)")
    args = ap.parse_args()
    out = Path(args.out).expanduser() if args.out else PROJECT_ROOT / f"relatorio_{time.strftime('%Y-%m-%d')}"
    out.mkdir(parents=True, exist_ok=True)
    registry = read_registry()
    if not registry:
        raise SystemExit(f"Registro vazio: {REGISTRY_FILE}")

    # 00 — configuração e registro
    copy(CONFIG_DIR, out / "00_config" / "configs")
    copy(REGISTRY_FILE, out / "00_config" / "registry.jsonl")

    # 01 — comparações (gera o resumo com todos os experimentos que rodaram)
    done = [e for e in EXPERIMENTS if any(r["experiment"] == e for r in registry)]
    if done:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "compare.py"), "--experiments", *done,
                        "--out", "resumo"], check=False, stdout=subprocess.DEVNULL)
        copy(RESULTS_DIR / "comparisons" / "resumo", out / "01_comparacoes")

    # 02 — Optuna
    study = load_yaml(CONFIG_DIR / "optuna.yaml").get("study_name", "hparams_v3")
    copy(RESULTS_DIR / "optuna" / study, out / "02_optuna")

    # 03 — validação cruzada: relatório + curvas de cada fold
    cv_dir = RESULTS_DIR / "cv"
    for name in ("cv_report.md", "cv_summary.json", "cv_runs.csv"):
        copy(cv_dir / name, out / "03_validacao_cruzada" / name)
    cv_summary = json.loads((cv_dir / "cv_summary.json").read_text(encoding="utf-8")) \
        if (cv_dir / "cv_summary.json").exists() else {}
    for fold, run_dir in sorted((cv_summary.get("runs") or {}).items()):
        rd = Path(run_dir)
        copy(rd / "results.png", out / "03_validacao_cruzada" / f"fold{fold}_results.png")
        copy(rd / "results.csv", out / "03_validacao_cruzada" / f"fold{fold}_results.csv")
        copy(rd / "BoxPR_curve.png", out / "03_validacao_cruzada" / f"fold{fold}_PR_curve.png")

    # 04 — modelo final
    final = [e for e in registry if e["experiment"] == "08_final"]
    if final:
        e = sorted(final, key=lambda r: r.get("finished", ""))[-1]
        copy_run(Path(e["run_dir"]), out / "04_modelo_final", with_weights=True)
        copy(Path(e["run_dir"]) / "val_best", out / "04_modelo_final" / "val_best")
    else:
        missing.append("execução do 08_final")

    # 05 — teste (a avaliação mais recente do 08_final)
    tests = sorted((RESULTS_DIR / "final").glob("08_final_*")) if (RESULTS_DIR / "final").exists() else []
    if tests:
        copy(tests[-1], out / "05_teste")
        copy(RESULTS_DIR / "final" / "TEST_LOCK.json", out / "05_teste" / "TEST_LOCK.json")
    else:
        missing.append("avaliação no teste (results/final/08_final_*)")

    # índice
    L = [f"# Resultados — {time.strftime('%Y-%m-%d %H:%M')}", ""]
    if cv_summary.get("metrics"):
        m = cv_summary["metrics"]
        L += [f"Configuração escolhida: `{cv_summary.get('config_from')}`.", "",
              "| | mAP50-95 | mAP50 | F1 |", "|---|---|---|---|",
              f"| Validação cruzada (3 folds, média ± desvio) | {m['mAP50_95']['mean']:.3f} ± {m['mAP50_95']['std']:.3f}"
              f" | {m['mAP50']['mean']:.3f} ± {m['mAP50']['std']:.3f} | {m['F1']['mean']:.3f} ± {m['F1']['std']:.3f} |"]
        if tests and (tests[-1] / "test_runs.csv").exists():
            import pandas as pd

            t = pd.read_csv(tests[-1] / "test_runs.csv").iloc[-1]
            L.append(f"| Teste (modelo final) | {t['mAP50_95']:.3f} | {t['mAP50']:.3f} | {t['F1']:.3f} |")
        over = cv_summary.get("overfitting") or []
        if over:
            g = sum(o["gap_mAP50"] for o in over) / len(over)
            L += ["", f"Overfitting (média dos folds): mAP50 no treino − mAP50 na validação = **{g:+.3f}**."]
    L += ["", "| Pasta | Conteúdo |", "|---|---|",
          "| `00_config/` | Configurações usadas e registro de todos os treinos |",
          "| `01_comparacoes/` | Tabela e gráficos de cada etapa (`summary.md`), com o veredito melhor/pior/empate |",
          "| `02_optuna/` | Histórico, importâncias dos hiperparâmetros e melhores parâmetros |",
          "| `03_validacao_cruzada/` | `cv_report.md` (por fold, por classe e overfitting) e curvas de cada fold |",
          "| `04_modelo_final/` | Curvas de treino, `labels.jpg`, curvas PR/F1, matrizes, real × previsto e `best.pt` |",
          "| `05_teste/` | Relatório do teste, curvas e matrizes no teste e as piores imagens |"]
    if missing:
        L += ["", "## Não encontrados", ""] + [f"- `{m}`" for m in missing]
    (out / "LEIAME.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nPasta pronta: {out}")
    print("Para copiar para o seu computador:  rsync -avz <usuario>@<maquina>:" + str(out) + "/ ./" + out.name + "/")


if __name__ == "__main__":
    main()
