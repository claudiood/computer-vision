#!/usr/bin/env python3
"""Etapa 10 — Avaliação final no conjunto de TESTE (uma única vez).

Avalia no teste TODAS as execuções do experimento final (uma por seed) e
relata média ± desvio. Não escolhe o "melhor seed" pelo teste (isso seria
vazamento): o modelo a entregar é escolhido pela VALIDAÇÃO e informado no
relatório, mas a nota do projeto é a média.

Proteção: a primeira execução grava results/final/TEST_LOCK.json. Rodar de novo
exige --rerun e fica registrado no relatório (o teste deixa de ser "virgem" se
alguém ajustar o modelo depois de ver o resultado).

Uso:
  python scripts/final_test.py --experiment 08_final --confirm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pavedet.config import RESULTS_DIR  # noqa: E402
from pavedet.evaluation import evaluate, summary_line  # noqa: E402
from pavedet.metrics import flat  # noqa: E402
from pavedet.tracking import read_registry  # noqa: E402

LOCK = RESULTS_DIR / "final" / "TEST_LOCK.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="08_final")
    ap.add_argument("--variant", default=None, help="padrão: todas as variantes do experimento")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--confirm", action="store_true", help="obrigatório: confirma que a escolha do modelo terminou")
    ap.add_argument("--rerun", action="store_true", help="roda de novo mesmo com o teste já usado (fica registrado)")
    args = ap.parse_args()

    if not args.confirm:
        raise SystemExit("O teste deve ser usado UMA vez, depois de TODAS as decisões. Se é o caso, rode com --confirm.")
    history = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else []
    if history and not args.rerun:
        raise SystemExit(f"O teste já foi usado em {history[-1]['when']} ({LOCK}). Use --rerun só se for "
                         "inevitável; isso ficará registrado no relatório.")
    runs = [e for e in read_registry() if e["experiment"] == args.experiment
            and (args.variant is None or e["variant"] == args.variant)]
    if not runs:
        raise SystemExit(f"Nenhuma execução concluída de {args.experiment}.")

    out_root = RESULTS_DIR / "final" / f"{args.experiment}_{time.strftime('%Y%m%d-%H%M%S')}"
    rows = []
    for e in sorted(runs, key=lambda e: (e["variant"], e["seed"])):
        imgsz = e["config"]["train"].get("imgsz", 640)
        out = out_root / f"{e['variant']}_s{e['seed']}_f{e['fold']}"
        print(f"\n== {e['variant']} seed {e['seed']} ({e['run_dir']})")
        m = evaluate(Path(e["weights"]["best"]), Path(e["data_yaml"]), "test", out, imgsz=imgsz,
                     device=e.get("device"), conf=args.conf)
        print(summary_line(m))
        row = {"variant": e["variant"], "seed": e["seed"], "val_mAP50_95": e["metrics"]["mAP50_95"],
               **flat(m), "ms_per_image": m["inference"].get("ms_per_image_mean"),
               "train_time_min": round(e.get("train_time_s", 0) / 60, 1)}
        rows.append(row)

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "test_runs.csv", index=False)
    cols = [c for c in ["mAP50", "mAP50_95", "P", "R", "F1", "P_op", "R_op", "F1_op", "F1_op_micro", "ms_per_image"]
            if c in df]
    cls_cols = [c for c in df.columns if c.split("__")[0] in ("F1", "F1_op", "mAP50", "mAP50_95", "P_op", "R_op")
                and "__" in c]
    L = [f"# Avaliação final no TESTE — {args.experiment}", "",
         f"Data: {time.strftime('%Y-%m-%d %H:%M')}. Execuções: {len(df)} "
         f"(seeds {sorted(df['seed'].tolist())}). Limiar do ponto de operação: conf={args.conf}.", ""]
    if history:
        L += [f"> **Atenção:** o teste já tinha sido usado {len(history)} vez(es) antes "
              f"({', '.join(h['when'] for h in history)}).", ""]
    for var, g in df.groupby("variant"):
        L += [f"## {var}", "", "| métrica | média | desvio | min | max |", "|---|---|---|---|---|"]
        for c in cols + sorted(cls_cols):
            v = g[c].astype(float)
            L.append(f"| {c} | {v.mean():.3f} | {v.std(ddof=1) if len(v) > 1 else float('nan'):.3f} | "
                     f"{v.min():.3f} | {v.max():.3f} |")
        pick = g.sort_values("val_mAP50_95", ascending=False).iloc[0]
        L += ["", f"Modelo a entregar (escolhido pela VALIDAÇÃO, não pelo teste): seed {int(pick['seed'])} — "
              f"`{[e for e in runs if e['variant'] == var and e['seed'] == pick['seed']][0]['weights']['best']}`", ""]
    (out_root / "test_report.md").write_text("\n".join(L), encoding="utf-8")
    history.append({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "experiment": args.experiment,
                    "runs": [e["run_dir"] for e in runs], "report": str(out_root / "test_report.md")})
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))
    print(f"\nRelatório: {out_root / 'test_report.md'}")


if __name__ == "__main__":
    main()
