#!/usr/bin/env python3
"""Avaliação detalhada de um modelo na VALIDAÇÃO (o teste é bloqueado aqui).

Gera em <execução>/eval_val/ (ou --out):
  metrics.json                     todas as métricas (gerais, por classe, custo, ambiente)
  per_class.csv / per_class*.png   P, R, F1, mAP50, mAP50-95 e P/R/F1 no conf fixo, por classe
  confusion_matrix_conf*.png       matriz de confusão no conf fixo (contagens e normalizada)
  ultralytics/                     curvas PR, F1, P, R e matriz da Ultralytics
  per_image.csv, worst_images/     imagens com mais erros (verde = real, vermelho = previsto)
  e o tempo de inferência (ms/imagem, FPS), parâmetros, GFLOPs e tamanho do arquivo.

Uso:
  python scripts/evaluate.py --run results/experiments/02_models/yolo26n/s0_f0
  python scripts/evaluate.py --best-of 03_imgsz              # melhor variante de um experimento
  python scripts/evaluate.py --weights caminho/best.pt --data data/pavement_v2/data_fold0.yaml --imgsz 960
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pavedet.evaluation import evaluate, summary_line  # noqa: E402
from pavedet.tracking import best_variant, read_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="pasta de uma execução (com run_meta.json)")
    src.add_argument("--best-of", help="avalia a melhor execução deste experimento")
    src.add_argument("--weights", help="arquivo .pt (exige --data)")
    ap.add_argument("--data", help="data.yaml (com --weights)")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.25, help="limiar do ponto de operação")
    ap.add_argument("--bench", type=int, default=50, help="imagens no teste de velocidade")
    ap.add_argument("--worst", type=int, default=8, help="quantas piores imagens salvar")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.best_of:
        meta = read_meta(Path(_best_run(args.best_of)))
    elif args.run:
        meta = read_meta(Path(args.run))
        if not meta:
            raise SystemExit(f"{args.run}/run_meta.json não encontrado")
    else:
        if not args.data:
            raise SystemExit("--weights exige --data")
        meta = None
    if meta:
        weights = Path(meta["weights"]["best"])
        data = Path(meta["data_yaml"])
        imgsz = args.imgsz or meta["config"]["train"].get("imgsz", 640)
        device = args.device or meta.get("device")
        out = Path(args.out) if args.out else Path(meta["run_dir"]) / "eval_val"
    else:
        weights, data = Path(args.weights), Path(args.data)
        imgsz, device = args.imgsz or 640, args.device
        out = Path(args.out) if args.out else weights.parent.parent / "eval_val"
    m = evaluate(weights, data, "val", out, imgsz=imgsz, device=device, conf=args.conf, bench_n=args.bench,
                 worst_k=args.worst)
    print(summary_line(m))
    for n, v in m["per_class"].items():
        print(f"   {n:12s} P={v.get('P') or 0:.3f} R={v.get('R') or 0:.3f} F1={v.get('F1') or 0:.3f} "
              f"AP50={v.get('mAP50') or 0:.3f}  | conf fixo: F1={v.get('F1_op', 0):.3f} "
              f"(TP {v.get('TP')}, FP {v.get('FP')}, FN {v.get('FN')})")
    print(f"Saída: {out}")


def _best_run(experiment: str) -> str:
    from pavedet.config import load_experiment, CONFIG_DIR

    metric = load_experiment(CONFIG_DIR / "base_train.yaml").get("selection_metric", "mAP50_95")
    _, _, runs = best_variant(experiment, metric)
    return runs[0]["run_dir"]


if __name__ == "__main__":
    main()
