"""Avaliação completa de um modelo treinado (usada por evaluate.py e final_test.py)."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .config import load_yaml
from .env import env_info, resolve_device
from .metrics import extract, merge_operating_point, operating_point, read_image_list
from .training import model_complexity


def benchmark_inference(model, images: list[Path], imgsz: int, device: str, n: int = 50, warmup: int = 5) -> dict:
    """Tempo de inferência ponta a ponta (pré + rede + pós) por imagem, batch 1."""
    imgs = [str(p) for p in images][: max(n, 1)]
    if not imgs:
        return {}
    for p in imgs[:warmup]:
        model.predict(p, imgsz=imgsz, device=device, verbose=False)
    times, parts = [], {"preprocess": [], "inference": [], "postprocess": []}
    for p in imgs:
        t0 = time.perf_counter()
        r = model.predict(p, imgsz=imgsz, device=device, verbose=False)[0]
        times.append((time.perf_counter() - t0) * 1000)
        for k in parts:
            parts[k].append(r.speed.get(k, np.nan))
    t = np.array(times)
    return {"device": device, "imgsz": imgsz, "n_images": len(imgs), "batch": 1,
            "ms_per_image_mean": round(float(t.mean()), 2), "ms_per_image_p50": round(float(np.median(t)), 2),
            "ms_per_image_p95": round(float(np.percentile(t, 95)), 2), "fps": round(1000 / float(t.mean()), 1),
            **{f"{k}_ms": round(float(np.nanmean(v)), 2) for k, v in parts.items()}}


def evaluate(weights: Path, data_yaml: Path, split: str, out_dir: Path, imgsz: int = 640, device=None,
             conf: float = 0.25, batch: int = 16, bench_n: int = 50, worst_k: int = 8) -> dict:
    from ultralytics import YOLO

    from . import plots

    device = resolve_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    names = list(model.names.values())
    # 1) métricas independentes de limiar + curvas PR/F1 e matriz (conf=0.001) da Ultralytics
    dm = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, device=device,
                   project=str(out_dir), name="ultralytics", exist_ok=True, plots=True, verbose=False)
    metrics = extract(dm, names)
    # 2) ponto de operação (conf fixo) + análise por imagem
    d = load_yaml(data_yaml)
    lst = Path(d[split]) if Path(d[split]).is_absolute() else Path(d.get("path", "")) / d[split]
    images = read_image_list(lst)
    op, per_image = operating_point(model, images, names, conf=conf, imgsz=imgsz, device=device)
    metrics = merge_operating_point(metrics, op)
    # 3) custo computacional
    metrics["complexity"] = model_complexity(model)
    metrics["weights_size_mb"] = round(Path(weights).stat().st_size / 1e6, 2)
    metrics["inference"] = benchmark_inference(model, images, imgsz, device, n=bench_n)
    metrics["env"] = env_info()
    metrics["split"] = split
    metrics["n_images"] = len(images)
    metrics["weights"] = str(weights)

    # ---- arquivos
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")
    with open(out_dir / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        keys = ["instances", "P", "R", "F1", "mAP50", "mAP50_95", "P_op", "R_op", "F1_op", "TP", "FP", "FN"]
        w = csv.writer(f)
        w.writerow(["classe"] + keys)
        for n, v in metrics["per_class"].items():
            w.writerow([n] + [v.get(k) for k in keys])
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "gt", "pred", "TP", "FP", "FN"])
        for r in sorted(per_image, key=lambda r: -(r["FP"] + r["FN"])):
            w.writerow([r["image"], r["gt"], r["pred"], r["TP"], r["FP"], r["FN"]])
    cm = op["confusion_matrix"]
    plots.confusion_matrix(cm, names, out_dir / f"confusion_matrix_conf{conf}.png",
                           f"Matriz de confusão ({split}, conf={conf}, IoU≥0.5) — contagens")
    plots.confusion_matrix(cm, names, out_dir / f"confusion_matrix_conf{conf}_norm.png",
                           f"Matriz de confusão ({split}, conf={conf}) — por classe verdadeira", normalize=True)
    plots.per_class_bars(metrics["per_class"], ["P", "R", "F1", "mAP50", "mAP50_95"], out_dir / "per_class.png",
                         f"Métricas por classe ({split}; P/R/F1 no conf de melhor F1 = {metrics.get('best_conf')})")
    plots.per_class_bars(metrics["per_class"], ["P_op", "R_op", "F1_op"], out_dir / "per_class_op.png",
                         f"P/R/F1 por classe no conf fixo {conf} ({split})")
    wdir = out_dir / "worst_images"
    wdir.mkdir(exist_ok=True)
    by_path = {p.name: p for p in images}
    for r in sorted(per_image, key=lambda r: -(r["FP"] + r["FN"]))[:worst_k]:
        plots.draw_predictions(by_path[r["image"]], r["gt_boxes"], r["pred_boxes"], names,
                               wdir / f"FP{r['FP']}_FN{r['FN']}_{Path(r['image']).stem}.jpg")
    return metrics


def summary_line(m: dict) -> str:
    inf = m.get("inference", {})
    return (f"mAP50={m['mAP50']:.3f}  mAP50-95={m['mAP50_95']:.3f}  P={m['P']:.3f}  R={m['R']:.3f}  "
            f"F1={m['F1']:.3f} (conf {m.get('best_conf') or 0:.2f})  |  conf={m.get('op_conf')}: "
            f"P={m['P_op']:.3f} R={m['R_op']:.3f} F1={m['F1_op']:.3f}  |  "
            f"{inf.get('ms_per_image_mean', '?')} ms/img ({inf.get('device', '?')})")
