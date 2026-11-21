"""Extração de métricas da Ultralytics (DetMetrics) num dicionário plano e estável.

Duas famílias de P/R/F1 são registradas, porque respondem perguntas diferentes:

* `P`, `R`, `F1` (e por classe): no limiar de confiança que maximiza o F1 médio
  — é o que a Ultralytics imprime. Mede o melhor ponto de operação possível.
* `P_op`, `R_op`, `F1_op`: num limiar FIXO (padrão 0.25), calculados por
  `operating_point()` a partir das previsões de `predict()` e de uma matriz de
  confusão própria. É o desempenho que o usuário vê com `predict(conf=0.25)` e
  inclui confusões entre classes. (A matriz de confusão do `val()` da
  Ultralytics usa conf=0.001, o limiar do cálculo do mAP, e não serve para isso.)

mAP50 e mAP50-95 completam o quadro, por não dependerem de limiar.
"""
from __future__ import annotations

import numpy as np


def _f1(p, r):
    p, r = np.asarray(p, float), np.asarray(r, float)
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)


def confusion_prf(matrix: np.ndarray) -> dict:
    """P/R/F1 por classe a partir da matriz da Ultralytics: matrix[predito, verdadeiro],
    com a última linha/coluna = fundo (background)."""
    m = np.asarray(matrix, float)
    nc = m.shape[0] - 1
    tp = np.diag(m)[:nc]
    pred_tot = m[:nc, :].sum(1)  # tudo que foi previsto como c (inclui FP de fundo e de outra classe)
    true_tot = m[:, :nc].sum(0)  # todos os objetos reais da classe c (inclui não detectados)
    p = np.where(pred_tot > 0, tp / np.maximum(pred_tot, 1e-12), 0.0)
    r = np.where(true_tot > 0, tp / np.maximum(true_tot, 1e-12), 0.0)
    return {"p": p, "r": r, "f1": _f1(p, r), "tp": tp, "fp": pred_tot - tp, "fn": true_tot - tp,
            "fp_background": m[:nc, nc], "fn_background": m[nc, :nc]}


def extract(dm, names: list[str] | dict) -> dict:
    """Converte o retorno de `model.val()` (DetMetrics) em dicionário."""
    names = list(names.values()) if isinstance(names, dict) else list(names)
    nc = len(names)
    box = dm.box
    out = {"P": float(box.mp), "R": float(box.mr), "mAP50": float(box.map50), "mAP50_95": float(box.map),
           "fitness": float(dm.fitness)}
    f1 = np.asarray(getattr(box, "f1", []), float)
    out["F1"] = float(f1.mean()) if f1.size else float(_f1(box.mp, box.mr))
    # limiar de confiança que maximiza o F1 médio
    try:
        px, f1c = np.asarray(box.px), np.asarray(box.f1_curve)
        out["best_conf"] = float(px[int(f1c.mean(0).argmax())])
    except Exception:
        out["best_conf"] = None
    idx = [int(i) for i in getattr(box, "ap_class_index", [])]
    nt = getattr(dm, "nt_per_class", None)
    per = {}
    for c, n in enumerate(names):
        d = {"instances": int(nt[c]) if nt is not None and c < len(nt) else None}
        if c in idx:
            k = idx.index(c)
            d.update(P=float(box.p[k]), R=float(box.r[k]), F1=float(f1[k]) if f1.size else None,
                     mAP50=float(box.ap50[k]), mAP50_95=float(box.ap[k]))
        else:  # classe sem instâncias nesta partição
            d.update(P=None, R=None, F1=None, mAP50=None, mAP50_95=None)
        per[n] = d
    out["per_class"] = per
    out["speed_ms"] = {k: round(float(v), 3) for k, v in (getattr(dm, "speed", {}) or {}).items()}
    return out


def flat(metrics: dict, prefix: str = "") -> dict:
    """Achata métricas (inclui por classe) para tabelas: F1__rachadura, ..."""
    row = {f"{prefix}{k}": v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    for cls, d in (metrics.get("per_class") or {}).items():
        for k, v in d.items():
            row[f"{prefix}{k}__{cls}"] = v
    for k, v in (metrics.get("speed_ms") or {}).items():
        row[f"{prefix}speed_{k}_ms"] = v
    return row


# ---------------------------------------------------------------------------
# Ponto de operação: previsões com conf fixo, casadas com os rótulos reais
# ---------------------------------------------------------------------------
def _xywhn_to_xyxy(b: np.ndarray) -> np.ndarray:
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-12)


def match_image(pred_cls, pred_conf, pred_xyxy, gt_cls, gt_xyxy, nc: int, iou_thr: float = 0.5) -> np.ndarray:
    """Matriz de confusão (nc+1 x nc+1, [predito, verdadeiro], último índice = fundo) de UMA imagem.

    Casamento guloso um-para-um, por confiança decrescente e sem olhar a classe
    (o mesmo critério da Ultralytics), para que a confusão entre classes apareça.
    """
    m = np.zeros((nc + 1, nc + 1), dtype=int)
    order = np.argsort(-np.asarray(pred_conf))
    iou = box_iou(np.asarray(pred_xyxy).reshape(-1, 4), np.asarray(gt_xyxy).reshape(-1, 4))
    claimed = np.zeros(len(gt_cls), dtype=bool)
    for i in order:
        j = -1
        if len(gt_cls):
            cand = np.where(~claimed & (iou[i] >= iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
        if j >= 0:
            claimed[j] = True
            m[int(pred_cls[i]), int(gt_cls[j])] += 1
        else:
            m[int(pred_cls[i]), nc] += 1  # FP: previsto onde não há objeto
    for j in np.where(~claimed)[0]:
        m[nc, int(gt_cls[j])] += 1  # FN: objeto não encontrado
    return m


def label_path_for(image_path) -> "Path":
    """Convenção YOLO: .../images/x.jpg -> .../labels/x.txt (última ocorrência de /images/)."""
    import os
    from pathlib import Path

    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    p = str(image_path)
    return Path(sb.join(p.rsplit(sa, 1))).with_suffix(".txt")


def read_image_list(list_file) -> list:
    """Lê uma lista de imagens da Ultralytics (entradas './' relativas à pasta da lista)."""
    from pathlib import Path

    list_file = Path(list_file)
    out = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            p = list_file.parent / line[2:] if line.startswith("./") else Path(line)
            out.append(p.resolve())
    return out


def operating_point(model, image_paths: list, names: list[str], conf: float = 0.25, iou_thr: float = 0.5,
                    imgsz: int = 640, device=None, batch: int = 8) -> tuple[dict, list[dict]]:
    """Roda predict(conf) nas imagens e calcula P/R/F1 por classe e a matriz de confusão.

    Os rótulos são lidos trocando /images/ por /labels/ (convenção YOLO).
    Retorna (métricas, lista por imagem com TP/FP/FN para análise de erros).
    """
    from pathlib import Path

    from .io_utils import read_labels

    nc = len(names)
    total = np.zeros((nc + 1, nc + 1), dtype=int)
    per_image = []
    paths = [str(p) for p in image_paths]
    for k in range(0, len(paths), batch):
        chunk = paths[k:k + batch]
        results = model.predict(chunk, conf=conf, imgsz=imgsz, device=device, verbose=False)
        for path, r in zip(chunk, results):
            lbl = label_path_for(path)
            gt = read_labels(lbl, nc).boxes
            g_cls = np.array([b.cls for b in gt], dtype=int)
            g_xy = _xywhn_to_xyxy(np.array([[b.x, b.y, b.w, b.h] for b in gt]).reshape(-1, 4))
            bx = r.boxes
            p_cls = bx.cls.cpu().numpy().astype(int)
            p_conf = bx.conf.cpu().numpy()
            p_xy = bx.xyxyn.cpu().numpy()
            m = match_image(p_cls, p_conf, p_xy, g_cls, g_xy, nc, iou_thr)
            total += m
            tp = int(np.trace(m[:nc, :nc]))
            per_image.append({"image": Path(path).name, "gt": len(gt), "pred": len(p_cls), "TP": tp,
                              "FP": int(m[:nc, :].sum() - tp), "FN": int(m[:, :nc].sum() - tp),
                              "pred_boxes": np.column_stack([p_cls, p_conf, p_xy]).tolist() if len(p_cls) else [],
                              "gt_boxes": np.column_stack([g_cls, g_xy]).tolist() if len(g_cls) else []})
    op = confusion_prf(total)
    out = {"conf": conf, "iou": iou_thr, "P_op": float(op["p"].mean()), "R_op": float(op["r"].mean()),
           "F1_op": float(op["f1"].mean()), "confusion_matrix": total.tolist(), "per_class": {}}
    # micro (todas as caixas juntas): útil com classes desbalanceadas
    tp, fp, fn = op["tp"].sum(), op["fp"].sum(), op["fn"].sum()
    out["P_op_micro"] = float(tp / max(tp + fp, 1))
    out["R_op_micro"] = float(tp / max(tp + fn, 1))
    out["F1_op_micro"] = float(_f1(out["P_op_micro"], out["R_op_micro"]))
    for c, n in enumerate(names):
        out["per_class"][n] = {"P_op": float(op["p"][c]), "R_op": float(op["r"][c]), "F1_op": float(op["f1"][c]),
                               "TP": int(op["tp"][c]), "FP": int(op["fp"][c]), "FN": int(op["fn"][c])}
    return out, per_image


def merge_operating_point(metrics: dict, op: dict) -> dict:
    """Junta as métricas do ponto de operação no dicionário de `extract()`."""
    for k in ("P_op", "R_op", "F1_op", "P_op_micro", "R_op_micro", "F1_op_micro"):
        metrics[k] = op[k]
    metrics["op_conf"] = op["conf"]
    for n, d in op["per_class"].items():
        metrics["per_class"].setdefault(n, {}).update(d)
    metrics["confusion_matrix_op"] = op["confusion_matrix"]
    return metrics
