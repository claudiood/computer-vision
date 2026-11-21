"""Gráficos: matriz de confusão, métricas por classe, comparações entre execuções."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#2a6f97", "#e76f51", "#2a9d8f", "#e9c46a", "#8d99ae", "#6d597a", "#f4a261", "#264653"]


def confusion_matrix(matrix, names: list[str], path: Path, title: str, normalize: bool = False) -> None:
    """matrix[predito, verdadeiro] (convenção Ultralytics) com 'fundo' no fim.
    Normalizado por coluna: cada coluna mostra para onde foram os objetos reais daquela classe."""
    m = np.asarray(matrix, float)
    labels = list(names) + ["fundo"]
    shown = m / np.maximum(m.sum(0, keepdims=True), 1e-9) if normalize else m
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2, 1.1 * len(labels) + 1.6))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == len(labels) - 1 and j == len(labels) - 1:
                continue  # fundo×fundo não existe em detecção
            v = shown[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center",
                    color="white" if v > (0.5 if normalize else shown.max() / 2) else "black", fontsize=10)
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("verdadeiro")
    ax.set_ylabel("predito")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def per_class_bars(per_class: dict, keys: list[str], path: Path, title: str) -> None:
    names = list(per_class)
    x = np.arange(len(names))
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 3, 4))
    for i, k in enumerate(keys):
        vals = [per_class[n].get(k) or 0 for n in names]
        b = ax.bar(x + (i - (len(keys) - 1) / 2) * w, vals, w, label=k, color=PALETTE[i % len(PALETTE)])
        ax.bar_label(b, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=len(keys))
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def comparison_bars(labels: list[str], means: list[float], stds: list[float], path: Path, title: str,
                    ylabel: str, ref: float | None = None, ref_band: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels) + 2), 4.2))
    b = ax.bar(labels, means, yerr=[s if s == s else 0 for s in stds], capsize=4, color=PALETTE[0])
    ax.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
    if ref is not None:
        ax.axhline(ref, color="#e76f51", ls="--", lw=1, label="referência")
        if ref_band:
            ax.axhspan(ref - ref_band, ref + ref_band, color="#e76f51", alpha=0.12, label="± ruído (1 desvio)")
        ax.legend(fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    lo = min(m - (s if s == s else 0) for m, s in zip(means, stds)) if means else 0
    ax.set_ylim(max(0, lo - 0.1), min(1.0, max(means) + 0.1) if means else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def grouped_bars(labels: list[str], series: dict[str, list[float]], path: Path, title: str, ylabel: str) -> None:
    x = np.arange(len(labels))
    w = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels) + 2), 4.2))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, [v if v == v else 0 for v in vals], w, label=name,
               color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x, labels, rotation=30)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def scatter_tradeoff(xs, ys, labels, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(xs, ys, s=40, color=PALETTE[0])
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(l, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def draw_predictions(image_path: Path, gt: list, pred: list, names: list[str], path: Path) -> None:
    """GT em verde (tracejado não disponível no PIL: linha fina), previsões em vermelho com confiança."""
    from PIL import Image, ImageDraw

    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, W // 400)
    for c, x1, y1, x2, y2 in gt:
        d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(40, 200, 80), width=lw)
        d.text((x1 * W + 3, y2 * H - 12), f"GT {names[int(c)]}", fill=(40, 200, 80))
    for c, conf, x1, y1, x2, y2 in pred:
        d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(230, 50, 50), width=lw)
        d.text((x1 * W + 3, y1 * H + 2), f"{names[int(c)]} {conf:.2f}", fill=(230, 50, 50))
    im.thumbnail((1280, 1280))
    im.save(path, quality=85)
