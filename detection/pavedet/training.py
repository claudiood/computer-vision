"""Montagem da configuração final de um treino e execução com registro completo."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from .config import DATA_DIR, RESULTS_DIR, deep_merge, load_yaml, resolve_path, save_yaml
from .env import env_info, free_memory, gpu_peak_mb, peak_rss_mb, resolve_device
from .metrics import extract, merge_operating_point, operating_point, read_image_list
from .tracking import append_registry, best_of, prepare_run_dir, read_meta, read_registry, run_dir_for, write_meta

PIPELINE_KEYS = {"dataset", "data", "model", "device", "selection_metric", "op_conf", "oversample", "train"}


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
def resolve_config(exp: dict, variant_cfg: dict | None = None, cli: dict | None = None,
                   allow_missing: bool = False) -> dict:
    """base (extends) → melhor execução de `inherit_best` → overrides do experimento → variante → CLI.

    allow_missing: se o experimento herdado ainda não rodou, segue com a base (só para --dry-run).
    """
    cfg = {k: v for k, v in exp.items() if k in PIPELINE_KEYS}
    inherit = exp.get("inherit_best")
    if inherit:
        # uma lista = escolhe o melhor ENTRE os experimentos listados (ex.: se o Optuna
        # não superou o E1–E4 na ablação, a configuração sem Optuna é a herdada)
        names = [inherit] if isinstance(inherit, str) else list(inherit)
        metric = cfg.get("selection_metric", "mAP50_95")
        entries = read_registry()
        best = best_of(names, metric, entries)
        done = {e.get("experiment") for e in entries}
        missing = [n for n in names if n not in done]
        if missing:
            msg = f"sem execuções concluídas de {missing} no registro (herança: {inherit})"
            if best is None and not allow_missing:
                raise SystemExit(f"Não dá para herdar a configuração: {msg}. Rode esse(s) experimento(s) antes.")
            print(f"AVISO: {msg}" + ("" if best else " — usando a config-base (dry-run)."))
        if best is not None:
            _, name, var, best_cfg, _ = best
            best_cfg = {k: v for k, v in best_cfg.items() if k in PIPELINE_KEYS}
            best_cfg.get("train", {}).pop("seed", None)  # seed e fold vêm do experimento atual
            best_cfg.pop("data", None)
            cfg = deep_merge(cfg, best_cfg)
            cfg["_inherited"] = f"{name}:{var}"
    cfg = deep_merge(cfg, exp.get("overrides") or {})
    cfg = deep_merge(cfg, variant_cfg or {})
    cfg = deep_merge(cfg, cli or {})
    if cfg.get("train", {}).get("epochs") == "cv_median":
        cfg["train"]["epochs"] = cv_median_epochs(allow_missing)
    return cfg


def cv_median_epochs(allow_missing: bool = False) -> int:
    """Épocas do treino definitivo em treino+validação: mediana da melhor época da validação
    cruzada (results/cv/cv_summary.json, gerado por scripts/cv_report.py)."""
    import json

    f = RESULTS_DIR / "cv" / "cv_summary.json"
    if not f.exists():
        if allow_missing:
            print("AVISO: results/cv/cv_summary.json não existe — usando 100 épocas (dry-run).")
            return 100
        raise SystemExit(f"{f} não existe. Rode antes: python scripts/cv_report.py")
    return int(json.loads(f.read_text(encoding="utf-8"))["median_best_epoch"])


def data_yaml_path(cfg: dict) -> Path:
    data = str(cfg.get("data", "fold0"))
    if data.endswith((".yaml", ".yml")):
        return resolve_path(data)
    p = DATA_DIR / cfg["dataset"] / f"data_{data}.yaml"
    if not p.exists():
        raise SystemExit(f"{p} não existe. Rode antes: python scripts/02_prepare_dataset.py")
    return p


# ---------------------------------------------------------------------------
# Desbalanceamento: sobreamostragem de imagens (só no TREINO)
# ---------------------------------------------------------------------------
def oversample_factors(label_files: list[Path], nc: int, spec: dict, seed: int) -> np.ndarray:
    """Quantas vezes cada imagem de treino entra por época.

    method: rfs    Repeat Factor Sampling (Gupta et al., LVIS, 2019):
                   r(c) = max(1, sqrt(t / f(c))), f(c) = fração de imagens com a classe c;
                   cada imagem repete max r(c) das classes que contém (arredondamento estocástico).
    method: fixed  factors: {classe: fator}; imagem repete o maior fator das classes que contém.
    """
    from .io_utils import read_labels

    present = []
    for f in label_files:
        cls = {b.cls for b in read_labels(f, nc).boxes}
        present.append(cls)
    method = spec.get("method", "rfs")
    if method == "rfs":
        t = float(spec.get("threshold", 0.3))
        freq = np.array([sum(c in p for p in present) for c in range(nc)]) / max(len(present), 1)
        rc = np.maximum(1.0, np.sqrt(t / np.maximum(freq, 1e-9)))
    elif method == "fixed":
        rc = np.ones(nc)
        for c, v in (spec.get("factors") or {}).items():
            rc[int(c)] = float(v)
    else:
        raise ValueError(f"oversample.method desconhecido: {method}")
    rng = np.random.default_rng(seed)
    out = []
    for p in present:
        r = max((rc[c] for c in p), default=1.0)
        out.append(int(np.floor(r) + (rng.random() < r - np.floor(r))))
    return np.array(out)


def build_oversampled_yaml(data_yaml: Path, spec: dict, seed: int, run_dir: Path) -> tuple[Path, dict]:
    from .metrics import label_path_for

    d = load_yaml(data_yaml)
    root = Path(d["path"])
    train_list = root / d["train"]
    imgs = read_image_list(train_list)
    nc = len(d["names"])
    reps = oversample_factors([label_path_for(p) for p in imgs], nc, spec, seed)
    lst = run_dir / "train_oversampled.txt"
    run_dir.mkdir(parents=True, exist_ok=True)
    lst.write_text("".join(f"{p}\n" * int(r) for p, r in zip(imgs, reps)), encoding="utf-8")
    new = dict(d)
    new["train"] = str(lst.resolve())
    for k in ("val", "test"):
        if k in new:
            new[k] = str((root / new[k]).resolve())
    new.pop("path", None)
    out = run_dir / "data_oversampled.yaml"
    save_yaml(new, out)
    info = {"images": len(imgs), "entries": int(reps.sum()), "max_repeat": int(reps.max()) if len(reps) else 0}
    return out, info


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def _read_results_csv(run_dir: Path) -> dict:
    f = run_dir / "results.csv"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8") as fh:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
    if not rows:
        return {}
    key = "metrics/mAP50-95(B)"
    vals = [float(r.get(key, "nan") or "nan") for r in rows]
    best = int(np.nanargmax(vals)) if not all(np.isnan(vals)) else len(rows) - 1
    return {"epochs_run": len(rows), "best_epoch": int(float(rows[best].get("epoch", best))) ,
            "train_time_csv_s": float(rows[-1].get("time", 0) or 0)}


def model_complexity(model) -> dict:
    try:
        info = model.info(verbose=False)  # (camadas, parâmetros, gradientes, GFLOPs)
        return {"params_M": round(info[1] / 1e6, 3), "gflops": round(float(info[3]), 2)}
    except Exception:
        return {}


def train_run(cfg: dict, experiment: str, variant: str, force: bool = False, dry_run: bool = False,
              callbacks: dict | None = None, extra_meta: dict | None = None, register: bool = True) -> dict:
    """Treina uma configuração e registra tudo. Retorna o run_meta."""
    seed = int(cfg["train"].get("seed", 0))
    fold = str(cfg.get("data", "fold0")).replace("fold", "") if str(cfg.get("data", "")).startswith("fold") \
        else str(cfg.get("data", "x")).replace("/", "_")
    run_dir = run_dir_for(experiment, variant, seed, fold)
    data_yaml = data_yaml_path(cfg) if not dry_run or (DATA_DIR / cfg["dataset"]).exists() else Path("<data.yaml>")
    device = resolve_device(cfg.get("device"))
    train_args = dict(cfg["train"])
    if dry_run:
        print(f"[dry-run] {experiment}/{variant}  →  {run_dir}")
        print(f"          model={cfg['model']} data={data_yaml} device={device} oversample={cfg.get('oversample')}")
        print(f"          train={train_args}")
        return {"status": "dry-run", "run_dir": str(run_dir), "config": cfg}
    action = prepare_run_dir(run_dir, force)
    if action == "skip":
        print(f"[pula] {run_dir} já concluído (use --force para rodar de novo; o anterior é preservado)")
        return read_meta(run_dir)

    from ultralytics import YOLO

    meta = {"status": "running", "experiment": experiment, "variant": variant, "seed": seed, "fold": fold,
            "run_dir": str(run_dir), "config": cfg, "data_yaml": str(data_yaml), "device": device,
            "env": env_info(), "started": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra_meta or {})}
    write_meta(run_dir, meta)
    try:
        if cfg.get("oversample"):
            data_yaml, info = build_oversampled_yaml(data_yaml, cfg["oversample"], seed, run_dir)
            meta["oversample_info"] = info
        model = YOLO(cfg["model"])
        for event, fn in (callbacks or {}).items():
            model.add_callback(event, fn)
        gpu_peak_mb(reset=True)
        t0 = time.time()
        model.train(data=str(data_yaml), device=device, project=str(run_dir.parent), name=run_dir.name,
                    exist_ok=True, **train_args)
        meta["train_time_s"] = round(time.time() - t0, 1)
        meta["gpu_peak_mb"] = gpu_peak_mb()
        meta.update(_read_results_csv(run_dir))
        best = run_dir / "weights" / "best.pt"
        meta["weights"] = {"best": str(best), "last": str(run_dir / "weights" / "last.pt"),
                           "best_size_mb": round(best.stat().st_size / 1e6, 2) if best.exists() else None}
        # validação explícita com o best.pt (sempre no split 'val': o teste fica guardado)
        bm = YOLO(str(best))
        meta.update(model_complexity(bm))
        imgsz = train_args.get("imgsz", 640)
        dm = bm.val(data=str(data_yaml), split="val", imgsz=imgsz, batch=train_args.get("batch", 16),
                    device=device, project=str(run_dir), name="val_best", exist_ok=True, plots=True, verbose=False)
        names = list(bm.names.values())
        metrics = extract(dm, names)
        d = load_yaml(data_yaml)
        root = Path(d.get("path", ""))
        val_list = Path(d["val"]) if Path(d["val"]).is_absolute() else root / d["val"]
        op, _ = operating_point(bm, read_image_list(val_list), names, conf=cfg.get("op_conf", 0.25),
                                imgsz=imgsz, device=device)
        metrics = merge_operating_point(metrics, op)
        meta["metrics"] = metrics
        meta["peak_rss_mb"] = peak_rss_mb()
        meta["status"] = "completed"
        meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_meta(run_dir, meta)
        if register:
            append_registry(meta)
        m = metrics
        print(f"[ok] {experiment}/{variant} s{seed} f{fold}: mAP50={m['mAP50']:.3f} mAP50-95={m['mAP50_95']:.3f} "
              f"P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} F1_op={m['F1_op']:.3f} "
              f"({meta['train_time_s'] / 60:.1f} min)")
        return meta
    except BaseException as e:
        meta["status"] = "pruned" if type(e).__name__ == "TrialPruned" else "failed"
        meta["error"] = f"{type(e).__name__}: {e}"
        write_meta(run_dir, meta)
        raise
    finally:
        free_memory()
