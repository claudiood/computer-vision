"""Registro de execuções.

Cada treino tem uma pasta própria e um `run_meta.json` com a configuração
completa, ambiente, tempos, métricas e caminho dos pesos. Além disso, uma linha
é acrescentada em results/registry.jsonl (nunca reescrito: só cresce), que é a
fonte para comparações e para a herança `inherit_best` entre experimentos.

Resultados anteriores nunca são apagados: uma execução já concluída é pulada;
uma execução interrompida é renomeada para `<pasta>__incompleta_<data>` antes
de rodar de novo.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from .config import REGISTRY_FILE, RESULTS_DIR

META = "run_meta.json"


def run_dir_for(experiment: str, variant: str, seed: int, fold) -> Path:
    return RESULTS_DIR / "experiments" / experiment / variant / f"s{seed}_f{fold}"


def read_meta(run_dir: Path) -> dict | None:
    p = Path(run_dir) / META
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_meta(run_dir: Path, meta: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / (META + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tmp.replace(run_dir / META)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def prepare_run_dir(run_dir: Path, force: bool = False) -> str:
    """Decide o que fazer com a pasta da execução.

    Retorna "skip" (já concluída e sem --force), "new" (pasta livre) ou
    "moved" (havia algo incompleto ou --force: a pasta antiga foi preservada com outro nome).
    """
    meta = read_meta(run_dir)
    if meta and meta.get("status") == "completed" and not force:
        return "skip"
    if run_dir.exists() and any(run_dir.iterdir()):
        tag = "anterior" if meta and meta.get("status") == "completed" else "incompleta"
        dst = run_dir.with_name(f"{run_dir.name}__{tag}_{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.move(str(run_dir), str(dst))
        return "moved"
    return "new"


def append_registry(entry: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")


def read_registry(path: Path = REGISTRY_FILE) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    # a mesma pasta pode ter sido registrada mais de uma vez (--force): vale a última
    latest = {}
    for e in out:
        latest[e.get("run_dir")] = e
    return [e for e in latest.values() if e.get("status") == "completed"]


def best_variant(experiment: str, metric: str, entries: list[dict] | None = None) -> tuple[str, dict, list[dict]]:
    """Melhor variante de um experimento pela média de `metric` entre seeds/folds.

    Retorna (nome da variante, config da melhor execução dessa variante, execuções da variante).
    """
    entries = [e for e in (entries if entries is not None else read_registry()) if e.get("experiment") == experiment]
    if not entries:
        raise SystemExit(f"Nenhuma execução concluída do experimento '{experiment}' em {REGISTRY_FILE}. "
                         "Rode esse experimento antes (ou use --set para definir a configuração manualmente).")
    by_var: dict[str, list[dict]] = {}
    for e in entries:
        by_var.setdefault(e["variant"], []).append(e)
    score = {v: float(np.mean([r["metrics"][metric] for r in runs])) for v, runs in by_var.items()}
    best = max(score, key=score.get)
    runs = sorted(by_var[best], key=lambda r: r["metrics"][metric], reverse=True)
    return best, runs[0]["config"], runs


def best_of(experiments: str | list[str], metric: str, entries: list[dict] | None = None,
            margin: float = 0.0):
    """Melhor configuração entre VÁRIOS experimentos, na ordem do pipeline.

    Cada experimento é resumido pela sua melhor variante (média de `metric` entre as execuções).
    Os experimentos são percorridos em ordem: a configuração vigente só é trocada se a nova
    superar a vigente por MAIS de `margin`. Assim, uma diferença menor que o ruído não muda
    a escolha, e fica a opção que já estava (a mais simples/barata, testada antes).

    Retorna (score, experimento, variante, config da melhor execução, execuções da variante)
    ou None se nenhum dos experimentos tem execuções concluídas.
    """
    entries = entries if entries is not None else read_registry()
    best = None
    for name in [experiments] if isinstance(experiments, str) else experiments:
        try:
            var, cfg, runs = best_variant(name, metric, entries)
        except SystemExit:
            continue
        score = float(np.mean([r["metrics"][metric] for r in runs]))
        if best is None or score > best[0] + margin:
            best = (score, name, var, cfg, runs)
    return best
