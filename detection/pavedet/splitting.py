"""Agrupamento de quase-duplicatas e divisão estratificada por grupo.

Por que agrupar: fotos da mesma cena (tiradas em sequência, ou quase iguais)
não podem cair uma no treino e outra na validação/teste — isso é vazamento
de dados e infla as métricas (manual, seção 3; aula de pré-processamento).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from datetime import timedelta

import numpy as np

from .io_utils import hamming


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


@dataclass
class Item:
    name: str
    annotator: str
    class_counts: np.ndarray  # instâncias por classe
    hash: np.ndarray | None = None
    datetime: object | None = None
    camera: str | None = None
    reasons: list[str] = field(default_factory=list)


_SEQ_RE = re.compile(r"^(?P<prefix>[A-Za-z_]+?)(?P<num>\d{3,})$")


def _seq_key(name: str):
    """'IMG_4701.jpg' -> ('IMG_', 4701). Nomes sem numeração sequencial -> None."""
    m = _SEQ_RE.match(name.rsplit(".", 1)[0])
    return (m["prefix"], int(m["num"])) if m else None


def build_groups(items: list[Item], hash_threshold: int = 70, time_window_s: float = 30,
                 seq_window: int = 0) -> tuple[list[int], list[tuple]]:
    """Retorna (id do grupo de cada item, lista de pares ligados com o motivo).

    Três regras ligam duas fotos ao mesmo grupo (a ligação é transitiva):
    hash_threshold: distância de Hamming máxima entre dHashes (256 bits).
        Pega cópias/reexportações da mesma foto. 0 desliga.
    time_window_s: mesma câmera e menos de N segundos entre as fotos (EXIF).
        Pega fotos do mesmo dano tiradas em sequência, MESMO que tenham sido
        distribuídas para anotadores diferentes. 0 desliga.
    seq_window: nomes sequenciais da câmera (IMG_4701, IMG_4702) a até N números
        de distância. Serve para fotos sem data no EXIF. 0 desliga.
    """
    n = len(items)
    uf = UnionFind(n)
    links = []
    seq = [_seq_key(it.name) for it in items]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]
            why = None
            if hash_threshold and a.hash is not None and b.hash is not None:
                d = hamming(a.hash, b.hash)
                if d <= hash_threshold:
                    why = f"dhash={d}"
            if (not why and time_window_s and a.datetime and b.datetime and a.camera == b.camera
                    and abs(a.datetime - b.datetime) <= timedelta(seconds=time_window_s)):
                why = f"dt={abs((a.datetime - b.datetime).total_seconds()):.0f}s"
            if (not why and seq_window and seq[i] and seq[j] and seq[i][0] == seq[j][0]
                    and abs(seq[i][1] - seq[j][1]) <= seq_window
                    and not (a.datetime and b.datetime)):
                why = f"seq={abs(seq[i][1] - seq[j][1])}"
            if why:
                uf.union(i, j)
                links.append((a.name, b.name, why))
    roots = [uf.find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return [remap[r] for r in roots], links


def _group_features(items, group_ids, annotators):
    ng = max(group_ids) + 1
    nc = len(items[0].class_counts)
    na = len(annotators)
    feats = np.zeros((ng, 1 + nc + na))
    for it, g in zip(items, group_ids):
        feats[g, 0] += 1  # imagens
        feats[g, 1:1 + nc] += it.class_counts
        feats[g, 1 + nc + annotators.index(it.annotator)] += 1
    return feats


def stratified_group_assign(items: list[Item], group_ids: list[int], fractions: list[float],
                            seed: int = 0, restarts: int = 300) -> list[int]:
    """Atribui cada GRUPO a uma partição, aproximando as frações-alvo em
    nº de imagens, instâncias por classe e imagens por anotador.

    Busca gulosa com reinícios aleatórios; retorna a partição de cada item.
    """
    annotators = sorted({it.annotator for it in items})
    feats = _group_features(items, group_ids, annotators)
    total = feats.sum(0)
    fr = np.asarray(fractions, dtype=float) / sum(fractions)
    target = fr[:, None] * total[None, :]
    scale = np.where(total > 0, total, 1.0)
    # classes raras pesam mais: errar 3 instâncias de 'remendo' é pior que 3 de 'rachadura'
    nc = len(items[0].class_counts)
    w = np.ones(feats.shape[1])
    w[1:1 + nc] = 2.0
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    order_base = np.argsort(-(feats[:, 1:1 + nc] / scale[1:1 + nc]).max(1))  # grupos "raros" primeiro
    for r in range(restarts):
        order = order_base.copy() if r == 0 else rng.permutation(len(feats))
        cur = np.zeros_like(target)
        assign = np.empty(len(feats), dtype=int)
        for g in order:
            costs = []
            for s in range(len(fr)):
                trial = cur.copy()
                trial[s] += feats[g]
                costs.append((w * ((trial - target) / scale) ** 2).sum())
            s = int(np.argmin(costs))
            assign[g] = s
            cur[s] += feats[g]
        cost = (w * ((cur - target) / scale) ** 2).sum()
        if cost < best_cost:
            best, best_cost = assign.copy(), cost
    return [int(best[g]) for g in group_ids]
