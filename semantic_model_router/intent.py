"""Learned intent classifier: multinomial logistic regression over hashed n-grams.

Routing a message (``project`` / ``general`` / ``trivial``, or any label set) with a small model
that answers in about a millisecond on a CPU, instead of asking an LLM. Features: word unigrams
and bigrams and character 3–5-grams of the lower-cased, accent-free text, hashed into 2^18
dimensions with sublinear TF and L2 norm, plus *context tokens* the caller passes (for example
``active_project`` or ``recent_general``) so the same words can route differently in a different
situation. Pure NumPy; trains in seconds on a few thousand examples.

Any object with ``labels``, ``predict_proba(texts, contexts)`` and ``save(directory)`` can be a
router model (see :mod:`semantic_model_router.registry`); this is the built-in kind.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DIM = 1 << 18
KIND = "hashed-logreg"
_WORD = re.compile(r"\w+", re.UNICODE)

Context = Optional[Iterable[str]]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _h(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little") % DIM


def features(text: str, context: Context = None) -> Tuple[np.ndarray, np.ndarray]:
    norm = _normalize(text)
    words = _WORD.findall(norm)
    counts: Dict[str, float] = {}

    def add(key: str, weight: float = 1.0) -> None:
        counts[key] = counts.get(key, 0.0) + weight

    for w in words:
        add("w:" + w)
    for a, b in zip(words, words[1:]):
        add(f"b:{a} {b}")
    squeezed = " ".join(words)
    for n in (3, 4, 5):
        for i in range(len(squeezed) - n + 1):
            add(f"c{n}:{squeezed[i:i + n]}")
    if len(words) <= 3:
        add("len:short")
    for ctx in context or ():
        add("ctx:" + ctx, 2.0)
        for w in words[:6]:  # context x leading words: "sigue" means different things
            add(f"x:{ctx}:{w}")
    merged: Dict[int, float] = {}
    for key, c in counts.items():
        idx = _h(key)
        merged[idx] = merged.get(idx, 0.0) + 1.0 + math.log(c) if c >= 1 else merged.get(idx, 0.0) + c
    idx = np.fromiter(merged.keys(), dtype=np.int64, count=len(merged))
    val = np.fromiter(merged.values(), dtype=np.float64, count=len(merged))
    norm2 = np.linalg.norm(val)
    return idx, (val / norm2 if norm2 > 0 else val)


class _CSR:
    def __init__(self, rows: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        self.indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        self.indptr[1:] = np.cumsum([len(i) for i, _ in rows])
        self.indices = np.concatenate([i for i, _ in rows]) if rows else np.zeros(0, dtype=np.int64)
        self.values = np.concatenate([v for _, v in rows]) if rows else np.zeros(0)
        self.rows = np.repeat(np.arange(len(rows)), np.diff(self.indptr))
        self.n = len(rows)

    def dot(self, W: np.ndarray) -> np.ndarray:
        """``X @ W`` for ``W`` of shape ``[DIM, k]``."""
        out = np.zeros((self.n, W.shape[1]))
        np.add.at(out, self.rows, W[self.indices] * self.values[:, None])
        return out

    def t_dot(self, G: np.ndarray) -> Dict[int, np.ndarray]:
        """``X.T @ G`` restricted to the features present (sparse gradient)."""
        contrib = G[self.rows] * self.values[:, None]
        uniq, inv = np.unique(self.indices, return_inverse=True)
        acc = np.zeros((len(uniq), G.shape[1]))
        np.add.at(acc, inv, contrib)
        return {"idx": uniq, "val": acc}  # type: ignore[dict-item]


class IntentClassifier:
    """Softmax regression; ``labels`` fixes the class order of ``predict_proba``."""

    kind = KIND

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = list(labels)
        self.W = np.zeros((DIM, len(self.labels)))
        self.b = np.zeros(len(self.labels))
        self.meta: Dict[str, object] = {}

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        contexts: Optional[Sequence[Context]] = None,
        epochs: int = 200,
        lr: float = 0.05,
        l2: float = 1e-4,
        balanced: bool = True,
    ) -> "IntentClassifier":
        contexts = contexts or [None] * len(texts)
        X = _CSR([features(t, c) for t, c in zip(texts, contexts)])
        index = {label: i for i, label in enumerate(self.labels)}
        y = np.array([index[label] for label in labels])
        Y = np.eye(len(self.labels))[y]
        counts = np.bincount(y, minlength=len(self.labels)).astype(float)
        weights = (len(y) / (len(self.labels) * np.maximum(counts, 1)))[y] if balanced else np.ones(len(y))
        # Adam on the full-batch gradient, touching only the rows of W that occur in the data.
        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        present = np.unique(X.indices)
        for t in range(1, epochs + 1):
            logits = X.dot(self.W) + self.b
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            G = (P - Y) * weights[:, None] / len(y)
            grad = X.t_dot(G)
            gW = np.zeros((len(present), len(self.labels)))
            pos = np.searchsorted(present, grad["idx"])
            gW[pos] = grad["val"]
            gW += l2 * self.W[present]
            gb = G.sum(axis=0)
            mW[present] = b1 * mW[present] + (1 - b1) * gW
            vW[present] = b2 * vW[present] + (1 - b2) * gW * gW
            mb = b1 * mb + (1 - b1) * gb
            vb = b2 * vb + (1 - b2) * gb * gb
            c1, c2 = 1 - b1**t, 1 - b2**t
            self.W[present] -= lr * (mW[present] / c1) / (np.sqrt(vW[present] / c2) + eps)
            self.b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
        self.meta.update({"train_examples": len(y), "epochs": epochs})
        return self

    def predict_proba(self, texts: Sequence[str], contexts: Optional[Sequence[Context]] = None) -> np.ndarray:
        contexts = contexts or [None] * len(texts)
        X = _CSR([features(t, c) for t, c in zip(texts, contexts)])
        logits = X.dot(self.W) + self.b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, text: str, context: Context = None) -> Tuple[str, float]:
        p = self.predict_proba([text], [context])[0]
        i = int(np.argmax(p))
        return self.labels[i], float(p[i])

    # ------------------------------------------------------------------ persistence

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        rows = np.nonzero(np.any(self.W != 0, axis=1))[0]
        np.savez_compressed(os.path.join(directory, "weights.npz"), rows=rows, W=self.W[rows], b=self.b)
        with open(os.path.join(directory, "model.json"), "w", encoding="utf-8") as fh:
            json.dump({"kind": KIND, "labels": self.labels, "meta": self.meta}, fh, indent=2)

    @classmethod
    def load(cls, directory: str) -> "IntentClassifier":
        with open(os.path.join(directory, "model.json"), encoding="utf-8") as fh:
            info = json.load(fh)
        clf = cls(info["labels"])
        clf.meta = info.get("meta", {})
        data = np.load(os.path.join(directory, "weights.npz"))
        clf.W[data["rows"]] = data["W"]
        clf.b = data["b"]
        return clf
