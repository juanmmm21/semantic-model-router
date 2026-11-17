"""Champion/challenger: a registry of router models where a new one is promoted only if it wins.

Layout of a registry directory::

    registry/
      champion.json            {"version": "v3", "promoted_at": ...}
      promotions.jsonl         one line per decision (promoted or rejected), with both scorecards
      models/v1/model.json     {"kind": "hashed-logreg", "labels": [...], ...} + the model's files
      models/v2/...

``register`` stores a candidate (a *challenger*). ``evaluate`` scores any version on a labelled
evaluation set. ``promote_if_better`` scores champion and challenger on **the same** set and
promotes the challenger only if

* its macro-F1 beats the champion's by at least ``min_gain``, and
* no class loses more than ``max_class_drop`` of recall (a better average that breaks one route
  is not better), and
* its p95 latency per message stays under ``max_latency_ms`` (when given).

Every decision is appended to ``promotions.jsonl`` with both scorecards, so the history of what
was promoted and why is auditable. The first model registered when there is no champion becomes
champion directly (there is nothing to beat).

Model kinds: the registry loads ``hashed-logreg`` (:class:`IntentClassifier`) itself. Other kinds
(for instance a LoRA-tuned transformer from llm-qlora-finetuner) are plugged in by whoever
combines the components, with ``register_kind`` — this package never imports them.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .intent import KIND as HASHED_KIND
from .intent import Context, IntentClassifier

Loader = Callable[[str], Any]
_LOADERS: Dict[str, Loader] = {HASHED_KIND: IntentClassifier.load}


def register_kind(kind: str, loader: Loader) -> None:
    """Teach every registry to load models of ``kind`` from their directory."""
    _LOADERS[kind] = loader


def load_model(directory: str) -> Any:
    with open(os.path.join(directory, "model.json"), encoding="utf-8") as fh:
        kind = json.load(fh).get("kind", HASHED_KIND)
    if kind not in _LOADERS:
        raise KeyError(f"no loader registered for model kind {kind!r} (use register_kind)")
    return _LOADERS[kind](directory)


@dataclass
class Scorecard:
    """How a model did on an evaluation set."""

    version: str
    n: int
    accuracy: float
    macro_f1: float
    per_class: Dict[str, Dict[str, float]]
    latency_ms_p50: float
    latency_ms_p95: float
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class Decision:
    promoted: bool
    champion: Optional[Scorecard]
    challenger: Scorecard
    reasons: List[str]
    at: float = field(default_factory=time.time)


def score(model: Any, texts: Sequence[str], labels: Sequence[str], contexts: Optional[Sequence[Context]] = None, version: str = "") -> Scorecard:
    """Accuracy, macro-F1, per-class precision/recall and per-message latency of ``model``."""
    contexts = list(contexts) if contexts is not None else [None] * len(texts)
    preds: List[str] = []
    times: List[float] = []
    for text, ctx in zip(texts, contexts):
        start = time.perf_counter()
        p = model.predict_proba([text], [ctx])[0]
        times.append((time.perf_counter() - start) * 1000)
        preds.append(model.labels[int(np.argmax(p))])
    classes = sorted(set(labels) | set(preds))
    per_class: Dict[str, Dict[str, float]] = {}
    confusion = {g: {p: 0 for p in classes} for g in classes}
    for g, p in zip(labels, preds):
        confusion[g][p] += 1
    f1s = []
    for c in sorted(set(labels)):
        tp = confusion[c][c]
        fp = sum(confusion[g][c] for g in classes if g != c)
        fn = sum(confusion[c][p] for p in classes if p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        f1s.append(f1)
    acc = sum(1 for g, p in zip(labels, preds) if g == p) / max(1, len(labels))
    return Scorecard(
        version=version,
        n=len(labels),
        accuracy=acc,
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        per_class=per_class,
        latency_ms_p50=float(np.percentile(times, 50)) if times else 0.0,
        latency_ms_p95=float(np.percentile(times, 95)) if times else 0.0,
        confusion=confusion,
    )


def decide(
    champion: Optional[Scorecard],
    challenger: Scorecard,
    min_gain: float = 0.0,
    max_class_drop: float = 0.02,
    max_latency_ms: Optional[float] = None,
) -> "tuple[bool, List[str]]":
    """The promotion rule: wins on macro-F1 by more than ``min_gain``, loses no more than
    ``max_class_drop`` of recall in any class, and stays under the latency budget."""
    if champion is None:
        return True, ["no champion to beat"]
    reasons: List[str] = []
    gain = challenger.macro_f1 - champion.macro_f1
    if gain <= min_gain:
        reasons.append(f"macro-F1 {challenger.macro_f1:.4f} does not beat {champion.macro_f1:.4f} by more than {min_gain}")
    for c, stats in champion.per_class.items():
        drop = stats["recall"] - challenger.per_class.get(c, {"recall": 0.0})["recall"]
        if drop > max_class_drop:
            reasons.append(f"recall of {c!r} drops {drop:.3f} (> {max_class_drop})")
    if max_latency_ms is not None and challenger.latency_ms_p95 > max_latency_ms:
        reasons.append(f"p95 latency {challenger.latency_ms_p95:.2f} ms > {max_latency_ms} ms")
    if reasons:
        return False, reasons
    return True, [f"macro-F1 +{gain:.4f} with no class regression"]


class ModelRegistry:
    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(os.path.join(root, "models"), exist_ok=True)

    # ------------------------------------------------------------------ versions

    def versions(self) -> List[str]:
        return sorted(os.listdir(os.path.join(self.root, "models")), key=lambda v: (len(v), v))

    def path(self, version: str) -> str:
        return os.path.join(self.root, "models", version)

    def register(self, model: Any, info: Optional[Dict[str, Any]] = None, version: Optional[str] = None) -> str:
        """Store ``model`` (via its ``save``) as a new version; returns the version name."""
        version = version or f"v{len(self.versions()) + 1}"
        target = self.path(version)
        if os.path.exists(target):
            raise FileExistsError(f"version {version} already exists")
        model.save(target)
        with open(os.path.join(target, "registry.json"), "w", encoding="utf-8") as fh:
            json.dump({"version": version, "registered_at": time.time(), **(info or {})}, fh, indent=2)
        if self.champion_version() is None:
            self._set_champion(version, reason="first model")
        return version

    def load(self, version: str) -> Any:
        return load_model(self.path(version))

    def info(self, version: str) -> Dict[str, Any]:
        path = os.path.join(self.path(version), "registry.json")
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------ champion

    def champion_version(self) -> Optional[str]:
        path = os.path.join(self.root, "champion.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)["version"]

    def champion(self) -> Any:
        version = self.champion_version()
        if version is None:
            raise LookupError("the registry has no champion yet")
        return self.load(version)

    def _set_champion(self, version: str, reason: str) -> None:
        tmp = os.path.join(self.root, "champion.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"version": version, "promoted_at": time.time(), "reason": reason}, fh)
        os.replace(tmp, os.path.join(self.root, "champion.json"))

    def promote_if_better(
        self,
        challenger: str,
        texts: Sequence[str],
        labels: Sequence[str],
        contexts: Optional[Sequence[Context]] = None,
        min_gain: float = 0.0,
        max_class_drop: float = 0.02,
        max_latency_ms: Optional[float] = None,
    ) -> Decision:
        """Score champion and challenger here and apply :func:`decide`."""
        champion = self.champion_version()
        challenger_card = score(self.load(challenger), texts, labels, contexts, challenger)
        champion_card = (
            score(self.load(champion), texts, labels, contexts, champion) if champion and champion != challenger else None
        )
        return self.apply(challenger, champion_card, challenger_card, min_gain, max_class_drop, max_latency_ms)

    def apply(
        self,
        challenger: str,
        champion_card: Optional[Scorecard],
        challenger_card: Scorecard,
        min_gain: float = 0.0,
        max_class_drop: float = 0.02,
        max_latency_ms: Optional[float] = None,
    ) -> Decision:
        """Decide with scorecards computed elsewhere (e.g. by an evaluation harness on the same
        evaluation set), record the decision and move the champion if it won."""
        if champion_card is None and self.champion_version() == challenger:
            decision = Decision(False, None, challenger_card, ["already champion"])
        else:
            promoted, reasons = decide(champion_card, challenger_card, min_gain, max_class_drop, max_latency_ms)
            decision = Decision(promoted, champion_card, challenger_card, reasons)
        if decision.promoted:
            self._set_champion(challenger, reason="; ".join(decision.reasons))
        with open(os.path.join(self.root, "promotions.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(decision)) + "\n")
        return decision

    def history(self) -> List[Dict[str, Any]]:
        path = os.path.join(self.root, "promotions.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def remove(self, version: str) -> None:
        if version == self.champion_version():
            raise ValueError("cannot remove the champion")
        shutil.rmtree(self.path(version))


class ChampionChallengerRouter:
    """Routes with the champion; optionally runs the challenger in shadow and counts agreement,
    so a candidate's behaviour on real traffic is known before any promotion."""

    def __init__(self, registry: ModelRegistry, challenger: Optional[str] = None) -> None:
        self.registry = registry
        self.champion_version = registry.champion_version()
        self.model = registry.champion()
        self.challenger_version = challenger
        self.challenger = registry.load(challenger) if challenger else None
        self.shadow_total = 0
        self.shadow_agree = 0

    def route(self, text: str, context: Context = None) -> Dict[str, Any]:
        start = time.perf_counter()
        p = self.model.predict_proba([text], [list(context or [])])[0]
        i = int(np.argmax(p))
        out: Dict[str, Any] = {
            "label": self.model.labels[i],
            "confidence": float(p[i]),
            "version": self.champion_version,
            "latency_ms": (time.perf_counter() - start) * 1000,
        }
        if self.challenger is not None:
            q = self.challenger.predict_proba([text], [list(context or [])])[0]
            label = self.challenger.labels[int(np.argmax(q))]
            self.shadow_total += 1
            self.shadow_agree += int(label == out["label"])
            out["challenger_label"] = label
        return out

    @property
    def shadow_agreement(self) -> Optional[float]:
        return self.shadow_agree / self.shadow_total if self.shadow_total else None
