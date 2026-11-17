"""Semantic model router: picks the model profile for each prompt."""

from .intent import IntentClassifier
from .registry import ChampionChallengerRouter, Decision, ModelRegistry, Scorecard, decide, register_kind, score
from .router import SemanticModelRouter, ModelProfile, RoutingDecision

__version__ = "0.1.0"

__all__ = ['IntentClassifier', 'ModelRegistry', 'ChampionChallengerRouter', 'Scorecard', 'Decision', 'decide', 'register_kind', 'score', 'SemanticModelRouter', 'ModelProfile', 'RoutingDecision']
