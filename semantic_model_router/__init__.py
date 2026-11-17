"""Semantic model router: picks the model profile for each prompt."""

from .router import SemanticModelRouter, ModelProfile, RoutingDecision

__version__ = "0.1.0"

__all__ = ['SemanticModelRouter', 'ModelProfile', 'RoutingDecision']
