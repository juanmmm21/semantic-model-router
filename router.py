"""Compatibilidad: el código vive en el paquete `semantic_model_router`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `router`
directamente."""

from semantic_model_router.router import *  # noqa: F401,F403
from semantic_model_router.router import SemanticModelRouter, ModelProfile, RoutingDecision  # noqa: F401
