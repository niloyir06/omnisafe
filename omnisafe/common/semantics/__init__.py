"""Semantic guidance components (CLIP embeddings, risk head, shaping).

All features are optional and activated via semantic_cfgs in algorithm config.
"""

from .semantic_manager import SemanticManager  # noqa: F401
from .risk_head import SemanticRiskHead  # noqa: F401
