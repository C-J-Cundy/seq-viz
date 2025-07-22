"""Sequence Training Visualizer - Real-time visualization for transformer token predictions."""

__version__ = "0.1.0"

# Make integrations available at package level
try:
    from .integrations import VisualizationCallback
    __all__ = ["VisualizationCallback"]
except ImportError:
    # Integrations not available (possibly missing dependencies)
    __all__ = []