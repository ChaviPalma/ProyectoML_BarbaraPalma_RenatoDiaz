"""Pipeline de ciencia de datos."""
from .pipeline_regresion import create_pipeline

# Crear alias 'pipeline' para mantener consistencia con otros módulos
pipeline = create_pipeline

__all__ = ["pipeline", "create_pipeline"]