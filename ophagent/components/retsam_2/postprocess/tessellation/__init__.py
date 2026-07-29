"""Tessellation (豹纹样变) quantification for retsam-2.0."""

from .api import (
    SEVERITY_LABELS,
    TessellationAnalysisConfig,
    analyze_tessellation,
    classify_severity,
)
from .viz import (
    TESS_COLOR,
    TessellationVizConfig,
    render_tessellation_visualizations,
)

__all__ = [
    "analyze_tessellation",
    "classify_severity",
    "TessellationAnalysisConfig",
    "SEVERITY_LABELS",
    "render_tessellation_visualizations",
    "TessellationVizConfig",
    "TESS_COLOR",
]
