"""Multimodal model adapters — bridge user's existing trained models without modifying them.

Modality submodules self-register adapters into GLOBAL_REGISTRY via the
@register decorator. Importing this package is enough to populate the
registry.
"""
from .base import AdapterBase, ToolMetadata, AdapterResult, AdapterRegistry, GLOBAL_REGISTRY, register  # noqa
from . import cfp   # noqa: F401
from . import oct   # noqa: F401
from . import uwf   # noqa: F401
# FFA adapter registers but won't load until training finishes.
try:
    from . import ffa   # noqa: F401
except Exception as _e:  # pragma: no cover
    import logging
    logging.getLogger(__name__).warning(f"FFA adapter not registered: {_e}")
# Paired_600 (CFP+FFA 5-class) — register class even if joint weights missing.
try:
    from . import paired   # noqa: F401
except Exception as _e:  # pragma: no cover
    import logging
    logging.getLogger(__name__).warning(f"Paired adapter not registered: {_e}")
# OCT volume-level (g-disc_OCT4) — heavy deps, lazy register.
try:
    from . import oct_volume   # noqa: F401
except Exception as _e:  # pragma: no cover
    import logging
    logging.getLogger(__name__).warning(f"oct_volume adapter not registered: {_e}")
