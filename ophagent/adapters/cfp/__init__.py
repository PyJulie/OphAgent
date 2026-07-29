"""CFP (Color Fundus Photography) adapters."""
from . import pdr_cascade, od_detection, retsam, glaucoma, eyeq, clip_disease, dr_421, dynamic_clip  # noqa: F401
# Independent English CLIPs — register lazily so a missing weights tree
# doesn't break the rest of the toolkit.
try:
    from . import retizero  # noqa: F401
except Exception as _e:
    import logging
    logging.getLogger(__name__).warning(f"RetiZero adapter not loaded: {_e}")
try:
    from . import flair  # noqa: F401
except Exception as _e:
    import logging
    logging.getLogger(__name__).warning(f"FLAIR adapter not loaded: {_e}")
from . import biomarkers  # noqa: F401  - composite cross-modal tools
# Open-vocabulary zero-shot (paper §S2.4.1) — additive, reuses the CVL CLIP.
# Rollback: delete evidence_zeroshot.py + this import.
try:
    from . import evidence_zeroshot  # noqa: F401
except Exception as _e:
    import logging
    logging.getLogger(__name__).warning(f"openvocab zeroshot adapter not loaded: {_e}")
# EFIQA is optional (large DINOv3 download); register lazily
try:
    from . import efiqa  # noqa: F401
except Exception as _e:
    import logging
    logging.getLogger(__name__).warning(f"EFIQA adapter not loaded: {_e}")
