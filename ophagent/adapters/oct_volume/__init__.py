"""OCT volume-level adapter (Topcon disc analysis from g-disc_OCT4).

Heavy deps (cupy, oct-converter, the external g-disc_OCT4 repo). Registered
lazily so a missing component does not break the rest of the OCT toolkit.
"""

try:
    from . import disc_analysis  # noqa: F401
except Exception as _e:
    import logging
    logging.getLogger(__name__).warning(
        f"oct_volume_disc not loaded (g-disc_OCT4 unavailable): {_e}"
    )

# Macular volume adapter: pure in-repo (Kermany / OCTDL / RETOUCH / Duke-DME
# models we trained ourselves; no external CLI). Register independently of
# the disc adapter so a missing g-disc install doesn't kill macular too.
try:
    from . import macular_volume  # noqa: F401
except Exception as _e:
    import logging
    logging.getLogger(__name__).warning(
        f"oct_volume_macular not loaded: {_e}"
    )

# OCTCubeM 3D foundation model (Liu et al. 2024, BSD-2). Lazy-loaded; failure
# here is non-fatal, so the rest of the OCT toolkit still works.
try:
    from . import octcubem  # noqa: F401
except Exception as _e:
    import logging
    logging.getLogger(__name__).warning(
        f"oct_volume_octcubem not loaded: {_e}"
    )
