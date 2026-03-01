"""Centralised logging for OphAgent using loguru."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _logger

# Tracks whether the default stderr sink has already been configured.
# Prevents _logger.remove() + re-add on every get_logger() call, which
# would discard previously registered file sinks and flood loguru internals.
_STDERR_SINK_REGISTERED: bool = False


def get_logger(name: str = "ophagent", log_file: Optional[Path] = None):
    """Return a loguru logger bound to *name*.

    A file sink is added on first call if *log_file* is provided or if the
    OPHAGENT log root exists.
    """
    global _STDERR_SINK_REGISTERED

    # Configure the default stderr sink exactly once for the process lifetime.
    if not _STDERR_SINK_REGISTERED:
        _logger.remove()
        _logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG",
        )
        _STDERR_SINK_REGISTERED = True

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            str(log_file),
            rotation="10 MB",
            retention="7 days",
            level="DEBUG",
            encoding="utf-8",
        )

    return _logger.bind(name=name)
