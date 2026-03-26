"""Centralised logging for OphAgent using loguru."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from loguru import logger as _logger
except ImportError:  # pragma: no cover - exercised in lean environments
    _logger = None

# Tracks whether the default stderr sink has already been configured.
# Prevents _logger.remove() + re-add on every get_logger() call, which
# would discard previously registered file sinks and flood loguru internals.
_STDERR_SINK_REGISTERED: bool = False
_STD_LOGGING_CONFIGURED: bool = False


def get_logger(name: str = "ophagent", log_file: Optional[Path] = None):
    """Return a loguru logger bound to *name*.

    A file sink is added on first call if *log_file* is provided or if the
    OPHAGENT log root exists.
    """
    global _STDERR_SINK_REGISTERED, _STD_LOGGING_CONFIGURED

    if _logger is None:
        if not _STD_LOGGING_CONFIGURED:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
            )
            _STD_LOGGING_CONFIGURED = True
        std_logger = logging.getLogger(name)
        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handler_exists = any(
                isinstance(handler, logging.FileHandler)
                and Path(handler.baseFilename) == log_file
                for handler in std_logger.handlers
            )
            if not handler_exists:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
                    )
                )
                std_logger.addHandler(file_handler)
        return std_logger

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
