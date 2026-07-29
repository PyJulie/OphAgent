from __future__ import annotations

import os
import tempfile
from pathlib import Path


TEST_RUNTIME = (
    Path(tempfile.gettempdir()) / f"ophagent-tests-{os.getpid()}"
).resolve()

# Tests must never read from or write to a developer's configured runtime.
os.environ["OPHAGENT_RUNTIME_DIR"] = str(TEST_RUNTIME)
os.environ["OPHAGENT_CKPT_DIR"] = str(TEST_RUNTIME / "checkpoints")
os.environ["OPHAGENT_EXTERNAL_DIR"] = str(TEST_RUNTIME / "external")
os.environ["OPHAGENT_OUTPUT_DIR"] = str(TEST_RUNTIME / "reports")
os.environ["OPHAGENT_CACHE_DIR"] = str(TEST_RUNTIME / "cache")
os.environ["OPHAGENT_ENV_FILE"] = str(TEST_RUNTIME / ".env")
