from __future__ import annotations

import os
import tempfile
from pathlib import Path


# The suite must give the same result on every machine. Settings inherited from
# the shell that starts pytest (provider keys and endpoints, OphAgent paths and
# switches) are removed from the test process before any OphAgent module reads
# them. A test that needs one of these variables sets it explicitly.
_AMBIENT_PREFIXES = (
    "OPH_",
    "OPHAGENT_",
    "WEB_",
    "RETSAM_",
    "OPENROUTER_",
    "CF_ACCESS_",
)
_AMBIENT_SUFFIXES = ("_API_KEY", "_BASE_URL")
for _name in list(os.environ):
    if (
        _name.startswith(_AMBIENT_PREFIXES)
        or _name.endswith(_AMBIENT_SUFFIXES)
        or _name == "EVAL_TEMPERATURE"
    ):
        del os.environ[_name]

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
