"""Entry point for the OphAgent Web UI."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ophagent.webchat.server import main

if __name__ == "__main__":
    main()
