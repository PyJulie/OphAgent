"""
OphAgent FastAPI Service Launcher.

Provides a CLI to start individual model microservices or all services
at once using a process pool.

Usage::
    # Start a specific model service
    python services/fastapi_service.py --model cfp_quality --port 8110

    # Start all pre-loaded services
    python services/fastapi_service.py --all
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path


# Service map: model_id -> (port, module_path, env_vars)
SERVICE_MAP = {
    "cfp_quality":        (8110, "ophagent.models.inference.service", {"MODEL_ID": "cfp_quality",       "MODEL_WEIGHT": "models_weights/cfp_quality/best.pth"}),
    "cfp_disease":        (8111, "ophagent.models.inference.service", {"MODEL_ID": "cfp_disease",        "MODEL_WEIGHT": "models_weights/cfp_disease/best.pth"}),
    "cfp_ffa_multimodal": (8112, "ophagent.models.inference.service", {"MODEL_ID": "cfp_ffa_multimodal", "MODEL_WEIGHT": "models_weights/cfp_ffa_multimodal/best.pth"}),
    "uwf_quality_disease":(8113, "ophagent.models.inference.service", {"MODEL_ID": "uwf_quality_disease","MODEL_WEIGHT": "models_weights/uwf_quality_disease/best.pth"}),
    "cfp_glaucoma":       (8114, "ophagent.models.inference.service", {"MODEL_ID": "cfp_glaucoma",       "MODEL_WEIGHT": "models_weights/cfp_glaucoma/best.pth"}),
    "cfp_pdr":            (8115, "ophagent.models.inference.service", {"MODEL_ID": "cfp_pdr",            "MODEL_WEIGHT": "models_weights/cfp_pdr/best.pth"}),
    "ffa_lesion":         (8120, "ophagent.models.inference.service", {"MODEL_ID": "ffa_lesion",         "MODEL_WEIGHT": "models_weights/ffa_lesion/best.pth"}),
    "disc_fovea":         (8121, "ophagent.models.inference.service", {"MODEL_ID": "disc_fovea",         "MODEL_WEIGHT": "models_weights/disc_fovea/best.pth"}),
    "fundus_expert":      (8101, "ophagent.models.inference.service", {"MODEL_ID": "fundus_expert",      "MODEL_WEIGHT": "models_weights/fundus_expert/"}),
    "vision_unite":       (8102, "ophagent.models.inference.service", {"MODEL_ID": "vision_unite",       "MODEL_WEIGHT": "models_weights/vision_unite/"}),
}


def start_service(model_id: str, port: int, module: str, env_vars: dict, strict: bool = False):
    env = {**os.environ, **env_vars}
    if strict:
        env["OPHAGENT_RUNTIME__MODE"] = "strict"
    cmd = [
        sys.executable, "-m", "uvicorn",
        f"{module}:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--workers", "1",
    ]
    print(f"Starting {model_id} on port {port}...")
    proc = subprocess.Popen(cmd, env=env)
    proc.wait()


def start_all(strict: bool = False):
    processes = []
    for model_id, (port, module, env_vars) in SERVICE_MAP.items():
        p = multiprocessing.Process(
            target=start_service,
            args=(model_id, port, module, env_vars, strict),
            daemon=True,
        )
        p.start()
        processes.append((model_id, p))
        time.sleep(0.5)

    print(f"Started {len(processes)} services.")
    try:
        for _, p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nStopping all services...")
        for _, p in processes:
            p.terminate()


def main():
    parser = argparse.ArgumentParser(description="OphAgent Service Launcher")
    parser.add_argument("--model", type=str, help="Model ID to launch")
    parser.add_argument("--port", type=int, help="Port override")
    parser.add_argument("--all", action="store_true", help="Launch all services")
    parser.add_argument("--strict", action="store_true", help="Require real models; do not allow heuristic fallbacks")
    args = parser.parse_args()

    if args.all:
        start_all(strict=args.strict)
    elif args.model:
        if args.model not in SERVICE_MAP:
            print(f"Unknown model: {args.model}. Available: {list(SERVICE_MAP.keys())}")
            sys.exit(1)
        port, module, env_vars = SERVICE_MAP[args.model]
        if args.port:
            port = args.port
        start_service(args.model, port, module, env_vars, strict=args.strict)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
