"""Public DR severity grading entrypoint.

This is the reader-facing path for single-image ICDR 0-4 grading. It wraps the
independent protocol runner with a task-oriented name, explicit architecture
arms, strict final JSON parsing, and per-sample trace persistence.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from ophagent.evaluation import (
    DR_SEVERITY_ICDR_SINGLE_IMAGE,
    EvaluationRunConfig,
    run_image,
)
from ophagent.evaluation.metrics import summarize_grading
from ophagent.chat.prompt_profiles import SUPPORTED_PROFILES


DR_SEVERITY_PROTOCOL = replace(
    DR_SEVERITY_ICDR_SINGLE_IMAGE,
    task_id="dr_severity_icdr_single_image",
    output_schema=DR_SEVERITY_ICDR_SINGLE_IMAGE.output_schema.replace(
        "dr_severity_icdr_single_image_v2",
        "dr_severity_icdr_single_image",
    ),
)


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return stem[:160] or "image"


def _read_manifest(path: Path, image_col: str, grade_col: str,
                   id_col: str | None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        image = row.get(image_col)
        if not image:
            continue
        gt_raw = row.get(grade_col)
        gt_grade = None
        if gt_raw is not None and str(gt_raw).strip().isdigit():
            gt_grade = int(str(gt_raw).strip())
        image_id = row.get(id_col) if id_col else None
        records.append({
            "image_id": image_id or Path(image).stem or f"row_{idx}",
            "image_path": image,
            "gt_grade": gt_grade,
            "source_row": idx,
        })
    return records


def _flatten_result(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    parsed = result.get("parsed_final") or {}
    return {
        "image_id": record.get("image_id"),
        "image_path": record.get("image_path"),
        "gt_grade": record.get("gt_grade"),
        "grade": parsed.get("grade"),
        "confidence": parsed.get("confidence"),
        "gradable": parsed.get("gradable"),
        "evidence_level": parsed.get("evidence_level"),
        "evidence": json.dumps(parsed.get("evidence") or {}, ensure_ascii=False),
        "evidence_gaps": json.dumps(parsed.get("evidence_gaps") or [], ensure_ascii=False),
        "rationale": parsed.get("rationale"),
        "parse_method": parsed.get("parse_method"),
        "parse_error": parsed.get("parse_error"),
        "error": result.get("error"),
        "elapsed_s": result.get("elapsed_s"),
        "n_tool_calls": len(result.get("tool_calls") or []),
        "tool_calls": json.dumps(result.get("tool_calls") or [], ensure_ascii=False),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "image_id", "image_path", "gt_grade", "grade", "confidence", "gradable",
        "evidence_level", "evidence", "evidence_gaps", "rationale",
        "parse_method", "parse_error", "error", "elapsed_s", "n_tool_calls",
        "tool_calls",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_one(record: dict[str, Any], cfg: EvaluationRunConfig, out_dir: Path,
             resume: bool) -> dict[str, Any]:
    cache_path = out_dir / "per_image_json" / f"{_safe_stem(str(record['image_id']))}.json"
    if resume and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("intermediate_results_version", 0) >= 1:
                return cached
        except Exception:
            pass

    result = run_image(
        record["image_path"],
        protocol=DR_SEVERITY_PROTOCOL,
        config=cfg,
    ).to_dict()
    result["image_id"] = record.get("image_id")
    result["gt_grade"] = record.get("gt_grade")
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _write_readme(out_dir: Path, run_config: dict[str, Any]) -> None:
    lines = [
        "# OphAgent DR Severity Grading Run",
        "",
        "This directory was produced by the public task entrypoint:",
        "",
        "```powershell",
        "python -m ophagent.tasks.dr_severity ...",
        "```",
        "",
        "The task is single-image ICDR 0-4 diabetic retinopathy severity grading.",
        "OphAgent chooses tools autonomously inside the declared evidence contract.",
        "",
        "Architecture arms:",
        "",
        "- `full`: planner plus verifier.",
        "- `planner`: planner without verifier.",
        "- `single`: no structured planner and no verifier.",
        "",
        "Outputs:",
        "",
        "- `run_config.json`: backend/model/effort/architecture settings.",
        "- `task_prompt.txt`: exact rendered task prompt.",
        "- `per_image_json/*.json`: full per-sample trace, tool calls, and final JSON.",
        "- `per_image.csv`: parsed per-sample predictions.",
        "- `summary.json`: grading metrics when ground-truth labels are provided.",
        "",
        "Run config:",
        "",
        "```json",
        json.dumps(run_config, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OphAgent public entrypoint for single-image ICDR DR severity grading."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=str, help="single CFP image")
    source.add_argument("--manifest", type=str, help="CSV manifest for a dataset")
    parser.add_argument("--gt-grade", type=int, choices=range(5), default=None)
    parser.add_argument("--image-col", default="image_path")
    parser.add_argument("--grade-col", default="gt_grade")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--backend", default="aigcbest")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--effort", default="medium",
                        choices=["low", "medium", "high", "max", "ultra"])
    parser.add_argument("--architecture-arm", default="full",
                        choices=["single", "planner", "full"])
    parser.add_argument("--prompt-profile", default="standard",
                        choices=SUPPORTED_PROFILES)
    parser.add_argument("--vision-backend", default=None)
    parser.add_argument("--vision-model", default=None)
    parser.add_argument("--native-effort", action="store_true")
    parser.add_argument("--max-tool-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or REPO_ROOT / "reports" / f"ophagent_dr_severity_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_image_json").mkdir(exist_ok=True)

    if args.image:
        records = [{
            "image_id": Path(args.image).stem,
            "image_path": args.image,
            "gt_grade": args.gt_grade,
            "source_row": 0,
        }]
    else:
        records = _read_manifest(Path(args.manifest), args.image_col, args.grade_col, args.id_col)
        if args.limit:
            records = records[:args.limit]

    cfg = EvaluationRunConfig(
        backend=args.backend,
        model=args.model,
        effort=args.effort,
        max_tool_steps=args.max_tool_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        workspace=str(out_dir / "_sessions"),
        vision_backend=args.vision_backend,
        vision_model=args.vision_model,
        native_effort=True if args.native_effort else None,
        architecture_arm=args.architecture_arm,
        prompt_profile=args.prompt_profile,
    )

    run_config = {
        "entrypoint": "python -m ophagent.tasks.dr_severity",
        "protocol": DR_SEVERITY_PROTOCOL.task_id,
        "backend": args.backend,
        "model": args.model,
        "effort": args.effort,
        "architecture_arm": args.architecture_arm,
        "prompt_profile": args.prompt_profile,
        "vision_backend": args.vision_backend,
        "vision_model": args.vision_model,
        "native_effort": bool(args.native_effort),
        "n_records": len(records),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "task_prompt.txt").write_text(
        DR_SEVERITY_PROTOCOL.build_user_prompt(args.effort),
        encoding="utf-8",
    )
    _write_readme(out_dir, run_config)

    print(f"Output dir: {out_dir}")
    print(
        f"Records: {len(records)}  backend={args.backend} model={args.model} "
        f"effort={args.effort} arm={args.architecture_arm}"
    )

    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for idx, record in enumerate(records, 1):
            result = _run_one(record, cfg, out_dir, args.resume)
            results.append(result)
            grade = (result.get("parsed_final") or {}).get("grade")
            print(
                f"[{idx}/{len(records)}] {record['image_id']} "
                f"gt={record.get('gt_grade')} pred={grade} err={result.get('error')}"
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one, r, cfg, out_dir, args.resume): r for r in records}
            for idx, fut in enumerate(as_completed(futures), 1):
                record = futures[fut]
                result = fut.result()
                results.append(result)
                grade = (result.get("parsed_final") or {}).get("grade")
                print(
                    f"[{idx}/{len(records)}] {record['image_id']} "
                    f"gt={record.get('gt_grade')} pred={grade} err={result.get('error')}"
                )

    flat_rows = [
        _flatten_result({
            "image_id": result.get("image_id"),
            "image_path": result.get("image_path"),
            "gt_grade": result.get("gt_grade"),
        }, result)
        for result in results
    ]
    _write_csv(out_dir / "per_image.csv", flat_rows)

    metric_rows = [
        {"gt_grade": row.get("gt_grade"), "grade": row.get("grade")}
        for row in flat_rows
    ]
    summary = summarize_grading(metric_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
