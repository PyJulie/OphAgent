from __future__ import annotations

import argparse
from pathlib import Path

from .runner import PipelineRequest, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Structured CLI for the OCT segmentation pipelines.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a pipeline profile.")
    run_parser.add_argument("--pipeline", choices=["dicom", "dcm", "fda", "vol_npy", "npy"], required=True)
    run_parser.add_argument("--input", help="Input file or directory. Optional when --manifest contains direct paths.")
    run_parser.add_argument("--manifest", help="Optional CSV/TSV/TXT/XLSX manifest of paths or case IDs.")
    run_parser.add_argument("--output", required=True, help="Output root directory.")
    run_parser.add_argument("--gpu", default="0", help="GPU id passed to legacy scripts.")
    run_parser.add_argument("--target", choices=["disc", "disc2", "macula", "all"], default="all")
    run_parser.add_argument(
        "--segmentation-source",
        choices=["ai", "vendor", "auto", "topcon"],
        help="Segmentation source preference. FDA supports ai/vendor/auto; DCM and NPY are AI only.",
    )
    run_parser.add_argument(
        "--slice-overlays",
        choices=["profile", "on", "off"],
        default="profile",
        help="Override slice overlay export without editing the profile.",
    )
    run_parser.add_argument(
        "--volume-preproc",
        choices=["profile", "on", "off"],
        default="profile",
        help="Override volume-level preprocessing (histogram match / averaging / CLAHE) for segmentation.",
    )
    run_parser.add_argument(
        "--allow-any-shape",
        action="store_true",
        help="For vol_npy runs, accept any 3D shape and resize internally instead of restricting to 512/885 presets.",
    )
    run_parser.add_argument(
        "--rotate-npy",
        choices=["cw", "ccw"],
        help="Rotate vol_npy inputs by 90 degrees before AI segmentation.",
    )
    run_parser.add_argument("--config", help="Optional JSON override merged onto the selected profile.")
    run_parser.add_argument("--dry-run", action="store_true", help="Print commands and write manifests without executing them.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        if not args.input and not args.manifest:
            parser.error("run requires --input or --manifest.")
        request = PipelineRequest(
            pipeline=args.pipeline,
            input_path=Path(args.input).expanduser().resolve() if args.input else None,
            manifest_path=Path(args.manifest).expanduser().resolve() if args.manifest else None,
            output_root=Path(args.output).expanduser().resolve(),
            gpu=str(args.gpu),
            target=args.target,
            segmentation_source=args.segmentation_source,
            slice_overlays=args.slice_overlays,
            volume_preproc=args.volume_preproc,
            allow_any_shape=bool(args.allow_any_shape),
            rotate_npy=args.rotate_npy,
            config_path=Path(args.config).expanduser().resolve() if args.config else None,
            dry_run=bool(args.dry_run),
        )
        manifest = run_pipeline(request)
        print(f"[OK] manifest: {manifest}")


if __name__ == "__main__":
    main()
