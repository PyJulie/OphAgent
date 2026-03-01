"""
Local Data Knowledge Source.

Manages two sub-sources:
  1. Image-report archive – past cases (image + clinical report pairs)
  2. Operational standards – clinical guidelines and SOPs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from ophagent.knowledge.vector_store import MultimodalVectorStore
from ophagent.utils.logger import get_logger
from ophagent.utils.text_utils import chunk_text

logger = get_logger("knowledge.local_data")


class ImageReportArchive:
    """
    Indexes image + report pairs from a local directory.

    Expected directory structure::

        local_archive/
            cases/
                case_001/
                    fundus.jpg
                    report.txt
                case_002/
                    ...
    """

    def __init__(self, data_root: Path, vector_store: MultimodalVectorStore):
        self.data_root = Path(data_root)
        self.vs = vector_store
        self._indexed_cases: set = set()

    def index_all(self, force: bool = False, with_metrics: bool = False) -> int:
        """
        Scan the archive directory and index any new cases.

        Args:
            force:        Re-index cases that have already been indexed.
            with_metrics: Also run RetSAM + AutoMorph on each image and index
                          the resulting quantitative metric vectors into the
                          separate metric index.  Requires the tools to be
                          configured and their weights to be available.
        """
        cases_dir = self.data_root / "cases"
        if not cases_dir.exists():
            logger.warning(f"Cases directory not found: {cases_dir}")
            return 0

        cases_root_resolved = cases_dir.resolve()
        count = 0
        for case_dir in sorted(cases_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            # Guard against symlinks that escape the cases directory
            try:
                case_dir.resolve().relative_to(cases_root_resolved)
            except ValueError:
                logger.warning(
                    f"Skipping {case_dir}: resolved path escapes cases directory."
                )
                continue
            case_id = case_dir.name
            if case_id in self._indexed_cases and not force:
                continue

            images = list(case_dir.glob("*.jpg")) + list(case_dir.glob("*.png"))
            report_file = case_dir / "report.txt"
            report = report_file.read_text(encoding="utf-8") if report_file.exists() else ""

            for img_path in images:
                self.vs.add_image(
                    image_path=str(img_path),
                    report=report,
                    metadata={"case_id": case_id, "source": "archive"},
                )
                count += 1

                # Quantitative metric indexing (optional, requires tool weights)
                if with_metrics:
                    metrics = self._compute_metrics(str(img_path))
                    if metrics:
                        self.vs.add_metric_doc(
                            metrics=metrics,
                            metadata={
                                "case_id": case_id,
                                "source": "archive",
                                "image_path": str(img_path),
                                "report": report[:200],
                            },
                        )
                        logger.debug(f"Indexed metric doc for {case_id}: {metrics}")

            if report:
                # Also index report text for text-based retrieval
                for chunk in chunk_text(report, chunk_size=512, chunk_overlap=64):
                    self.vs.add_text(
                        chunk,
                        metadata={"case_id": case_id, "source": "archive_report"},
                    )

            self._indexed_cases.add(case_id)

        logger.info(f"Indexed {count} images from archive.")
        return count

    @staticmethod
    def _compute_metrics(image_path: str) -> Dict[str, float]:
        """
        Run RetSAM and AutoMorph on *image_path* to extract quantitative
        biomarkers for metric-scale indexing.

        Each tool call is fault-tolerant: failure of one tool does not prevent
        the other from contributing its metrics.  An empty dict is returned
        only if both tools fail.
        """
        from ophagent.tools.scheduler import ToolScheduler
        scheduler = ToolScheduler()
        metrics: Dict[str, float] = {}

        # ── RetSAM ──────────────────────────────────────────────────────────
        try:
            out = scheduler.run("retsam", {
                "image_path": image_path,
                "prompt_type": "auto",
            })
            if "area_fraction" in out:
                metrics["area_fraction"] = float(out["area_fraction"])
        except Exception as e:
            logger.warning(f"RetSAM skipped for {image_path}: {e}")

        # ── AutoMorph ───────────────────────────────────────────────────────
        try:
            out = scheduler.run("automorph", {"image_path": image_path})
            m = out.get("metrics", {})
            for key in ("fractal_dim", "vessel_density", "av_ratio", "cup_disc_ratio"):
                if key in m:
                    metrics[key] = float(m[key])
        except Exception as e:
            logger.warning(f"AutoMorph skipped for {image_path}: {e}")

        return metrics


class OperationalStandards:
    """
    Indexes clinical guidelines, SOPs, and standard reference documents.

    Supports PDF and plain-text files.
    """

    def __init__(self, data_root: Path, vector_store: MultimodalVectorStore):
        self.data_root = Path(data_root)
        self.vs = vector_store
        self._indexed_files: set = set()

    def index_all(self, force: bool = False) -> int:
        standards_dir = self.data_root / "standards"
        if not standards_dir.exists():
            logger.warning(f"Standards directory not found: {standards_dir}")
            return 0

        count = 0
        for f in standards_dir.rglob("*"):
            if f.suffix.lower() not in (".txt", ".md", ".pdf"):
                continue
            if str(f) in self._indexed_files and not force:
                continue

            text = self._read_file(f)
            if not text:
                continue

            for chunk in chunk_text(text, chunk_size=512, chunk_overlap=64):
                self.vs.add_text(
                    chunk,
                    metadata={"source": "standard", "filename": f.name},
                )
                count += 1

            self._indexed_files.add(str(f))

        logger.info(f"Indexed {count} chunks from operational standards.")
        return count

    @staticmethod
    def _read_file(path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            try:
                import PyPDF2
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    return "\n".join(
                        page.extract_text() or "" for page in reader.pages
                    )
            except Exception as e:
                logger.warning(f"Failed to parse PDF {path}: {e}")
                return ""
        return path.read_text(encoding="utf-8", errors="replace")


class LocalDataSource:
    """Unified local data manager combining archive + standards."""

    def __init__(self, vector_store: Optional[MultimodalVectorStore] = None):
        from config.settings import get_settings
        cfg = get_settings()
        self.vs = vector_store or MultimodalVectorStore()
        self.archive = ImageReportArchive(cfg.knowledge_base.local_data_root, self.vs)
        self.standards = OperationalStandards(cfg.knowledge_base.local_data_root, self.vs)

    def build(self, force: bool = False, with_metrics: bool = False) -> None:
        """
        Index all local data sources.

        Args:
            force:        Re-index already-indexed sources.
            with_metrics: Run RetSAM + AutoMorph on archive images to also
                          populate the quantitative metric index.
        """
        n_archive = self.archive.index_all(force=force, with_metrics=with_metrics)
        n_standards = self.standards.index_all(force=force)
        self.vs.save()
        logger.info(f"Local data built: {n_archive} images, {n_standards} standard chunks.")

    def retrieve(self, query: str, top_k: int = 5) -> str:
        return self.vs.retrieve_text(query, top_k=top_k)
