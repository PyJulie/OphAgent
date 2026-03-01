"""
Multi-Scale RAG with Unlabelled Local Databases (Section 2.4.3).

Innovation: Standard RAG uses only text queries.  OphAgent extends this with
a multi-scale strategy that retrieves at four granularities:
  1. Global (patient-level):       full image embedding vs. case-level image embeddings
  2. Regional (structure-level):   cropped ROI embedding vs. region embeddings
  3. Textual:                      disease/finding query text vs. report/guideline text
  4. Quantitative metric (NEW):    biomarker vector (area_fraction, fractal_dim,
                                   vessel_density, av_ratio, cup_disc_ratio) via
                                   RetSAM + AutoMorph, searched against a separate
                                   metric index for biomarker-proximity retrieval

This enables retrieval from an unlabelled image archive — no text labels needed —
by using CLIP image embeddings for direct image-to-image similarity search, AND
by matching quantitative morphological profiles across cases.

Architecture:
  - For each scale, retrieve top-K candidates.
  - Fuse with weighted re-ranking (weights sum to 1.0).
  - Return the combined context string.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ophagent.knowledge.vector_store import Document, MultimodalVectorStore
from ophagent.utils.logger import get_logger

logger = get_logger("strategies.multiscale_rag")


@dataclass
class ScaleConfig:
    name: str
    top_k: int
    weight: float         # relative weight for score fusion
    query_type: str       # "text" | "image" | "metric"
    crop_region: Optional[str] = None  # tool_id to call for ROI, or None for full image


class MultiScaleRAG:
    """
    Multi-scale retrieval-augmented generation for ophthalmic images.

    Usage::

        rag = MultiScaleRAG(vector_store=kb.vs)
        context = rag.retrieve(
            query="findings in macula region",
            image_path="fundus.jpg",
            image_modality="CFP",
        )
    """

    DEFAULT_SCALES = [
        ScaleConfig("global_image",        top_k=3, weight=0.40, query_type="image"),
        ScaleConfig("regional_image",      top_k=2, weight=0.25, query_type="image", crop_region="disc_fovea"),
        ScaleConfig("text",                top_k=5, weight=0.20, query_type="text"),
        ScaleConfig("quantitative_metric", top_k=2, weight=0.15, query_type="metric"),  # NEW
    ]

    def __init__(
        self,
        vector_store: Optional[MultimodalVectorStore] = None,
        scales: Optional[List[ScaleConfig]] = None,
        scheduler=None,
    ):
        self.vs = vector_store or MultimodalVectorStore()
        self.scales = scales or self.DEFAULT_SCALES
        self._scheduler = scheduler

    @property
    def scheduler(self):
        if self._scheduler is None:
            from ophagent.tools.scheduler import ToolScheduler
            self._scheduler = ToolScheduler()
        return self._scheduler

    # ------------------------------------------------------------------
    # Main retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        image_path: Optional[str] = None,
        image_modality: str = "CFP",
        top_k_final: int = 5,
    ) -> str:
        """
        Retrieve relevant context across all configured scales.

        Args:
            query:          Text query (clinical question or finding description).
            image_path:     Path to the query image (enables image-based retrieval).
            image_modality: Modality string ("CFP", "OCT", "UWF", "FFA").
            top_k_final:    Number of final documents to return after fusion.

        Returns:
            Formatted context string for LLM injection.
        """
        all_docs: Dict[str, Tuple[Document, float]] = {}  # doc_id -> (doc, fused_score)

        for scale in self.scales:
            results = self._retrieve_at_scale(scale, query, image_path)
            for doc, score in results:
                weighted_score = score * scale.weight
                if doc.doc_id in all_docs:
                    _, existing = all_docs[doc.doc_id]
                    all_docs[doc.doc_id] = (doc, existing + weighted_score)
                else:
                    all_docs[doc.doc_id] = (doc, weighted_score)

        # Sort by fused score
        sorted_docs = sorted(all_docs.values(), key=lambda x: x[1], reverse=True)
        top_docs = sorted_docs[:top_k_final]

        if not top_docs:
            return "No relevant context found in knowledge base."

        lines = []
        for doc, score in top_docs:
            source = doc.metadata.get("source", "unknown")
            if doc.content_type in ("image", "image+report"):
                content = doc.metadata.get("report", "[image]")[:300]
            else:
                content = doc.content[:400]
            lines.append(f"[{source}|score={score:.3f}] {content}")

        logger.info(f"MultiScaleRAG retrieved {len(top_docs)} docs for query: {query[:50]}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Per-scale retrieval
    # ------------------------------------------------------------------

    def _retrieve_at_scale(
        self,
        scale: ScaleConfig,
        query: str,
        image_path: Optional[str],
    ) -> List[Tuple[Document, float]]:
        try:
            if scale.query_type == "image":
                if image_path is None:
                    return []
                # Optionally crop a specific region
                query_image = self._get_query_image(image_path, scale.crop_region)
                return self.vs.retrieve(
                    query="",
                    top_k=scale.top_k,
                    query_type="image",
                    image_path=query_image,
                )
            elif scale.query_type == "metric":
                if image_path is None:
                    return []
                metrics = self._compute_query_metrics(image_path)
                if not metrics:
                    return []
                return self.vs.retrieve_metric(metrics, top_k=scale.top_k)
            else:
                return self.vs.retrieve(query, top_k=scale.top_k, query_type="text")
        except Exception as e:
            # Re-raise critical runtime errors; swallow only retrieval-level failures.
            if isinstance(e, (MemoryError, SystemError)):
                raise
            logger.warning(f"Retrieval at scale '{scale.name}' failed: {e}")
            return []

    def _compute_query_metrics(self, image_path: str) -> Dict[str, float]:
        """
        Run RetSAM + AutoMorph on *image_path* to extract quantitative biomarkers
        for metric-scale retrieval.  Each tool call is independently fault-tolerant:
        partial results (e.g. only RetSAM succeeded) are still returned and used.
        """
        metrics: Dict[str, float] = {}

        # ── RetSAM: segmentation area fraction ──────────────────────────────
        try:
            out = self.scheduler.run("retsam", {
                "image_path": image_path,
                "prompt_type": "auto",
            })
            if "area_fraction" in out:
                metrics["area_fraction"] = float(out["area_fraction"])
        except Exception as e:
            logger.warning(f"RetSAM skipped in metric retrieval ({image_path}): {e}")

        # ── AutoMorph: vascular morphology metrics ───────────────────────────
        try:
            out = self.scheduler.run("automorph", {"image_path": image_path})
            m = out.get("metrics", {})
            for key in ("fractal_dim", "vessel_density", "av_ratio", "cup_disc_ratio"):
                if key in m:
                    metrics[key] = float(m[key])
        except Exception as e:
            logger.warning(f"AutoMorph skipped in metric retrieval ({image_path}): {e}")

        return metrics

    def _get_query_image(
        self,
        image_path: str,
        crop_tool_id: Optional[str],
    ) -> str:
        """Return the image to use for retrieval at a given scale."""
        if crop_tool_id is None:
            return image_path

        try:
            result = self.scheduler.run(crop_tool_id, {"image_path": image_path})
            # disc_fovea returns fovea_center/disc_center; do a standard crop
            disc_center = result.get("disc_center")
            disc_radius = result.get("disc_radius", 100)
            if disc_center:
                crop_result = self.scheduler.run("roi_cropping", {
                    "image_path": image_path,
                    "x": int(disc_center[0] - disc_radius * 1.5),
                    "y": int(disc_center[1] - disc_radius * 1.5),
                    "width": int(disc_radius * 3),
                    "height": int(disc_radius * 3),
                })
                # For retrieval, we need a file path; save the cropped image temporarily.
                # Register an atexit handler so the temp file is always removed.
                import atexit
                import base64
                import io
                import os
                import tempfile
                from PIL import Image
                img_data = base64.b64decode(crop_result["cropped_b64"])
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_data)
                    tmp_name = tmp.name
                atexit.register(lambda p=tmp_name: os.unlink(p) if os.path.exists(p) else None)
                return tmp_name
        except Exception as e:
            logger.warning(f"Region crop for scale retrieval failed: {e}; using full image.")

        return image_path

    # ------------------------------------------------------------------
    # Unlabelled database indexing
    # ------------------------------------------------------------------

    def index_unlabelled_images(
        self,
        image_dir: str,
        extensions: Tuple[str, ...] = (".jpg", ".png", ".jpeg"),
    ) -> int:
        """
        Index a directory of unlabelled images using CLIP embeddings.
        No text labels are required — images are retrievable by image similarity.
        """
        from pathlib import Path
        img_dir = Path(image_dir)
        if not img_dir.exists():
            logger.warning(f"Image directory not found: {img_dir}")
            return 0

        count = 0
        for img_path in img_dir.rglob("*"):
            if img_path.suffix.lower() not in extensions:
                continue
            try:
                self.vs.add_image(
                    image_path=str(img_path),
                    metadata={"source": "unlabelled", "path": str(img_path)},
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to index {img_path}: {e}")

        logger.info(f"Indexed {count} unlabelled images from {image_dir}.")
        self.vs.save()
        return count
