"""
Multimodal FAISS Vector Store for OphAgent Knowledge Base.

Supports both text and image embeddings (CLIP) in a unified index.
Used by all KB components: local data, textbooks, search results, memory.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ophagent.utils.logger import get_logger

logger = get_logger("knowledge.vector_store")


@dataclass
class Document:
    """A unit of knowledge stored in the vector store."""
    doc_id: str
    content: str                          # text content or base64 image
    content_type: str                     # "text" | "image" | "image+report"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d



# ---------------------------------------------------------------------------
# Canonical quantitative metric keys (fixed order → reproducible vector)
# ---------------------------------------------------------------------------
METRIC_KEYS: List[str] = [
    "area_fraction",   # RetSAM: segmented area / total image area
    "fractal_dim",     # AutoMorph: fractal dimension of vessel tree
    "vessel_density",  # AutoMorph: vessel density ratio
    "av_ratio",        # AutoMorph: arterio-venous ratio
    "cup_disc_ratio",  # AutoMorph: cup-to-disc ratio (CDR)
]


class MultimodalVectorStore:
    """
    Unified FAISS index for text + image documents, plus a lightweight
    separate numpy index for quantitative metric documents.

    - Text is embedded with SentenceTransformer.
    - Images are embedded with a CLIP model.
    - Both are projected to the same dimensionality (IMAGE_DIM) for joint retrieval.
    - Quantitative metrics (from RetSAM / AutoMorph) are stored in a separate
      in-memory numpy matrix for biomarker-based similarity search.
    """

    TEXT_DIM = 768    # MedCPT-Query-Encoder (paper §2.4.3)
    IMAGE_DIM = 512   # ViT-B/32 CLIP

    def __init__(
        self,
        index_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
    ):
        from config.settings import get_settings
        cfg = get_settings().knowledge_base
        self.index_path = Path(index_path or cfg.faiss_index_path)
        self.meta_path = Path(meta_path or cfg.faiss_metadata_path)

        self._documents: List[Document] = []
        self._index = None           # FAISS index (text + image)
        self._dim: Optional[int] = None

        self._text_embedder = None   # lazy
        self._clip_model = None      # lazy
        self._clip_preprocess = None

        # Metric index — separate from FAISS to avoid modality mixing
        self._metric_docs: List[Document] = []
        self._metric_matrix: Optional[np.ndarray] = None  # (N, IMAGE_DIM) cache
        self._metric_lock = __import__("threading").RLock()  # guards lazy matrix build

        self._load_if_exists()

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_text_embedder(self):
        if self._text_embedder is None:
            from sentence_transformers import SentenceTransformer
            from config.settings import get_settings
            model_name = get_settings().knowledge_base.text_embed_model
            self._text_embedder = SentenceTransformer(model_name)
        return self._text_embedder

    def _get_clip(self):
        if self._clip_model is None:
            import open_clip
            import torch
            from config.settings import get_settings
            model_name = get_settings().knowledge_base.image_embed_model
            self._clip_model, _, self._clip_preprocess = (
                open_clip.create_model_and_transforms(model_name)
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model = self._clip_model.to(device).eval()
        return self._clip_model, self._clip_preprocess

    def embed_metric(self, metrics: Dict[str, float]) -> np.ndarray:
        """
        Encode a quantitative metric dict as a normalized 512-dim vector.

        Values for each key in METRIC_KEYS are placed at fixed positions
        (indices 0..4); the remaining 507 dims are zeros.  After L2-normalisation
        the cosine similarity of two metric vectors equals the cosine similarity
        of their 5-dim metric sub-vectors.
        """
        vec = np.zeros(self.IMAGE_DIM, dtype=np.float32)
        for i, key in enumerate(METRIC_KEYS):
            vec[i] = float(metrics.get(key, 0.0))
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

    def embed_text(self, text: str) -> np.ndarray:
        embedder = self._get_text_embedder()
        vec = embedder.encode([text], normalize_embeddings=True)[0]
        # Align text vector to IMAGE_DIM so text and image docs share the same FAISS index.
        # MedCPT produces 768-dim; ViT-B/32 CLIP produces 512-dim (IMAGE_DIM).
        if len(vec) < self.IMAGE_DIM:
            # Pad with zeros (e.g. legacy 384-dim model)
            vec = np.concatenate([vec, np.zeros(self.IMAGE_DIM - len(vec))])
        elif len(vec) > self.IMAGE_DIM:
            # Truncate longer embeddings (MedCPT 768 → 512) and re-normalise so
            # cosine similarity via inner product remains valid.
            vec = vec[:self.IMAGE_DIM]
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm
        return vec.astype(np.float32)

    def embed_image(self, image_path_or_b64: str) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import io

        model, preprocess = self._get_clip()
        device = next(model.parameters()).device

        # Decode source
        try:
            img = Image.open(image_path_or_b64).convert("RGB")
        except Exception:
            data = base64.b64decode(image_path_or_b64)
            img = Image.open(io.BytesIO(data)).convert("RGB")

        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(tensor)
            feat = F.normalize(feat, dim=-1).cpu().numpy()[0]
        return feat.astype(np.float32)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _init_index(self, dim: int) -> None:
        import faiss
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)   # cosine on unit vectors

    @property
    def _metric_meta_path(self) -> Path:
        """Sidecar file for metric docs, e.g. ophagent_meta_metrics.jsonl"""
        return self.meta_path.with_name(self.meta_path.stem + "_metrics" + self.meta_path.suffix)

    def _load_if_exists(self) -> None:
        _REQUIRED_FIELDS = {"doc_id", "content", "content_type"}

        if self.meta_path.exists():
            try:
                with open(self.meta_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if not _REQUIRED_FIELDS.issubset(data):
                                logger.debug(
                                    "Skipping non-vector-store record while loading metadata."
                                )
                                continue
                            self._documents.append(Document(**data))
                        except Exception as exc:
                            logger.warning(f"Skipping malformed document entry: {exc}")
                logger.info(f"Loaded {len(self._documents)} documents from vector store.")
                if self.index_path.exists():
                    import faiss
                    self._index = faiss.read_index(str(self.index_path))
                    self._dim = self._index.d
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}")

        # Load metric docs from sidecar file
        if self._metric_meta_path.exists():
            try:
                with open(self._metric_meta_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if not _REQUIRED_FIELDS.issubset(data):
                                logger.debug(
                                    "Skipping non-metric record while loading metric metadata."
                                )
                                continue
                            self._metric_docs.append(Document(**data))
                        except Exception as exc:
                            logger.warning(f"Skipping malformed metric doc entry: {exc}")
                logger.info(f"Loaded {len(self._metric_docs)} metric documents.")
            except Exception as e:
                logger.warning(f"Failed to load metric docs: {e}")

    def save(self) -> None:
        import faiss
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for doc in self._documents:
                f.write(json.dumps(doc.to_dict()) + "\n")
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))
        # Save metric docs to sidecar
        with open(self._metric_meta_path, "w", encoding="utf-8") as f:
            for doc in self._metric_docs:
                f.write(json.dumps(doc.to_dict()) + "\n")
        logger.info(
            f"Saved {len(self._documents)} documents + {len(self._metric_docs)} metric docs."
        )

    # ------------------------------------------------------------------
    # Add / Retrieve
    # ------------------------------------------------------------------

    def add_document(self, doc: Document) -> None:
        """Embed and index a document."""
        if doc.content_type == "image":
            vec = self.embed_image(doc.content)
        else:
            vec = self.embed_text(doc.content)

        if self._index is None:
            self._init_index(len(vec))

        self._index.add(np.expand_dims(vec, 0))
        doc.embedding = vec.tolist()
        self._documents.append(doc)

    def add_text(self, text: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> Document:
        import uuid
        doc = Document(
            doc_id=doc_id or str(uuid.uuid4())[:8],
            content=text,
            content_type="text",
            metadata=metadata or {},
        )
        self.add_document(doc)
        return doc

    def add_image(self, image_path: str, report: Optional[str] = None, metadata: Optional[Dict] = None) -> Document:
        import uuid
        doc = Document(
            doc_id=str(uuid.uuid4())[:8],
            content=image_path,
            content_type="image+report" if report else "image",
            metadata={**(metadata or {}), "report": report or ""},
        )
        self.add_document(doc)
        return doc

    def add_metric_doc(
        self,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None,
    ) -> Document:
        """
        Index a quantitative metric dict into the separate metric index.

        Args:
            metrics:  Dict with keys from METRIC_KEYS (missing keys default to 0).
            metadata: Arbitrary metadata (e.g. case_id, image_path, report snippet).

        Returns:
            The created Document.
        """
        import uuid
        vec = self.embed_metric(metrics)
        doc = Document(
            doc_id=str(uuid.uuid4())[:8],
            content=json.dumps({k: metrics.get(k, 0.0) for k in METRIC_KEYS}),
            content_type="metric",
            metadata=metadata or {},
            embedding=vec.tolist(),
        )
        self._metric_docs.append(doc)
        self._metric_matrix = None  # invalidate lazy cache
        return doc

    def retrieve_metric(
        self,
        metrics: Dict[str, float],
        top_k: int = 2,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve the top-k archived cases most similar in biomarker space.

        Args:
            metrics: Query metric dict (same keys as METRIC_KEYS).
            top_k:   Number of results to return.

        Returns:
            List of (Document, cosine_similarity) tuples, descending by score.
        """
        if not self._metric_docs:
            return []

        q_vec = self.embed_metric(metrics)

        # Build matrix lazily; shape (N, IMAGE_DIM).  RLock guards concurrent access.
        with self._metric_lock:
            if self._metric_matrix is None:
                self._metric_matrix = np.array(
                    [d.embedding for d in self._metric_docs], dtype=np.float32
                )
            scores: np.ndarray = self._metric_matrix @ q_vec  # cosine similarities (N,)
        k = min(top_k, len(self._metric_docs))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self._metric_docs[int(i)], float(scores[i])) for i in top_idx]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        query_type: str = "text",
        image_path: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents most similar to *query*.
        Returns list of (document, similarity_score) tuples.
        """
        if not self._documents or self._index is None:
            return []

        if query_type == "image" and image_path:
            q_vec = self.embed_image(image_path)
        else:
            q_vec = self.embed_text(query)

        k = min(top_k, len(self._documents))
        distances, indices = self._index.search(np.expand_dims(q_vec, 0), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._documents):
                results.append((self._documents[idx], float(dist)))
        return results

    def retrieve_text(self, query: str, top_k: int = 5) -> str:
        """Retrieve and format as a context string for the LLM."""
        docs = self.retrieve(query, top_k=top_k)
        if not docs:
            return "No relevant documents found."
        lines = []
        for doc, score in docs:
            snippet = doc.content[:500] if doc.content_type == "text" else f"[Image: {doc.metadata.get('report', '')}]"
            lines.append(f"[{doc.doc_id}] (score={score:.3f}) {snippet}")
        return "\n\n".join(lines)

    def __len__(self) -> int:
        return len(self._documents)
