"""
Textbook/Guideline Knowledge Source.

Ingests ophthalmology textbooks (PDF, TXT) and clinical guidelines,
chunks them, and indexes into the shared vector store.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ophagent.knowledge.vector_store import MultimodalVectorStore
from ophagent.utils.logger import get_logger
from ophagent.utils.text_utils import chunk_text

logger = get_logger("knowledge.textbook")


class TextbookSource:
    """
    Indexes ophthalmology textbooks and guidelines for RAG retrieval.

    Directory structure::

        textbooks/
            duanes_ophthalmology.pdf
            aao_guidelines/
                dr_guidelines.pdf
                ...
    """

    def __init__(
        self,
        textbook_root: Optional[Path] = None,
        vector_store: Optional[MultimodalVectorStore] = None,
    ):
        from config.settings import get_settings
        cfg = get_settings().knowledge_base
        self.textbook_root = Path(textbook_root or cfg.textbook_root)
        self.vs = vector_store or MultimodalVectorStore()
        self._indexed: set = set()

    def index_all(self, force: bool = False) -> int:
        if not self.textbook_root.exists():
            logger.warning(f"Textbook root not found: {self.textbook_root}")
            return 0

        count = 0
        for f in sorted(self.textbook_root.rglob("*")):
            if f.suffix.lower() not in (".txt", ".md", ".pdf"):
                continue
            if str(f) in self._indexed and not force:
                continue

            logger.info(f"Indexing textbook: {f.name}")
            text = self._extract_text(f)
            if not text.strip():
                continue

            for chunk in chunk_text(text, chunk_size=512, chunk_overlap=64):
                self.vs.add_text(
                    chunk,
                    metadata={
                        "source": "textbook",
                        "filename": f.name,
                        "relative_path": str(f.relative_to(self.textbook_root)),
                    },
                )
                count += 1

            self._indexed.add(str(f))

        logger.info(f"Indexed {count} textbook chunks.")
        return count

    @staticmethod
    def _extract_text(path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            try:
                import PyPDF2
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = [p.extract_text() or "" for p in reader.pages]
                return "\n".join(pages)
            except ImportError:
                logger.warning("PyPDF2 not installed; skipping PDF.")
                return ""
            except Exception as e:
                logger.warning(f"Failed to parse PDF {path}: {e}")
                return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def retrieve(self, query: str, top_k: int = 5) -> str:
        return self.vs.retrieve_text(query, top_k=top_k)
