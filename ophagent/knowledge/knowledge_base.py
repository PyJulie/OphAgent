"""
Unified Knowledge Base for OphAgent.

Aggregates all knowledge sources into a single retrieval interface:
  - Local Data (image-report archive + operational standards)
  - Textbooks / guidelines
  - Search Engine (PubMed, arXiv)
  - Interactive session context
  - Long-term case memory (managed by MemoryManager)

The KB is the single entry point for all RAG retrieval in the agent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ophagent.knowledge.local_data import LocalDataSource
from ophagent.knowledge.search_engine import InteractiveSource, SearchEngineSource
from ophagent.knowledge.textbook import TextbookSource
from ophagent.knowledge.vector_store import MultimodalVectorStore
from ophagent.utils.logger import get_logger

logger = get_logger("knowledge.kb")


class KnowledgeBase:
    """
    Unified entry point for all knowledge retrieval in OphAgent.

    All sources share a single MultimodalVectorStore for deduplication
    and unified similarity search.

    Usage::

        kb = KnowledgeBase()
        context = kb.retrieve("diabetic retinopathy treatment guidelines", top_k=5)
    """

    def __init__(self, shared_vector_store: Optional[MultimodalVectorStore] = None):
        self.vs = shared_vector_store or MultimodalVectorStore()
        # Sources share the same vector store
        self.local = LocalDataSource(vector_store=self.vs)
        self.textbook = TextbookSource(vector_store=self.vs)
        self.search = SearchEngineSource(vector_store=self.vs)
        self.interactive = InteractiveSource(vector_store=self.vs)
        logger.info(f"KnowledgeBase initialised with {len(self.vs)} documents.")

    # ------------------------------------------------------------------
    # Build (offline indexing)
    # ------------------------------------------------------------------

    def build(self, force: bool = False, with_metrics: bool = False) -> None:
        """
        Index all local sources.  Run this once offline before deploying.

        Args:
            force:        Re-index already-indexed sources.
            with_metrics: Run RetSAM + AutoMorph on archive images to populate
                          the quantitative metric index (Scale 4 of Multi-Scale RAG).
                          Requires tool weights to be present and configured.
        """
        logger.info("Building knowledge base...")
        self.local.build(force=force, with_metrics=with_metrics)
        self.textbook.index_all(force=force)
        self.vs.save()
        logger.info(f"Knowledge base built: {len(self.vs)} total documents.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve the most relevant context for *query* across all sources.
        Returns a formatted string for LLM context injection.
        """
        return self.vs.retrieve_text(query, top_k=top_k)

    def retrieve_with_image(
        self,
        query: str,
        image_path: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """
        Retrieve context using both text and image similarity (multi-scale RAG).
        See strategies/multiscale_rag.py for the full multi-scale pipeline.
        """
        text_docs = self.vs.retrieve(query, top_k=top_k // 2 or 1, query_type="text")
        img_docs = []
        if image_path:
            img_docs = self.vs.retrieve(
                query="", top_k=top_k // 2 or 1,
                query_type="image", image_path=image_path,
            )

        all_docs = text_docs + img_docs
        # Deduplicate by doc_id
        seen = set()
        unique = []
        for doc, score in all_docs:
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                unique.append((doc, score))
        # Sort by score descending
        unique.sort(key=lambda x: x[1], reverse=True)

        if not unique:
            return "No relevant documents found."

        lines = []
        for doc, score in unique[:top_k]:
            content = doc.content[:400]
            meta_str = doc.metadata.get("source", "unknown")
            lines.append(f"[{doc.doc_id}|{meta_str}|score={score:.3f}] {content}")
        return "\n\n".join(lines)

    def live_search(self, query: str, max_results: int = 5) -> str:
        """
        Trigger a live web search and return indexed results as context.
        Also saves new results to the vector store for future retrieval.
        """
        results = self.search.search(query, max_results=max_results)
        self.vs.save()
        if not results:
            return "No search results found."
        lines = [f"- {r['title']}: {r['snippet'][:200]}" for r in results]
        return "\n".join(lines)

    def add_interactive_context(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Add user-provided text to the knowledge base for this session."""
        self.interactive.add_context(text, metadata=metadata)
