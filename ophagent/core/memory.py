"""
OphAgent Memory Manager.

Short-term memory  – conversation turn buffer for the current session.
Long-term memory   – persistent FAISS-backed case store with RAG retrieval.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ophagent.utils.logger import get_logger

logger = get_logger("core.memory")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single conversation turn."""
    role: str                    # "user" | "assistant" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    """A long-term memory record created after a completed session."""
    entry_id: str
    summary: str
    key_findings: List[str]
    tools_used: List[str]
    modalities: List[str]
    tags: List[str]
    timestamp: float = field(default_factory=time.time)
    raw_report: Optional[str] = None


# ---------------------------------------------------------------------------
# Short-term memory
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """Ring-buffer of conversation turns for the active session."""

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self._turns: List[Turn] = []

    def add(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        turn = Turn(role=role, content=content, metadata=metadata or {})
        self._turns.append(turn)
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """Return turns as list of {"role": ..., "content": ...} dicts for LLM."""
        turns = self._turns[-last_n:] if last_n else self._turns
        return [{"role": t.role, "content": t.content} for t in turns
                if t.role in ("user", "assistant")]

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)


# ---------------------------------------------------------------------------
# Long-term memory (FAISS-backed)
# ---------------------------------------------------------------------------

class LongTermMemory:
    """
    Persistent memory store backed by FAISS.
    Each entry is embedded as text and stored for future RAG retrieval.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        from config.settings import get_settings
        cfg = get_settings().knowledge_base
        self.index_path = Path(index_path or cfg.faiss_index_path)
        self.metadata_path = Path(metadata_path or cfg.faiss_metadata_path)
        self.embed_model_name = embed_model

        self._embedder = None      # lazy init
        self._index = None         # lazy init
        self._entries: List[MemoryEntry] = []
        self._load_if_exists()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _build_index(self) -> None:
        import faiss
        import numpy as np
        if not self._entries:
            return
        texts = [self._entry_to_text(e) for e in self._entries]
        embedder = self._get_embedder()
        vectors = embedder.encode(texts, normalize_embeddings=True)
        dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # inner-product on normalised = cosine
        self._index.add(vectors.astype(np.float32))

    def _entry_to_text(self, entry: MemoryEntry) -> str:
        return (
            f"{entry.summary} "
            f"Findings: {', '.join(entry.key_findings)}. "
            f"Tags: {', '.join(entry.tags)}."
        )

    def _load_if_exists(self) -> None:
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, encoding="utf-8") as f:
                    raw = [json.loads(line) for line in f if line.strip()]
                self._entries = [MemoryEntry(**r) for r in raw]
                logger.info(f"Loaded {len(self._entries)} long-term memory entries.")
                self._build_index()
            except Exception as e:
                logger.warning(f"Failed to load long-term memory: {e}")

    def save(self) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(asdict(entry)) + "\n")
        if self._index is not None:
            import faiss
            faiss.write_index(self._index, str(self.index_path))
        logger.info("Long-term memory saved.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_entry(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        self._build_index()   # rebuild (acceptable for small stores)
        logger.debug(f"Added memory entry: {entry.entry_id}")

    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Retrieve the most relevant memory entries for *query*."""
        if not self._entries or self._index is None:
            return []
        import numpy as np
        embedder = self._get_embedder()
        q_vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        distances, indices = self._index.search(q_vec, min(top_k, len(self._entries)))
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._entries):
                results.append(self._entries[idx])
        return results

    def format_for_context(self, entries: List[MemoryEntry]) -> str:
        """Format retrieved entries as a context string for the LLM."""
        if not entries:
            return "No relevant past cases found."
        lines = []
        for e in entries:
            lines.append(
                f"[Case {e.entry_id}] {e.summary}\n"
                f"  Findings: {', '.join(e.key_findings)}\n"
                f"  Tools: {', '.join(e.tools_used)}"
            )
        return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Unified MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Single access point for both short-term and long-term memory.
    Used by OphAgent, Planner, Verifier.
    """

    def __init__(self):
        from config.settings import get_settings
        cfg = get_settings()
        self.short = ShortTermMemory(max_turns=cfg.session_history_limit)
        self.long = LongTermMemory()

    # Short-term delegation
    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None):
        self.short.add(role, content, metadata)

    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        return self.short.get_history(last_n)

    def clear_session(self):
        self.short.clear()

    # Long-term delegation
    def retrieve(self, query: str, top_k: int = 5) -> str:
        entries = self.long.search(query, top_k=top_k)
        return self.long.format_for_context(entries)

    def consolidate_session(
        self,
        query: str,
        report: str,
        tools_used: List[str],
        llm=None,
    ) -> None:
        """
        Use the LLM to extract a structured memory entry from the completed
        session and persist it to long-term memory.
        """
        import uuid
        from ophagent.llm.prompts import PromptLibrary

        if llm is None:
            logger.warning("No LLM provided; skipping session consolidation.")
            return

        system = PromptLibrary.MEMORY_CONSOLIDATION_SYSTEM
        user = PromptLibrary.MEMORY_CONSOLIDATION_USER.substitute(
            query=query,
            report=report,
            tools_used=", ".join(tools_used),
        )
        raw = llm.chat_json(
            [{"role": "user", "content": user}],
            system=system,
        )
        entry = MemoryEntry(
            entry_id=str(uuid.uuid4())[:8],
            summary=raw.get("summary", ""),
            key_findings=raw.get("key_findings", []),
            tools_used=raw.get("tools_used", tools_used),
            modalities=raw.get("modalities", []),
            tags=raw.get("tags", []),
            raw_report=report,
        )
        self.long.add_entry(entry)
        self.long.save()
        logger.info(f"Session consolidated to long-term memory: {entry.entry_id}")
