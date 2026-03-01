"""
Search Engine Knowledge Source.

Provides domain-restricted web search (PubMed, arXiv) and on-the-fly
indexing of retrieved content into the shared vector store.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ophagent.knowledge.vector_store import MultimodalVectorStore
from ophagent.utils.logger import get_logger
from ophagent.utils.text_utils import chunk_text, truncate_text

logger = get_logger("knowledge.search_engine")


class SearchEngineSource:
    """
    Performs domain-restricted web searches and indexes results.

    Uses DuckDuckGo Search as the backend.  Results are deduplicated
    and added to the vector store for future retrieval.
    """

    def __init__(
        self,
        vector_store: Optional[MultimodalVectorStore] = None,
        domains: Optional[List[str]] = None,
        max_results: int = 10,
    ):
        from config.settings import get_settings
        cfg = get_settings().tools
        self.vs = vector_store or MultimodalVectorStore()
        self.domains = domains or cfg.search_domains
        self.max_results = max_results
        self._seen_urls: set = set()

    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Search for *query* and return a list of result dicts.
        Results are also indexed in the vector store for future retrieval.
        """
        from duckduckgo_search import DDGS
        n = max_results or self.max_results
        domain_filter = " OR ".join(f"site:{d}" for d in self.domains)
        full_query = f"{query} ({domain_filter})"

        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(full_query, max_results=n):
                    url = r.get("href", "")
                    if url in self._seen_urls:
                        continue
                    self._seen_urls.add(url)
                    item = {
                        "title": r.get("title", ""),
                        "url": url,
                        "snippet": r.get("body", ""),
                    }
                    results.append(item)
                    # Index snippet for future retrieval
                    snippet = item["snippet"]
                    if snippet:
                        self.vs.add_text(
                            snippet,
                            metadata={
                                "source": "web_search",
                                "url": url,
                                "title": item["title"],
                                "query": query,
                            },
                        )
        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")

        logger.info(f"Search returned {len(results)} results for: {query[:60]}")
        return results

    def search_pubmed(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search PubMed via the Entrez API (requires Biopython).
        Falls back to web search if unavailable.
        """
        try:
            from Bio import Entrez
            Entrez.email = "ophagent@research.local"
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()
            ids = record.get("IdList", [])
            if not ids:
                return []
            fetch_handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text")
            abstracts = fetch_handle.read()
            fetch_handle.close()
            # Index and return
            for chunk in chunk_text(abstracts, chunk_size=512):
                self.vs.add_text(chunk, metadata={"source": "pubmed", "query": query})
            return [{"title": "PubMed results", "url": "https://pubmed.ncbi.nlm.nih.gov", "snippet": truncate_text(abstracts, 500)}]
        except ImportError:
            logger.warning("Biopython not installed; falling back to web search for PubMed.")
            return self.search(f"site:pubmed.ncbi.nlm.nih.gov {query}", max_results=max_results)
        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
            return []

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """Retrieve indexed search content relevant to *query*."""
        return self.vs.retrieve_text(query, top_k=top_k)


class InteractiveSource:
    """
    Handles user-provided context injected during an interactive session
    (e.g. free-text notes, additional history provided by the clinician).
    """

    def __init__(self, vector_store: Optional[MultimodalVectorStore] = None):
        self.vs = vector_store or MultimodalVectorStore()

    def add_context(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Index user-provided text context."""
        for chunk in chunk_text(text, chunk_size=512, chunk_overlap=64):
            self.vs.add_text(chunk, metadata={**(metadata or {}), "source": "interactive"})

    def retrieve(self, query: str, top_k: int = 3) -> str:
        return self.vs.retrieve_text(query, top_k=top_k)
