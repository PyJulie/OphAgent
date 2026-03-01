"""Web/Literature Search auxiliary tool."""
from __future__ import annotations
from typing import Any, Dict, List
from ophagent.tools.base import BaseTool, ToolMetadata


class WebSearchTool(BaseTool):
    """
    Searches PubMed and arXiv for ophthalmology literature.
    Input:  {"query": str, "max_results": int (default 10),
             "domains": list[str] (optional)}
    Output: {"results": list[{"title": str, "url": str, "snippet": str}]}
    """
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        from duckduckgo_search import DDGS
        from config.settings import get_settings
        cfg = get_settings().tools
        query = inputs["query"]
        max_results = inputs.get("max_results", cfg.search_max_results)
        domains = inputs.get("domains", cfg.search_domains)
        domain_filter = " OR ".join(f"site:{d}" for d in domains)
        full_query = f"{query} ({domain_filter})"
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(full_query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return {"results": results, "query": query, "count": len(results)}
