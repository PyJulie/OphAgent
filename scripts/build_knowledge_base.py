"""
Build / update the OphAgent knowledge base.

Indexes all local data sources (image-report archive, operational standards,
textbooks) into the FAISS vector store.

Usage::
    python scripts/build_knowledge_base.py
    python scripts/build_knowledge_base.py --force          # re-index everything
    python scripts/build_knowledge_base.py --unlabelled /path/to/images
    python scripts/build_knowledge_base.py --search "DR guidelines"  # test retrieval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Build OphAgent Knowledge Base")
    parser.add_argument("--force", action="store_true", help="Force re-indexing of all sources")
    parser.add_argument("--unlabelled", type=str, help="Directory of unlabelled images to index")
    parser.add_argument("--search", type=str, help="Test retrieval with this query")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval test")
    parser.add_argument(
        "--with-metrics",
        action="store_true",
        help=(
            "Run RetSAM + AutoMorph on archive images to index quantitative metric "
            "vectors (Scale 4 of Multi-Scale RAG).  Requires model weights to be present."
        ),
    )
    args = parser.parse_args()

    from ophagent.knowledge.knowledge_base import KnowledgeBase
    from ophagent.strategies.multiscale_rag import MultiScaleRAG

    print("Initialising KnowledgeBase...")
    kb = KnowledgeBase()

    print("Building knowledge base (this may take a while)...")
    kb.build(force=args.force, with_metrics=args.with_metrics)

    if args.unlabelled:
        print(f"Indexing unlabelled images from: {args.unlabelled}")
        rag = MultiScaleRAG(vector_store=kb.vs)
        n = rag.index_unlabelled_images(args.unlabelled)
        print(f"Indexed {n} unlabelled images.")

    if args.search:
        print(f"\nTesting retrieval for: '{args.search}'")
        context = kb.retrieve(args.search, top_k=args.top_k)
        print("---")
        print(context)
        print("---")

    print(f"\nKnowledge base ready. Total documents: {len(kb.vs)}")


if __name__ == "__main__":
    main()
