"""
OphAgent CLI — Interactive and single-query modes.

Usage::
    # Interactive session
    python scripts/run_agent.py

    # Single query with image
    python scripts/run_agent.py \
        --query "Analyse this fundus image for diabetic retinopathy." \
        --images patient_cfp.jpg

    # Batch from JSON file
    python scripts/run_agent.py --batch queries.json --output results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OphAgent Clinical AI Agent")
    p.add_argument("--query", "-q", type=str, help="Single query string")
    p.add_argument("--images", "-i", nargs="*", help="Image file paths")
    p.add_argument("--batch", "-b", type=str, help="JSON file with list of queries")
    p.add_argument("--output", "-o", type=str, help="Output JSON file for batch mode")
    p.add_argument("--interactive", "-I", action="store_true", help="Interactive REPL mode")
    p.add_argument("--session-id", type=str, default=None, help="Session identifier")
    p.add_argument("--json-out", action="store_true", help="Print response as JSON")
    return p


def run_single(agent, query: str, images, session_id, json_out: bool):
    response = agent.run(query=query, image_paths=images, session_id=session_id)
    if json_out:
        print(json.dumps(response.to_dict(), indent=2, default=str))
    else:
        print("\n" + "=" * 60)
        print(response.report)
        print("=" * 60)
        if response.needs_human_review:
            print("[!] This case has been flagged for human review.")
        print(f"Duration: {response.duration_s:.2f}s")


def run_batch(agent, batch_file: str, output_file: str):
    with open(batch_file, encoding="utf-8") as f:
        queries = json.load(f)

    results = []
    for i, item in enumerate(queries):
        query = item.get("query", item) if isinstance(item, dict) else item
        images = item.get("images", []) if isinstance(item, dict) else []
        print(f"[{i+1}/{len(queries)}] Processing: {query[:60]}...")
        response = agent.run(query=query, image_paths=images)
        results.append(response.to_dict())

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {output_file}")


def interactive_loop(agent):
    print("OphAgent Interactive Session (type 'exit' to quit, 'reset' to clear session)")
    session_id = "interactive"
    while True:
        try:
            query = input("\nQuery> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "reset":
            agent.reset_session()
            print("[Session reset]")
            continue

        # Optional image paths
        images_input = input("Images (paths, space-separated, or Enter to skip)> ").strip()
        images = images_input.split() if images_input else []

        response = agent.run(query=query, image_paths=images, session_id=session_id)
        print("\n" + response.report)
        if response.needs_human_review:
            print("\n[!] Flagged for human review.")


def main():
    parser = build_parser()
    args = parser.parse_args()

    from ophagent.core.agent import OphAgent
    agent = OphAgent()

    if args.interactive:
        interactive_loop(agent)
    elif args.batch:
        output = args.output or "results.json"
        run_batch(agent, args.batch, output)
    elif args.query:
        run_single(agent, args.query, args.images or [], args.session_id, args.json_out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
