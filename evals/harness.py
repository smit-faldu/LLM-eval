"""Phase 0 eval harness: run every dataset case through the real agent and grade it with code.

Run:  uv run python -m evals.harness                 # all cases
      uv run python -m evals.harness --only refund   # cases whose id or category contains "refund"
Results are saved to evals/results/<timestamp>_<model>.json for later comparison between models.
"""
import argparse
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from evals.graders import grade_case  # noqa: E402
from shop.db import get_conn, init_db  # noqa: E402
from shop.graph import build_graph, run_turn  # noqa: E402

EVAL_DIR = Path(__file__).parent


def load_cases(path: Path = EVAL_DIR / "dataset.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_case(graph, case: dict) -> dict:
    init_db()  # every case starts from the same seed, so write actions in one case can't leak into another
    thread_id = str(uuid.uuid4())
    start = time.perf_counter()
    try:
        for turn in case["turns"][:-1]:  # earlier turns only set up context; we grade the last one
            list(run_turn(graph, turn, thread_id))
        events = list(run_turn(graph, case["turns"][-1], thread_id))
    except Exception as e:  # one broken case shouldn't stop the whole run
        return {"id": case["id"], "category": case["category"], "passed": False, "error": repr(e),
                "checks": {}, "tool_precision": None, "seconds": round(time.perf_counter() - start, 1), "events": []}
    seconds = round(time.perf_counter() - start, 1)
    with get_conn() as conn:
        graded = grade_case(case, events, conn)
    return {"id": case["id"], "category": case["category"], **graded, "seconds": seconds, "events": events}


def print_report(results: list[dict]) -> None:
    for r in results:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']:<28} {r['seconds']:>6}s")
        if r.get("error"):
            print(f"      error: {r['error']}")
        for name, c in r["checks"].items():
            if not c["passed"]:
                print(f"      {name}: {c['detail']}")

    print("\nBy category:")
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["passed"])
    for cat, passes in by_cat.items():
        print(f"  {cat:<14} {sum(passes)}/{len(passes)}")

    print("\nBy check (pass rate where the check applies):")
    by_check = defaultdict(list)
    for r in results:
        for name, c in r["checks"].items():
            by_check[name].append(c["passed"])
    for name, passes in by_check.items():
        print(f"  {name:<17} {sum(passes)}/{len(passes)}")

    precisions = [r["tool_precision"] for r in results if r["tool_precision"] is not None]
    passed = sum(r["passed"] for r in results)
    print(f"\nOverall: {passed}/{len(results)} passed ({passed / len(results):.0%})")
    if precisions:
        print(f"Mean tool precision: {sum(precisions) / len(precisions):.2f}")
    print(f"Mean latency: {sum(r['seconds'] for r in results) / len(results):.1f}s per case")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="run cases whose id or category contains this text")
    args = parser.parse_args()

    cases = [c for c in load_cases() if not args.only or args.only in c["id"] or args.only in c["category"]]
    if not cases:
        raise SystemExit(f"No cases match {args.only!r}")
    model = os.getenv("OLLAMA_MODEL", "gemma4:12b")
    print(f"Running {len(cases)} cases with {model}\n")

    graph = build_graph()
    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}", flush=True)
        results.append(run_case(graph, case))
    init_db()  # leave the shop DB clean for the app

    print()
    print_report(results)
    out = EVAL_DIR / "results" / f"{datetime.now():%Y%m%d-%H%M%S}_{model.replace(':', '-').replace('/', '-')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": model, "cases": results}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
