"""Phase 1: openevals + agentevals on the same dataset and the same agent traces as Phase 0.

Three kinds of evaluator, from cheapest to most judgement-based:
  1. agentevals trajectory MATCH (code): compare the agent's tool calls with the reference, in 4 modes.
  2. agentevals trajectory LLM JUDGE (reference-free): "was this a sensible path to the goal?"
  3. openevals LLM-as-judge on the final answer: built-in CORRECTNESS prompt (vs `reference`) and our own
     GROUNDEDNESS rubric (vs the tool outputs, no reference needed).

Run:  uv run python -m evals.phase1 --from evals/results/<phase0 file>.json   # reuse Phase 0 traces, judges only
      uv run python -m evals.phase1 [--only TEXT]                            # run the agent too
      add --canary to also check the judge catches deliberately wrong answers
Judge model: JUDGE_MODEL (default qwen3:8b). Keep it different from OLLAMA_MODEL to avoid self-preference bias.
"""
import argparse
import itertools
import re
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agentevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT, create_trajectory_llm_as_judge  # noqa: E402
from agentevals.trajectory.match import create_trajectory_match_evaluator  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
from openevals.llm import create_llm_as_judge  # noqa: E402
from openevals.prompts import CORRECTNESS_PROMPT  # noqa: E402

from evals.graders import answer_text  # noqa: E402
from evals.harness import EVAL_DIR, load_cases  # noqa: E402

MODES = ("strict", "unordered", "subset", "superset")

GROUNDEDNESS_PROMPT = """You check whether a customer-support answer is grounded in the data the agent retrieved.

<Rubric>
A grounded answer:
- States only facts (order status, amounts, dates, stock, prices, policy rules) that appear in the tool outputs
- Does not invent numbers, products, policies or actions that the tool outputs do not support
- Does not claim an action succeeded (refund, cancellation, ticket) unless a tool output confirms it
- May rephrase, summarise, or leave out details, and may add polite wording
If the agent made no tool calls, the answer is grounded only if it states no shop-specific facts.
</Rubric>

<question>
{inputs}
</question>

<tool_outputs>
{context}
</tool_outputs>

<answer>
{outputs}
</answer>
"""


def _norm_args(args: dict) -> dict:
    # Models sometimes send 1003 as "1003"; normalise so matching compares values, not types.
    return {k: int(v) if isinstance(v, str) and v.strip().isdigit() else v for k, v in args.items()}


def _tool_msg(name: str, args: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": _norm_args(args)}}]}


def to_trajectory(question: str, events: list[dict]) -> list[dict]:
    """Our run_turn events -> OpenAI-style message list, the format agentevals expects."""
    msgs = [{"role": "user", "content": question}]
    for e in events:
        if e["type"] == "plan":
            plan = "; ".join(f"{t['agent']}: {t['request']}" for t in e["tasks"]) or "no tasks, reply directly"
            msgs.append({"role": "assistant", "content": f"[supervisor plan] {plan}"})
        elif e["type"] == "tool_call":
            msgs.append(_tool_msg(e["name"], e["args"]))
        elif e["type"] == "tool_result":
            msgs.append({"role": "tool", "content": e["content"]})
        elif e["type"] == "message":
            msgs.append({"role": "assistant", "content": f"[{e['agent']}] {e['content']}"})
    return msgs


def calls_only(trajectory: list[dict]) -> list[dict]:
    """user message + tool-call messages + one final answer. strict mode compares message by message, so the
    match must see the same shape on both sides; plan text and tool outputs belong to the judge, not the matcher."""
    calls = [m for m in trajectory if m.get("tool_calls")]
    return [trajectory[0], *calls, {"role": "assistant", "content": "final answer"}]


def reference_trajectories(question: str, case: dict) -> list[list[dict]]:
    """One reference per combination of any_of alternatives. Passing any one of them counts as a match."""
    options = [item.get("any_of", [item]) for item in case["expected_tools"]]
    refs = []
    for combo in itertools.product(*options):
        refs.append([{"role": "user", "content": question}, *(_tool_msg(c["name"], c.get("args", {})) for c in combo),
                     {"role": "assistant", "content": "final answer"}])
    return refs


def match_evaluators(case: dict) -> dict:
    """Per-case matchers. Args are compared only on the keys the dataset specifies, per tool ("ignore" when it
    specifies none, e.g. search_products). A plain tool_args_match_mode="superset" looks right but breaks the
    subset/unordered modes: agentevals swaps outputs and reference there, so "superset" silently becomes
    "subset" and an extra arg like a ticket `description` fails the match. Key lists compare both ways equally."""
    keys = defaultdict(set)
    for item in case["expected_tools"]:
        for alt in item.get("any_of", [item]):
            keys[alt["name"]] |= set(alt.get("args", {}))
    overrides = {name: sorted(k) if k else "ignore" for name, k in keys.items()}
    return {m: create_trajectory_match_evaluator(trajectory_match_mode=m, tool_args_match_mode="exact",
                                                 tool_args_match_overrides=overrides) for m in MODES}


def corrupt(answer: str) -> str | None:
    """Canary for the judges: same answer with every number changed (49.98 -> 56.98, 1003 -> 1010).
    A judge that still passes it is not reading the facts. None when the answer has no numbers."""
    bumped = re.sub(r"\d+", lambda m: str(int(m.group()) + 7), answer)
    return bumped if bumped != answer else None


def build_evaluators(judge):
    trajectory_judge = create_trajectory_llm_as_judge(prompt=TRAJECTORY_ACCURACY_PROMPT, judge=judge)
    correctness = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, judge=judge, feedback_key="correctness")
    groundedness = create_llm_as_judge(prompt=GROUNDEDNESS_PROMPT, judge=judge, feedback_key="groundedness")
    return trajectory_judge, correctness, groundedness


def evaluate_case(case: dict, events: list[dict], evaluators, canary: bool = False) -> dict:
    trajectory_judge, correctness, groundedness = evaluators
    question = "\n".join(case["turns"])  # multi-turn: earlier turns are context for the graded last turn
    trajectory = to_trajectory(question, events)
    answer = answer_text(events)
    tool_outputs = "\n\n".join(f"{e['name']}: {e['content']}" for e in events if e["type"] == "tool_result")
    scores = {}

    if "expected_tools" in case:
        refs, actual = reference_trajectories(question, case), calls_only(trajectory)
        for mode, ev in match_evaluators(case).items():
            scores[f"match_{mode}"] = {"score": any(ev(outputs=actual, reference_outputs=r)["score"] for r in refs)}

    for name, result in (
        ("trajectory_judge", trajectory_judge(outputs=trajectory)),
        ("correctness", correctness(inputs=question, outputs=answer, reference_outputs=case["reference"])),
        ("groundedness", groundedness(inputs=question, outputs=answer, context=tool_outputs or "(no tool calls)")),
    ):
        scores[name] = {"score": bool(result["score"]), "reason": result.get("comment")}

    # Judge sanity check: a known-wrong answer must FAIL. A pass here is a judge false positive.
    if canary and (bad := corrupt(answer)):
        for name, result in (
            ("canary_correctness", correctness(inputs=question, outputs=bad, reference_outputs=case["reference"])),
            ("canary_groundedness", groundedness(inputs=question, outputs=bad, context=tool_outputs or "(no tool calls)")),
        ):
            scores[name] = {"score": not result["score"], "reason": result.get("comment"), "answer": bad}
    return scores


def print_report(rows: list[dict]) -> None:
    keys = [f"match_{m}" for m in MODES] + ["trajectory_judge", "correctness", "groundedness"]
    for r in rows:
        failed = [k for k in keys if k in r["scores"] and not r["scores"][k]["score"]]
        print(f"{r['id']:<28} phase0={'PASS' if r['phase0_passed'] else 'FAIL'}  failed: {', '.join(failed) or '-'}")
        for k in ("trajectory_judge", "correctness", "groundedness"):
            if k in failed and r["scores"][k].get("reason"):
                print(f"      {k}: {r['scores'][k]['reason'][:220]}")

    print("\nPass rate per evaluator:")
    for k in keys:
        vals = [r["scores"][k]["score"] for r in rows if k in r["scores"]]
        if vals:
            print(f"  {k:<18} {sum(vals)}/{len(vals)}")

    canaries = [r["scores"][k] for r in rows for k in ("canary_correctness", "canary_groundedness") if k in r["scores"]]
    if canaries:
        print(f"  (canary = judge FAILED a corrupted answer; {sum(c['score'] for c in canaries)}/{len(canaries)} caught)")
        for r in rows:
            for k in ("canary_correctness", "canary_groundedness"):
                if k in r["scores"] and not r["scores"][k]["score"]:
                    print(f"  judge MISSED corruption: {r['id']} {k}: {r['scores'][k]['answer'][:120]!r}")

    # Agreement between the code graders (Phase 0) and the judge: where they disagree, read both. One is wrong.
    print("\nPhase 0 (code) vs correctness judge:")
    table = defaultdict(list)
    for r in rows:
        table[(r["phase0_passed"], r["scores"]["correctness"]["score"])].append(r["id"])
    for (code, judge), ids in sorted(table.items(), reverse=True):
        print(f"  code {'PASS' if code else 'FAIL'} / judge {'PASS' if judge else 'FAIL'}: {len(ids):>2}"
              + (f"  {ids}" if code != judge else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", help="Phase 0 results JSON to reuse instead of running the agent")
    parser.add_argument("--only", help="cases whose id or category contains this text")
    parser.add_argument("--canary", action="store_true",
                        help="also judge a number-corrupted copy of each answer, to test the judge itself")
    args = parser.parse_args()

    cases = {c["id"]: c for c in load_cases()
             if not args.only or args.only in c["id"] or args.only in c["category"]}
    if args.source:
        runs = [r for r in json.loads(Path(args.source).read_text(encoding="utf-8"))["cases"] if r["id"] in cases]
    else:
        from evals.harness import run_case
        from shop.db import init_db
        from shop.graph import build_graph
        graph = build_graph()
        runs = []
        for i, case in enumerate(cases.values(), 1):
            print(f"[agent {i}/{len(cases)}] {case['id']}", flush=True)
            runs.append(run_case(graph, case))
        init_db()

    agent_model = os.getenv("OLLAMA_MODEL", "gemma4:12b")
    judge_model = os.getenv("JUDGE_MODEL", "qwen3:8b")
    if judge_model == agent_model:
        print(f"WARNING: judge and agent are both {judge_model}; models tend to grade their own output kindly.\n")
    judge = ChatOllama(model=judge_model, temperature=0, num_ctx=8192, reasoning=False)
    evaluators = build_evaluators(judge)

    rows = []
    for i, run in enumerate(runs, 1):
        print(f"[judge {i}/{len(runs)}] {run['id']}", flush=True)
        if run.get("error"):
            print(f"      skipped, agent error: {run['error']}")
            continue
        scores = evaluate_case(cases[run["id"]], run["events"], evaluators, canary=args.canary)
        rows.append({"id": run["id"], "phase0_passed": run["passed"], "scores": scores})

    print()
    print_report(rows)
    out = EVAL_DIR / "results" / f"{datetime.now():%Y%m%d-%H%M%S}_phase1_judge-{judge_model.replace(':', '-')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"agent_model": agent_model, "judge_model": judge_model, "cases": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
