"""Phase 2: RAGAS on the policy RAG, plus RAGAS's agent metrics.

A RAG answer can fail in two places, and the metrics are split the same way:
  RETRIEVAL   did search_policy return the right chunks?      context precision / recall (+ ID-based versions)
  GENERATION  did the agent use them faithfully and on-topic?  faithfulness, answer relevancy

Subcommands (dataset: evals/rag_dataset.jsonl, 22 hand-written policy questions with the sections that answer them):
  retrieval  Experiment grid: chunking x k, scored by section IDs. Embeddings only, no LLM, fast.
  rag        Ask each question through the full agent, then score it with RAGAS LLM metrics.
  agent      RAGAS ToolCallAccuracy / ToolCallF1 / AgentGoalAccuracy on Phase 0 traces (--from).
  generate   RAGAS synthetic test set from policies.md, written to evals/rag_synthetic.jsonl for review.

Run:  uv run python -m evals.phase2 retrieval
      uv run python -m evals.phase2 rag
      uv run python -m evals.phase2 agent --from evals/results/<phase0 file>.json
      uv run python -m evals.phase2 generate --size 10
RAGAS talks to Ollama through its OpenAI-compatible endpoint (OLLAMA_HOST, default http://localhost:11434).
Judge model: JUDGE_MODEL (default qwen3:8b).
"""
import argparse
import asyncio
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from evals.graders import answer_text  # noqa: E402
from evals.harness import EVAL_DIR, load_cases  # noqa: E402
from shop.tools import POLICY_K, build_policy_store, policy_chunks  # noqa: E402

CHUNKINGS = ("section", "sentence", "sentence+title")
KS = (1, 3, 5)


def load_rag_cases() -> list[dict]:
    return load_cases(EVAL_DIR / "rag_dataset.jsonl")


def judge_name() -> str:
    return os.getenv("JUDGE_MODEL", "qwen3:8b")


def ragas_models():
    """RAGAS 0.4 metrics take an Instructor-wrapped OpenAI client; Ollama serves the same API under /v1."""
    from openai import AsyncOpenAI
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    client = AsyncOpenAI(base_url=f"{host.rstrip('/')}/v1", api_key="ollama")  # key unused, but required
    llm = llm_factory(judge_name(), client=client, temperature=0)
    embeddings = embedding_factory("openai", model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"), client=client)
    return llm, embeddings


def save(name: str, payload: dict) -> None:
    out = EVAL_DIR / "results" / f"{datetime.now():%Y%m%d-%H%M%S}_phase2_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


# ---------- retrieval: ID-based, deterministic ----------

def retrieval_scores(retrieved: list[str], reference: list[str]) -> dict | None:
    """Score retrieved section IDs (in rank order) against the sections that answer the question.
    precision: share of retrieved chunks from a relevant section (noise the generator must ignore)
    recall:    share of relevant sections retrieved at all (missing ones = facts the answer can't have)
    mrr:       1 / rank of the first relevant chunk (does the best evidence come first?)
    RAGAS ships the same idea as IDBasedContextPrecision / IDBasedContextRecall; it is two lines of code."""
    if not reference:  # unanswerable question: nothing is relevant, precision/recall are undefined
        return None
    hits = [s in reference for s in retrieved]
    first = next((i for i, h in enumerate(hits, 1) if h), None)
    return {"precision": sum(hits) / len(hits), "recall": len(set(retrieved) & set(reference)) / len(reference),
            "mrr": 1 / first if first else 0.0}


def run_retrieval(cases: list[dict], chunkings=CHUNKINGS, ks=KS, embeddings=None) -> list[dict]:
    rows = []
    for chunking in chunkings:
        store = build_policy_store(chunking, embeddings)
        ranked = {c["id"]: [d.metadata["section"] for d in store.similarity_search(c["question"], k=max(ks))]
                  for c in cases}
        for k in ks:
            per_case = {c["id"]: retrieval_scores(ranked[c["id"]][:k], c["reference_sections"]) for c in cases}
            scored = [s for s in per_case.values() if s]
            rows.append({"chunking": chunking, "k": k, **{m: mean(s[m] for s in scored) for m in ("precision", "recall", "mrr")},
                         "misses": [cid for cid, s in per_case.items() if s and s["recall"] < 1],
                         "retrieved": {cid: r[:k] for cid, r in ranked.items()}})
    return rows


def cmd_retrieval(args) -> None:
    cases = load_rag_cases()
    rows = run_retrieval(cases)
    print(f"{'chunking':<16}{'k':>3}{'precision':>11}{'recall':>8}{'mrr':>7}  questions not fully covered")
    for r in rows:
        current = "  <- agent today" if (r["chunking"], r["k"]) == ("section", POLICY_K) else ""
        print(f"{r['chunking']:<16}{r['k']:>3}{r['precision']:>11.2f}{r['recall']:>8.2f}{r['mrr']:>7.2f}  "
              f"{len(r['misses'])}{current}")
    best = next(r for r in rows if (r["chunking"], r["k"]) == ("section", POLICY_K))
    print("\nMisses for the agent's current setting (section, k=%d):" % POLICY_K)
    by_id = {c["id"]: c for c in cases}
    for cid in best["misses"]:
        print(f"  {cid:<26} wanted {by_id[cid]['reference_sections']}, got {best['retrieved'][cid]}")
    save("retrieval", {"rows": rows})


# ---------- rag: full agent + RAGAS LLM metrics ----------

def split_contexts(tool_output: str) -> list[str]:
    """search_policy returns "[Section]\\ntext" blocks joined by blank lines; RAGAS wants one string per chunk."""
    return [c.strip() for c in re.split(r"\n\n(?=\[)", tool_output) if c.strip()]


def contexts_from_events(events: list[dict]) -> list[str]:
    return [c for e in events if e["type"] == "tool_result" and e["name"] == "search_policy"
            for c in split_contexts(e["content"])]


def section_of(context: str) -> str:
    return context[1:context.index("]")] if context.startswith("[") else ""


async def score_rag(metrics: dict, case: dict, response: str, contexts: list[str]) -> dict:
    q, ref = case["question"], case["reference"]
    calls = {"answer_relevancy": lambda m: m.ascore(user_input=q, response=response)}
    if contexts:  # these three are about the retrieved text; without retrieval they are meaningless
        calls |= {
            "faithfulness": lambda m: m.ascore(user_input=q, response=response, retrieved_contexts=contexts),
            "context_precision": lambda m: m.ascore(user_input=q, reference=ref, retrieved_contexts=contexts),
            "context_recall": lambda m: m.ascore(user_input=q, retrieved_contexts=contexts, reference=ref),
        }
    scores = {}
    for name, call in calls.items():
        try:
            scores[name] = round((await call(metrics[name])).value, 3)
        except Exception as e:  # small judges sometimes return invalid JSON; record and move on
            scores[name] = None
            print(f"      {name} failed: {e!r:.160}")
    return scores


def cmd_rag(args) -> None:
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness

    from shop.graph import build_graph, run_turn

    cases = [c for c in load_rag_cases() if not args.only or args.only in c["id"]]
    llm, embeddings = ragas_models()
    metrics = {"faithfulness": Faithfulness(llm=llm), "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
               "context_precision": ContextPrecisionWithReference(llm=llm), "context_recall": ContextRecall(llm=llm)}
    graph = build_graph()
    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}", flush=True)
        events = list(run_turn(graph, case["question"], str(uuid.uuid4())))
        response, contexts = answer_text(events), contexts_from_events(events)
        queries = [e["args"].get("query") for e in events if e["type"] == "tool_call" and e["name"] == "search_policy"]
        ids = retrieval_scores([section_of(c) for c in contexts], case["reference_sections"]) if contexts else None
        scores = asyncio.run(score_rag(metrics, case, response, contexts))
        rows.append({"id": case["id"], "queries": queries, "sections": [section_of(c) for c in contexts],
                     "id_recall": ids and ids["recall"], "response": response, "scores": scores})

    names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    print(f"\n{'case':<26}{'faith':>7}{'relev':>7}{'c.prec':>8}{'c.rec':>7}{'id.rec':>8}  search query")
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "  -"
    for r in rows:
        s = r["scores"]
        print(f"{r['id']:<26}{fmt(s.get('faithfulness')):>7}{fmt(s.get('answer_relevancy')):>7}"
              f"{fmt(s.get('context_precision')):>8}{fmt(s.get('context_recall')):>7}{fmt(r['id_recall']):>8}  {r['queries']}")
    print(f"\n{'mean':<26}" + "".join(f"{fmt(mean(r['scores'].get(n) for r in rows)):>{w}}"
                                     for n, w in zip(names, (7, 7, 8, 7))) + f"{fmt(mean(r['id_recall'] for r in rows)):>8}")
    print("\nRead it as: low context_recall -> retrieval missed facts; good recall but low faithfulness -> the agent"
          "\nadded claims the chunks don't support; low relevancy -> the answer drifted from the question.")
    no_search = [r["id"] for r in rows if not r["queries"]]
    if no_search:
        print(f"No search_policy call (answered without retrieval): {no_search}")
    save(f"rag_judge-{judge_name().replace(':', '-')}", {"judge_model": judge_name(), "cases": rows})


# ---------- agent: RAGAS agent metrics on Phase 0 traces ----------

def to_ragas_messages(case: dict, events: list[dict]) -> list:
    from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

    msgs = [HumanMessage(content="\n".join(case["turns"]))]
    for e in events:
        if e["type"] == "tool_call":
            msgs.append(AIMessage(content="", tool_calls=[ToolCall(name=e["name"], args=e["args"])]))
        elif e["type"] == "tool_result":
            msgs.append(ToolMessage(content=e["content"]))
        elif e["type"] == "message":
            msgs.append(AIMessage(content=e["content"]))
    return msgs


def reference_call_lists(case: dict) -> list[list]:
    """One reference list per any_of alternative (taken position-wise, like the Phase 0 oracle)."""
    from ragas.messages import ToolCall

    items = case["expected_tools"]
    width = max([len(i.get("any_of", [i])) for i in items] or [1])
    return [[ToolCall(name=c["name"], args=c.get("args", {}))
             for c in (i["any_of"][min(p, len(i["any_of"]) - 1)] if "any_of" in i else i for i in items)]
            for p in range(width)]


def project_args(case: dict, events: list[dict]) -> list[dict]:
    """Keep only the arg keys the dataset specifies for each tool. RAGAS compares args as given: an unspecified
    search `query` scores as a mismatch against {}, and ToolCallF1 needs exact args, so an extra refund `reason`
    zeroes it. Same rule as the Phase 1 matcher: grade what the dataset defines, nothing else."""
    keys: dict[str, set] = {}
    for item in case["expected_tools"]:
        for alt in item.get("any_of", [item]):
            keys.setdefault(alt["name"], set()).update(alt.get("args", {}))
    return [{**e, "args": {k: v for k, v in e["args"].items() if k in keys[e["name"]]}}
            if e["type"] == "tool_call" and e["name"] in keys else e for e in events]


async def score_tool_calls(case: dict, events: list[dict]) -> dict:
    """Best score over the valid reference paths. Non-LLM, deterministic."""
    from ragas.metrics.collections import ToolCallAccuracy, ToolCallF1

    msgs = to_ragas_messages(case, project_args(case, events))
    out = {}
    for name, metric in (("tool_call_accuracy", ToolCallAccuracy()), ("tool_call_f1", ToolCallF1())):
        values = [(await metric.ascore(user_input=msgs, reference_tool_calls=ref)).value
                  for ref in reference_call_lists(case)]
        out[name] = round(max(values), 3)
    return out


def cmd_agent(args) -> None:
    from ragas.metrics.collections import AgentGoalAccuracyWithReference

    if not args.source:
        raise SystemExit("agent needs --from <Phase 0 results JSON>")
    cases = {c["id"]: c for c in load_cases()}
    runs = [r for r in json.loads(Path(args.source).read_text(encoding="utf-8"))["cases"]
            if not r.get("error") and (not args.only or args.only in r["id"])]
    goal = None if args.no_llm else AgentGoalAccuracyWithReference(llm=ragas_models()[0])
    rows = []
    for i, run in enumerate(runs, 1):
        case = cases[run["id"]]
        print(f"[{i}/{len(runs)}] {run['id']}", flush=True)
        scores = asyncio.run(score_tool_calls(case, run["events"])) if case.get("expected_tools") else {}
        if goal:
            try:
                scores["goal_accuracy"] = asyncio.run(goal.ascore(
                    user_input=to_ragas_messages(case, run["events"]), reference=case["reference"])).value
            except Exception as e:
                scores["goal_accuracy"] = None
                print(f"      goal_accuracy failed: {e!r:.160}")
        recall = run["checks"].get("tools", {}).get("passed")
        rows.append({"id": run["id"], "phase0_tool_recall": recall, "scores": scores})

    print(f"\n{'case':<28}{'p0 recall':>10}{'acc':>6}{'f1':>6}{'goal':>6}")
    fmt = lambda v: "  -" if v is None else f"{float(v):.2f}"
    for r in rows:
        s = r["scores"]
        print(f"{r['id']:<28}{fmt(r['phase0_tool_recall']):>10}{fmt(s.get('tool_call_accuracy')):>6}"
              f"{fmt(s.get('tool_call_f1')):>6}{fmt(s.get('goal_accuracy')):>6}")
    for name in ("tool_call_accuracy", "tool_call_f1", "goal_accuracy"):
        print(f"mean {name:<20} {fmt(mean(r['scores'].get(name) for r in rows))}")
    print("\ntool_call_accuracy is sequence-strict: one extra call (get_order before cancel_order) scores 0."
          "\ntool_call_f1 gives partial credit (2 calls, 1 expected -> 0.67). Phase 0 recall ignores extras.")
    save("agent", {"judge_model": None if args.no_llm else judge_name(), "cases": rows})


# ---------- generate: synthetic test set ----------

def cmd_generate(args) -> None:
    from langchain_core.documents import Document
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.testset import TestsetGenerator

    # The generator's document transforms (headline extraction, splitting, summaries) need a long document,
    # so give it the whole policy file as one doc rather than our small per-section chunks.
    text = "\n\n".join(f"# {d.metadata['section']}\n{d.page_content.partition(chr(10))[2]}" for d in policy_chunks())
    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(ChatOllama(model=judge_name(), temperature=0, num_ctx=8192, reasoning=False)),
        embedding_model=LangchainEmbeddingsWrapper(OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))),
    )
    testset = generator.generate_with_langchain_docs([Document(page_content=text)], testset_size=args.size,
                                                     raise_exceptions=False)
    out = EVAL_DIR / "rag_synthetic.jsonl"
    rows = testset.to_pandas().to_dict(orient="records")
    out.write_text("".join(json.dumps({k: v for k, v in r.items() if k in ("user_input", "reference", "reference_contexts",
                                                                          "synthesizer_name")}, default=str) + "\n"
                           for r in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} generated questions to {out}.\nReview them by hand: keep the good ones, fix references,"
          " add reference_sections, then move them into evals/rag_dataset.jsonl.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["retrieval", "rag", "agent", "generate"])
    parser.add_argument("--from", dest="source", help="Phase 0 results JSON (agent command)")
    parser.add_argument("--only", help="only cases whose id contains this text")
    parser.add_argument("--no-llm", action="store_true", help="agent command: skip the LLM goal-accuracy metric")
    parser.add_argument("--size", type=int, default=10, help="generate command: number of questions")
    args = parser.parse_args()
    {"retrieval": cmd_retrieval, "rag": cmd_rag, "agent": cmd_agent, "generate": cmd_generate}[args.command](args)


if __name__ == "__main__":
    main()
