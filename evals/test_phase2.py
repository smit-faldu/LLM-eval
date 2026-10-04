"""Offline checks for Phase 2 (no Ollama). Run: uv run python -m evals.test_phase2"""
import asyncio

from langchain_core.embeddings import DeterministicFakeEmbedding

from evals.harness import load_cases
from evals.phase2 import (CHUNKINGS, KS, contexts_from_events, load_rag_cases, ragas_models, retrieval_scores,
                          run_retrieval, score_tool_calls, section_of)
from shop.tools import format_policy_hits, policy_chunks

# 1. Dataset oracle: every reference section really exists, ids are unique.
sections = {d.metadata["section"] for d in policy_chunks()}
rag = load_rag_cases()
bad = [(c["id"], s) for c in rag for s in c["reference_sections"] if s not in sections]
assert not bad, bad
assert len({c["id"] for c in rag}) == len(rag)
for chunking in CHUNKINGS:  # every chunking keeps every section reachable
    assert {d.metadata["section"] for d in policy_chunks(chunking)} == sections

# 2. ID-based retrieval math.
assert retrieval_scores(["Shipping", "Payment", "Warranty"], ["Shipping"]) == {"precision": 1 / 3, "recall": 1.0, "mrr": 1.0}
assert retrieval_scores(["Payment", "Shipping"], ["Shipping", "Membership Tiers"]) == {"precision": 0.5, "recall": 0.5, "mrr": 0.5}
assert retrieval_scores(["Payment"], ["Shipping"])["mrr"] == 0.0
assert retrieval_scores(["Payment"], []) is None  # unanswerable question is not scored

# 3. Experiment grid wiring with fake embeddings (scores are meaningless, shape is not).
rows = run_retrieval(rag, embeddings=DeterministicFakeEmbedding(size=32))
assert len(rows) == len(CHUNKINGS) * len(KS)
assert all(len(r["retrieved"]["rag-emi"]) == r["k"] for r in rows)

# 4. Contexts parsed back out of a real search_policy output.
docs = policy_chunks()[:3]
events = [{"type": "tool_result", "name": "search_policy", "content": format_policy_hits(docs)},
          {"type": "tool_result", "name": "get_order", "content": "[not policy]"}]
ctx = contexts_from_events(events)
assert len(ctx) == 3 and [section_of(c) for c in ctx] == [d.metadata["section"] for d in docs]

# 5. RAGAS tool-call metrics (non-LLM) on hand-made traces.
cases = {c["id"]: c for c in load_cases()}
lookup_then_cancel = [
    {"type": "tool_call", "name": "get_order", "args": {"order_id": 1007}},
    {"type": "tool_result", "name": "get_order", "content": "{}"},
    {"type": "tool_call", "name": "cancel_order", "args": {"order_id": 1007}},
    {"type": "tool_result", "name": "cancel_order", "content": "ok"},
    {"type": "message", "agent": "order_agent", "content": "Cancelled."},
]
s = asyncio.run(score_tool_calls(cases["cancel-pending"], lookup_then_cancel))
assert s == {"tool_call_accuracy": 0.0, "tool_call_f1": 0.667}, s  # strict sequence vs partial credit
s = asyncio.run(score_tool_calls(cases["cancel-pending"], lookup_then_cancel[2:]))
assert s == {"tool_call_accuracy": 1.0, "tool_call_f1": 1.0}, s
# any_of: refusing via get_order is a valid path for refund-already-done.
s = asyncio.run(score_tool_calls(cases["refund-already-done"],
                                 [{"type": "tool_call", "name": "get_order", "args": {"order_id": 1004}}]))
assert s["tool_call_accuracy"] == 1.0, s

# Regression from the real Phase 0 traces: unspecified / extra args must not zero the score.
search = [{"type": "tool_call", "name": "search_products", "args": {"query": "ultrawide"}}]
assert asyncio.run(score_tool_calls(cases["product-price"], search)) == {"tool_call_accuracy": 1.0, "tool_call_f1": 1.0}
refund = [{"type": "tool_call", "name": "issue_refund", "args": {"order_id": 1002, "reason": "faulty"}}]
assert asyncio.run(score_tool_calls(cases["refund-ok"], refund)) == {"tool_call_accuracy": 1.0, "tool_call_f1": 1.0}
wrong = [{"type": "tool_call", "name": "issue_refund", "args": {"order_id": 1001, "reason": "faulty"}}]
assert asyncio.run(score_tool_calls(cases["refund-ok"], wrong))["tool_call_f1"] == 0.0

# 6. RAGAS model objects build without contacting Ollama.
llm, emb = ragas_models()
assert llm is not None and emb is not None
print("phase2 wiring ok")
