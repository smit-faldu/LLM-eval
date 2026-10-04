"""Offline checks for Phase 1 (no Ollama). Run: uv run python -m evals.test_phase1

Uses a fake judge model that records the prompt it was given, so we test our wiring
(trajectory conversion, match modes, any_of, prompt variables) without a real LLM.
"""
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from evals.harness import load_cases
from evals.phase1 import MODES, build_evaluators, evaluate_case

prompts = []


class FakeJudge(FakeListChatModel):
    responses: list[str] = [""]

    def with_structured_output(self, schema, **kwargs):
        def judge(messages):
            text = "\n".join(m["content"] for m in messages)
            prompts.append(text)
            return {"reasoning": "fake", "score": "WRONG" not in text}
        return RunnableLambda(judge)


evaluators = build_evaluators(FakeJudge())
cases = {c["id"]: c for c in load_cases()}

# Agent looked the order up first, then cancelled ("1007" as a string, as models sometimes send it).
events = [
    {"type": "plan", "tasks": [{"agent": "order_agent", "request": "Cancel order 1007."}]},
    {"type": "tool_call", "agent": "order_agent", "name": "get_order", "args": {"order_id": 1007}},
    {"type": "tool_result", "agent": "order_agent", "name": "get_order", "content": '{"status": "pending"}'},
    {"type": "tool_call", "agent": "order_agent", "name": "cancel_order", "args": {"order_id": "1007"}},
    {"type": "tool_result", "agent": "order_agent", "name": "cancel_order", "content": "Order 1007 cancelled."},
    {"type": "message", "agent": "order_agent", "content": "Order 1007 has been cancelled."},
]
s = evaluate_case(cases["cancel-pending"], events, evaluators)
# Reference is [cancel_order]; actual is [get_order, cancel_order]: only superset accepts the extra call.
assert {m: s[f"match_{m}"]["score"] for m in MODES} == {
    "strict": False, "unordered": False, "subset": False, "superset": True}, s
assert s["correctness"]["score"] and s["groundedness"]["score"] and s["trajectory_judge"]["score"]
assert any("Order 1007 cancelled." in p and "<tool_outputs>" in p for p in prompts)  # context reached the prompt
assert any(cases["cancel-pending"]["reference"] in p for p in prompts)  # reference reached the prompt

# Exactly the reference path: every mode matches.
exact = [events[0], *events[3:]]
s = evaluate_case(cases["cancel-pending"], exact, evaluators)
assert all(s[f"match_{m}"]["score"] for m in MODES), s

# any_of: refusing a refund after get_order matches the get_order alternative.
refusal = [{"type": "tool_call", "agent": "order_agent", "name": "get_order", "args": {"order_id": 1004}},
           {"type": "message", "agent": "order_agent", "content": "Already refunded."}]
assert evaluate_case(cases["refund-already-done"], refusal, evaluators)["match_strict"]["score"]

# Wrong order id fails every mode.
wrong = [{"type": "tool_call", "agent": "order_agent", "name": "cancel_order", "args": {"order_id": 1005}}]
assert not any(evaluate_case(cases["cancel-pending"], wrong, evaluators)[f"match_{m}"]["score"] for m in MODES)

# Safety case has no expected_tools: only the judges run.
s = evaluate_case(cases["safety-pii-other-customer"], [{"type": "message", "agent": "supervisor", "content": "WRONG"}],
                  evaluators)
assert "match_strict" not in s and not s["correctness"]["score"]
assert all("reference" in c for c in cases.values())
print("phase1 wiring ok")
