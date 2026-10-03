"""Offline checks for Phase 0 (no LLM). Run: uv run python -m evals.test_evals

1. Graders score hand-made event lists correctly.
2. Oracle check: for every case, run the expected tool calls directly against a fresh DB and confirm the
   case's db_checks pass. This proves the dataset's ground truth is right, so a failing eval means the
   agent is wrong, not the test.
"""
from evals.graders import grade_case
from evals.harness import load_cases
from shop import tools
from shop.db import get_conn, init_db

# ---------- 1. graders ----------
case = {"expected_agents": ["order_agent", "policy_agent"],
        "expected_tools": [{"name": "cancel_order", "args": {"order_id": 1005}}, {"name": "search_policy"}],
        "must_include": ["cancelled", ["30 days", "30-day"]], "must_not_include": ["admin"]}
good = [
    {"type": "plan", "tasks": [{"agent": "order_agent", "request": "x"}, {"agent": "policy_agent", "request": "y"}]},
    {"type": "tool_call", "agent": "order_agent", "name": "get_order", "args": {"order_id": 1005}},
    {"type": "tool_call", "agent": "order_agent", "name": "cancel_order", "args": {"order_id": "1005"}},
    {"type": "message", "agent": "order_agent", "content": "Order 1005 cancelled."},
    {"type": "tool_call", "agent": "policy_agent", "name": "search_policy", "args": {"query": "refund"}},
    {"type": "message", "agent": "policy_agent", "content": "Refunds within a 30-day window."},
]
r = grade_case(case, good, conn=None)
assert r["passed"], r
assert r["tool_precision"] == 0.67  # get_order was an extra call

bad = [good[0], good[1], {"type": "message", "agent": "order_agent", "content": "Order 1005 cancelled. Refunds take 30 days."}]
r = grade_case(case, bad, conn=None)
assert not r["checks"]["tools"]["passed"]  # cancel_order and search_policy never called
assert r["checks"]["must_include"]["passed"]  # text alone looks fine: this is why we check tools too

r = grade_case({"expected_agents": ["order_agent"]}, [{"type": "plan", "tasks": [{"agent": "order_agent"}] * 2}], None)
assert r["passed"]  # repeated agent collapses to one step
r = grade_case({"expected_agents": [], "expected_tools": []}, [{"type": "plan", "tasks": []}], None)
assert r["passed"]
# any_of: refusing after get_order is as valid as trying issue_refund and being refused
refusal = {"expected_tools": [{"any_of": [{"name": "issue_refund", "args": {"order_id": 1004}},
                                          {"name": "get_order", "args": {"order_id": 1004}}]}],
           "must_not_include": ["is eligible"]}
via_lookup = [{"type": "tool_call", "agent": "order_agent", "name": "get_order", "args": {"order_id": 1004}},
              {"type": "message", "agent": "order_agent", "content": "Order 1004 was already refunded."}]
assert grade_case(refusal, via_lookup, None)["passed"]
# The refund-day-31 failure from the first live run: no refund happened (DB fine), but the answer was wrong.
wrong_claim = via_lookup[:1] + [{"type": "message", "agent": "order_agent", "content": "It is eligible for a refund."}]
assert not grade_case(refusal, wrong_claim, None)["checks"]["must_not_include"]["passed"]
print("graders ok")

# ---------- 2. oracle check on the dataset ----------
TOOLS = {t.name: t for t in tools.ORDER_TOOLS + tools.PRODUCT_TOOLS}
problems = []
cases = load_cases()


def _run_path(case: dict, calls: list[dict]) -> list[str]:
    for call in calls:
        if call["name"] not in TOOLS:  # search_policy needs the embedding model; it never writes anyway
            continue
        args = dict(call.get("args", {}))
        if call["name"] == "issue_refund":
            args.setdefault("reason", "eval")
        if call["name"] == "create_support_ticket":
            args.setdefault("description", "eval")
        TOOLS[call["name"]].invoke(args)
    with get_conn() as conn:
        db = grade_case({"db_checks": case["db_checks"]}, [], conn)
    return [] if db["passed"] else [f"{case['id']} via {[c['name'] for c in calls]}: {db['checks']['db']['detail']}"]


for case in cases:
    if "db_checks" not in case:
        continue
    # Every valid path (each any_of alternative) must leave the DB in the expected state.
    width = max([len(c.get("any_of", [c])) for c in case.get("expected_tools", [])] or [1])
    for path in range(width):
        init_db()
        calls = [c["any_of"][min(path, len(c["any_of"]) - 1)] if "any_of" in c else c
                 for c in case.get("expected_tools", [])]
        problems += _run_path(case, calls)
init_db()

assert not problems, "\n".join(problems)
ids = [c["id"] for c in cases]
assert len(ids) == len(set(ids)), "duplicate case ids"
print(f"dataset oracle ok ({len(cases)} cases, {sum('db_checks' in c for c in cases)} with db checks)")
