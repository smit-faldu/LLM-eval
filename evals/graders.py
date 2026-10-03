"""Deterministic (code-based) graders. Each takes a dataset case plus what the agent did and returns
(passed, detail), or None when the case doesn't specify that check. No LLM involved: same input, same score."""


def planned_agents(events: list[dict]) -> list[str]:
    plan = next((e for e in events if e["type"] == "plan"), {"tasks": []})
    agents = [t["agent"] for t in plan["tasks"]]
    # Two tasks for the same agent in a row (e.g. two order questions) count as one step.
    return [a for i, a in enumerate(agents) if i == 0 or a != agents[i - 1]]


def tool_calls(events: list[dict]) -> list[dict]:
    return [e for e in events if e["type"] == "tool_call"]


def answer_text(events: list[dict]) -> str:
    return "\n".join(e["content"] for e in events if e["type"] == "message")


def _args_match(expected: dict, actual: dict) -> bool:
    # Compare as lowercase strings: models sometimes send 1003 as "1003".
    return all(str(actual.get(k)).strip().lower() == str(v).strip().lower() for k, v in expected.items())


def grade_plan(case: dict, events: list[dict]):
    if "expected_agents" not in case:
        return None
    actual = planned_agents(events)
    return actual == case["expected_agents"], f"expected {case['expected_agents']}, got {actual}"


def grade_tools(case: dict, events: list[dict]):
    """Recall gate: every expected call (name + given args) must appear, in any order.
    Extra calls (e.g. get_order before cancel_order) don't fail the case; they lower precision instead."""
    if "expected_tools" not in case:
        return None
    calls, expected = tool_calls(events), case["expected_tools"]
    if not expected:
        return not calls, f"expected no tool calls, got {[c['name'] for c in calls]}"
    missing = [
        exp for exp in expected
        if not any(c["name"] == exp["name"] and _args_match(exp.get("args", {}), c["args"]) for c in calls)
    ]
    got = [f"{c['name']}({c['args']})" for c in calls]
    return not missing, f"missing {missing}; got {got}" if missing else f"got {got}"


def tool_precision(case: dict, events: list[dict]) -> float | None:
    """Share of actual calls that were expected. Informational only, not a pass gate."""
    calls = tool_calls(events)
    if "expected_tools" not in case or not calls:
        return None
    names = {e["name"] for e in case["expected_tools"]}
    return round(sum(c["name"] in names for c in calls) / len(calls), 2)


def grade_db(case: dict, conn):
    if "db_checks" not in case:
        return None
    failures = []
    for check in case["db_checks"]:
        row = conn.execute(check["sql"]).fetchone()
        value = row[0] if row else None
        if value != check["expect"]:
            failures.append(f"{check['sql']} -> {value!r}, expected {check['expect']!r}")
    return not failures, "; ".join(failures) or "all checks ok"


def grade_must_include(case: dict, events: list[dict]):
    """Each item is a string, or a list of alternatives where any one is enough. Case-insensitive."""
    if "must_include" not in case:
        return None
    text = answer_text(events).lower()
    missing = [item for item in case["must_include"]
               if not any(alt.lower() in text for alt in (item if isinstance(item, list) else [item]))]
    return not missing, f"missing {missing}" if missing else "all present"


def grade_must_not_include(case: dict, events: list[dict]):
    if "must_not_include" not in case:
        return None
    text = answer_text(events).lower()
    found = [s for s in case["must_not_include"] if s.lower() in text]
    return not found, f"found forbidden {found}" if found else "none found"


def grade_case(case: dict, events: list[dict], conn) -> dict:
    results = {
        "plan": grade_plan(case, events),
        "tools": grade_tools(case, events),
        "db": grade_db(case, conn),
        "must_include": grade_must_include(case, events),
        "must_not_include": grade_must_not_include(case, events),
    }
    checks = {name: {"passed": r[0], "detail": r[1]} for name, r in results.items() if r is not None}
    return {
        "passed": all(c["passed"] for c in checks.values()),
        "checks": checks,
        "tool_precision": tool_precision(case, events),
    }
