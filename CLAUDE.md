# CLAUDE.md

Learning project: build a multi-agent LLM app, then learn LLM/agent evaluation on it step by step.
Phase roadmap and status live in `PLAN.md`; update it whenever a phase starts or finishes.

## Working with the user
- The user is learning evals. After every phase, explain **what was built and why** (concepts, design choices,
  how to read the results), not just a change list.
- Fully local / open source. No LangSmith or other eval cloud. Eval libraries: own harness (Phase 0),
  openevals/agentevals, RAGAS, DeepEval (set `DEEPEVAL_TELEMETRY_OPT_OUT=YES`).
- The user runs the agent in Google Colab with Ollama `gemma4:12b`. This Windows machine has no Ollama
  (8 GB RAM, no GPU), so verify locally with offline tests and fakes, and say when something was not run live.

## App
E-commerce support agent ("electronics shop"), Python 3.14, uv.
- `shop/db.py` SQLite schema + deterministic seed (`init_db()` rebuilds `shop.db`; dates relative to today).
  Customers 1-8 and orders 1001-1020 are hand-written edge cases that tests and eval cases depend on:
  **do not change them**. Rest is generated with `random.Random(42)`.
- `shop/tools.py` LangChain `@tool`s. Business rules (cancel only pending, refund only delivered ≤30 days,
  one refund per order, ticket categories) are enforced in code, not prompts, so evals can test them.
  Refund rules live only in `_refund_blocked_reason`; `get_order` exposes `refund_eligible` so the model never
  does date math (live eval showed gemma4 calling 31 days "within 30").
- `shop/policies.md` policy docs; `search_policy` = InMemoryVectorStore + OllamaEmbeddings (`nomic-embed-text`).
- `shop/graph.py` LangGraph: `supervisor` planner (structured output `Plan` of `Task`s) → runs
  `order_agent` / `product_agent` / `policy_agent` (`create_agent`) in plan order. Each specialist sees only
  its own sub-request plus earlier specialists' results. `run_turn()` yields trace events:
  `plan`, `tool_call`, `tool_result`, `message`; the CLI, API, and evals all consume these.
- `main.py` CLI; `app.py` FastAPI (`/api/tables`, `/api/chat`, `/api/reset`, `/api/policies`) serving
  `static/index.html` (data browser + chat with trace).
- Model: `ChatOllama`, env `OLLAMA_MODEL` (default `gemma4:12b`), `OLLAMA_EMBED_MODEL`.

## Evals
- `evals/dataset.jsonl` cases: `id, category, turns, expected_agents, expected_tools[{name, args?}],
  db_checks[{sql, expect}], must_include (str or list of alternatives), must_not_include`.
  An expected_tools item can be `{"any_of": [call, ...]}` when several paths are valid.
  Every case has a `reference` answer (used by LLM judges); keep it consistent with the seed DB.
- `evals/graders.py` deterministic graders; `evals/harness.py` runner + report, results to `evals/results/`.
- `evals/phase1.py` agentevals trajectory match (4 modes, on a calls-only trajectory) + trajectory judge +
  openevals correctness/groundedness judges. Judge = `JUDGE_MODEL` (default `qwen3:8b`), must differ from agent.
  `--from <phase0 json>` reuses saved traces.

## Commands
```
uv run python test_smoke.py          # tool business rules, offline
uv run python test_graph.py          # planner/specialist wiring with fake LLM, offline
uv run python -m evals.test_evals    # graders + dataset ground-truth oracle, offline
uv run python -m evals.test_phase1   # Phase 1 wiring with fake judge, offline
uv run python -m evals.harness [--only TEXT]   # real eval run, needs Ollama
uv run python -m evals.phase1 [--from FILE] [--only TEXT]   # Phase 1 judges, needs Ollama
uv run main.py                       # CLI chat
uv run uvicorn app:app --reload      # web UI on :8000
```
