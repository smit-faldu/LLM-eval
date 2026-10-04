# Eval Learning Plan

Goal: learn LLM and agent evaluation step by step on our own shop support agent. Fully local, open source.
Status: `[x]` done, `[~]` in progress, `[ ]` not started.

## Build the agent
- [x] SQLite shop DB with deterministic seed and edge cases
- [x] Tools with business rules in code
- [x] Policy RAG (in-memory vector store)
- [x] LangGraph supervisor + 3 LangChain specialists (now plan-then-execute)
- [x] Ollama (gemma4:12b in Colab)
- [x] FastAPI backend + web UI with trace

## Phase 0: own harness, no eval library  `[x]`
**Learn:** what a test case, metric, threshold, and grader are; deterministic vs judgement; why a fixed seed matters.
- [x] `evals/dataset.jsonl`: 39 cases across order, product, policy, multi_intent, multi_turn, safety, chitchat
- [x] `evals/graders.py`: plan match, tool recall (gate), tool precision (info), DB state, must/must-not include
- [x] `evals/harness.py`: reset DB per case, run turns, grade, report by category and check, save JSON
- [x] `evals/test_evals.py`: grader unit checks + oracle check that the dataset's ground truth is right
- [x] First live run in Colab: baseline 36/39 (see log)
- [x] Fixes from failure analysis: refund eligibility computed in code (`refund_eligible` in get_order),
      planner no longer adds unasked policy tasks, grader `any_of` for multiple valid paths,
      refusal cases must not claim eligibility
- [x] Re-run in Colab after fixes: 39/39 (JSON not saved)

## Phase 1: openevals + agentevals  `[x]`
**Learn:** trajectory evaluation modes, LLM-as-judge prompts, judge bias.
- [x] `reference` answer added to all 39 cases (checked against the DB)
- [x] `evals/phase1.py`: trajectory match in all 4 modes, trajectory LLM judge (reference-free),
      CORRECTNESS judge (vs reference), custom GROUNDEDNESS rubric (vs tool outputs), code-vs-judge agreement table
- [x] `evals/test_phase1.py`: offline wiring test with a fake judge
- [x] Live run 1 (agent and judge both gemma4:12b): found a matcher bug (agentevals swaps sides in
      subset/unordered, so tool_args "superset" became "subset") -> per-case arg-key overrides; 3 over-specified
      references trimmed to must-have facts; judge hallucinated a missing "$59.99" that was present
- [x] `--canary`: judge also grades a number-corrupted copy of each answer; must FAIL it (tests the judge)
- [x] Phase closed by user; judge comparison with `--canary` can be revisited in Phase 4

## Phase 2: RAGAS on the policy agent  `[~]` built, waiting for live runs
**Learn:** retrieval vs generation failures, RAG metrics, experiments.
- [x] `evals/rag_dataset.jsonl`: 22 hand-written policy questions with reference + reference_sections
      (incl. multi-section and one unanswerable)
- [x] `shop/tools.py`: `policy_chunks(chunking)` / `build_policy_store` so chunking and k can be varied
- [x] `phase2.py retrieval`: chunking {section, sentence, sentence+title} x k {1,3,5}, ID-based precision/recall/MRR
- [x] `phase2.py rag`: full agent, RAGAS Faithfulness, AnswerRelevancy, ContextPrecisionWithReference, ContextRecall
- [x] `phase2.py agent`: ToolCallAccuracy, ToolCallF1 (non-LLM) + AgentGoalAccuracyWithReference
- [x] `phase2.py generate`: RAGAS TestsetGenerator -> evals/rag_synthetic.jsonl for hand review
- [x] Ran `agent --no-llm` locally on Phase 0 traces: first 0.46 acc (args artifact: unspecified/extra args),
      after projecting args to dataset keys 0.91 acc / 0.97 F1; same 3 lookup-first cases as Phase 1 strict
- [ ] Live in Colab: retrieval grid, rag metrics, synthetic set review; pick a retrieval setting from the numbers

## Phase 3: DeepEval suite
**Learn:** evals as pytest/CI, thresholds, custom criteria.
- [ ] pytest suite run with `deepeval test run`
- [ ] ToolCorrectness, TaskCompletion, Faithfulness, AnswerRelevancy
- [ ] G-Eval custom criteria (refund policy adherence, no PII leaks)
- [ ] Multi-turn conversational test cases

## Phase 4: model comparison + judge calibration
**Learn:** variance, judge-human agreement, real vs noise differences.
- [ ] qwen3:4b vs gemma4:12b on all suites (accuracy, latency)
- [ ] Hand-label ~20 cases, measure judge agreement
- [ ] Repeat runs ×3 to measure flakiness

## Phase 5: safety + robustness
- [ ] Prompt injection, PII probing, typos, Hinglish, vague requests
- [ ] Optional: deepteam red-teaming

## Phase 6 (optional): local tracing UI (Phoenix or self-hosted Langfuse)

## Results log
| Date | Model | Suite | Score | Notes |
|---|---|---|---|---|
| 2026-10-03 | gemma4:12b | Phase 0 (39) | 36/39 (92%) | Baseline. Fails: refund-day-31 (model date math said 31 days was eligible), refund-ok (extra policy task, 63s), refund-already-done (valid path, dataset too strict) |
| 2026-10-03 | gemma4:12b | Phase 0 (39) | 39/39 (100%) | After fixes (refund_eligible, planner rules, any_of). JSON not saved |
| 2026-10-04 | gemma4:12b (judge gemma4:12b) | Phase 1 (39) | strict 34/37, superset 37/37, traj judge 39/39, correctness 36/39, groundedness 39/39 | Same-model judge. unordered/subset 16/37 was a matcher bug (fixed, now 34/37). 3 correctness fails = judge too strict/hallucinated, not agent |
| 2026-10-04 | gemma4:12b traces | Phase 2 agent (RAGAS, no LLM) | ToolCallAccuracy 0.91, ToolCallF1 0.97 | Only misses: get_order before write (cancel-pending, refund-ok, refund-day-30) |
