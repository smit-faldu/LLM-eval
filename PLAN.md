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

## Phase 0: own harness, no eval library  `[~]` baseline done, fixes applied, waiting for re-run
**Learn:** what a test case, metric, threshold, and grader are; deterministic vs judgement; why a fixed seed matters.
- [x] `evals/dataset.jsonl`: 39 cases across order, product, policy, multi_intent, multi_turn, safety, chitchat
- [x] `evals/graders.py`: plan match, tool recall (gate), tool precision (info), DB state, must/must-not include
- [x] `evals/harness.py`: reset DB per case, run turns, grade, report by category and check, save JSON
- [x] `evals/test_evals.py`: grader unit checks + oracle check that the dataset's ground truth is right
- [x] First live run in Colab: baseline 36/39 (see log)
- [x] Fixes from failure analysis: refund eligibility computed in code (`refund_eligible` in get_order),
      planner no longer adds unasked policy tasks, grader `any_of` for multiple valid paths,
      refusal cases must not claim eligibility
- [ ] Re-run in Colab and compare with the baseline

## Phase 1: openevals + agentevals
**Learn:** trajectory evaluation modes, LLM-as-judge prompts, judge bias.
- [ ] Trajectory match (strict / unordered / subset / superset) on tool sequences
- [ ] Trajectory LLM judge for cases without one right path
- [ ] `create_llm_as_judge` correctness, then a custom rubric; use a different model as judge

## Phase 2: RAGAS on the policy agent
**Learn:** retrieval vs generation failures, RAG metrics, experiments.
- [ ] Synthetic test set from `policies.md` (review by hand)
- [ ] Context precision / recall, faithfulness, response relevancy
- [ ] Experiment: change chunking or k, compare scores
- [ ] Agent metrics: ToolCallAccuracy, AgentGoalAccuracy

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
