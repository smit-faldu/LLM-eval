"""Offline check of planner -> specialist wiring with a fake LLM and fake agents. Run: uv run python test_graph.py"""
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import shop.graph as g

seen = []
class FakeLLM:
    def with_structured_output(self, schema):
        def plan(msgs):
            text = [m for m in msgs if m.type == "human"][-2].content
            if "and" in text:
                return g.Plan(tasks=[g.Task(agent="order_agent", request="Cancel order 1005."),
                                     g.Task(agent="policy_agent", request="What is the refund policy?")])
            return g.Plan(reply="Hello!")
        return RunnableLambda(plan)

def fake_agent(llm, tools, system_prompt, name):
    def run(inp):
        seen.append((name, [m.type for m in inp["messages"]], inp["messages"][-1].content))
        return {"messages": [AIMessage(f"{name} done")]}
    return RunnableLambda(run)

g.get_llm = lambda: FakeLLM()
g.create_agent = fake_agent
graph = g.build_graph()
events = list(g.run_turn(graph, "Cancel order 1005 and tell me the refund policy", "t1"))
print([e["type"] for e in events])
assert events[0] == {"type": "plan", "tasks": [{"agent": "order_agent", "request": "Cancel order 1005."},
                                               {"agent": "policy_agent", "request": "What is the refund policy?"}]}
assert [e["agent"] for e in events if e["type"] == "message"] == ["order_agent", "policy_agent"]
assert seen[0] == ("order_agent", ["human"], "Cancel order 1005.")  # only own sub-request
assert "order_agent: order_agent done" in seen[1][2]  # later task sees earlier result
events = list(g.run_turn(graph, "hi", "t1"))
assert events[-1] == {"type": "message", "agent": "supervisor", "content": "Hello!"}
print("graph wiring ok")
