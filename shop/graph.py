"""Supervisor multi-agent graph: a planner splits each user message into per-specialist sub-tasks,
then LangGraph runs the LangChain tool-calling specialists one after another."""
import os
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from shop.tools import ORDER_TOOLS, POLICY_TOOLS, PRODUCT_TOOLS

MAX_TASKS = 4
AGENTS = ("order_agent", "product_agent", "policy_agent")

SPECIALISTS = {
    "order_agent": (ORDER_TOOLS, (
        "You are the order and account specialist for an electronics shop. You look up orders and customer"
        " profiles, cancel pending orders, issue refunds, and open or list support tickets using your tools. Never invent order data. Only say an action succeeded if the tool says so;"
        " if a tool refuses, explain why. For refund eligibility trust the refund_eligible field; never compute dates yourself. Never answer policy or product questions from memory; only report what your tools return."
    )),
    "product_agent": (PRODUCT_TOOLS, (
        "You are the product specialist for an electronics shop. Use your tools to search the catalog, check"
        " stock and read reviews. Never invent products, prices, stock levels or ratings. Never answer order or policy questions from memory; only report what your tools return."
    )),
    "policy_agent": (POLICY_TOOLS, (
        "You are the policy specialist for an electronics shop. Always call search_policy and answer only from the"
        " returned text. If the policy does not cover the question, say so."
    )),
}

SUPERVISOR_PROMPT = """You plan how to answer a customer of an electronics shop. Split the latest user message
into sub-tasks and assign each to exactly one specialist:
- order_agent: a specific order or customer account: order status, order history, cancellations, refunds,
  customer profile/tier, opening or checking support tickets (warranty, damaged, shipping, billing).
- product_agent: product search, prices, brands, stock availability, warranty length, reviews and ratings.
- policy_agent: general rules: returns, refunds, exchanges, cancellation, shipping, late/damaged deliveries,
  warranty terms, membership tiers, price match, payment, privacy.

Rules:
- One sub-task per distinct question or action. A message with two questions for two specialists gives two tasks.
- Only plan what the user asked for. Do not add a policy_agent task to explain rules unless the user asks about
  a policy; order_agent already explains why an action is refused.
- Keep the user's intent: "refund order X" or "can I get my money back for X" is a refund request
  ("Refund order X."), not just a status check.
- Each `request` must be self-contained: copy order IDs, product IDs, emails and names into it, and resolve
  words like "it" or "that order" using the conversation.
- Order the tasks so that a task needing another's result comes after it.
- If no specialist is needed (greeting, thanks, off-topic), return no tasks and put a short reply in `reply`.

Examples:
"Cancel order 1005 and tell me the refund policy" ->
  [order_agent: "Cancel order 1005."], [policy_agent: "What is the refund policy?"]
"Is the smartwatch ultra in stock and how long is shipping?" ->
  [product_agent: "Is the Smartwatch Ultra in stock?"], [policy_agent: "How long does shipping take?"]
"Where is order 1003?" -> [order_agent: "Where is order 1003? Give status and tracking."]
"Refund order 1002, it is broken" -> [order_agent: "Refund order 1002. Reason: item is broken."]
"Thanks!" -> no tasks, reply: "You're welcome! Anything else?"
"""


class Task(BaseModel):
    agent: Literal["order_agent", "product_agent", "policy_agent"]
    request: str = Field(description="Self-contained instruction for this specialist only.")


class Plan(BaseModel):
    tasks: list[Task] = Field(default_factory=list)
    reply: str = Field(default="", description="Only when tasks is empty.")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tasks: list[dict]  # remaining sub-tasks for this turn, consumed front to back


def _split_turn(messages: list[AnyMessage]) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """(history before the latest user message, specialist answers given so far this turn)."""
    last = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return messages[:last], messages[last + 1:]


def get_llm() -> ChatOllama:
    # reasoning=False turns off thinking output on models that have it: faster, and keeps it out of answers.
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma4:12b"), temperature=0, num_ctx=8192, reasoning=False)


def build_graph(checkpointer=None):
    llm = get_llm()
    planner = llm.with_structured_output(Plan)

    def supervisor(state: State) -> dict:
        # Trailing human turn: models plan more reliably when the prompt ends on a user message.
        plan = planner.invoke([SystemMessage(SUPERVISOR_PROMPT), *state["messages"],
                               HumanMessage("Plan the sub-tasks for my latest message.")])
        update = {"tasks": [t.model_dump() for t in plan.tasks[:MAX_TASKS]]}
        if not plan.tasks:
            update["messages"] = [AIMessage(plan.reply or "How can I help you with your order?", name="supervisor")]
        return update

    def make_node(name: str):
        tools, prompt = SPECIALISTS[name]
        agent = create_agent(llm, tools, system_prompt=prompt, name=name)

        def node(state: State) -> dict:
            task, *rest = state["tasks"]
            history, done = _split_turn(state["messages"])
            request = task["request"]
            if done:  # let later tasks use earlier results, e.g. a customer_id found by another specialist
                earlier = "\n".join(f"- {m.name}: {m.text}" for m in done)
                request += f"\n\nResults from other specialists so far:\n{earlier}"
            # The specialist sees only its own sub-request, so it can't answer parts that belong to others.
            result = agent.invoke({"messages": [*history, HumanMessage(request)]})
            # Keep only the specialist's final answer in shared history; tool traces stay in the subgraph stream.
            return {"messages": [AIMessage(result["messages"][-1].text, name=name)], "tasks": rest}

        return node

    def next_step(state: State) -> str:
        return state["tasks"][0]["agent"] if state["tasks"] else END

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    for name in AGENTS:
        graph.add_node(name, make_node(name))
        graph.add_conditional_edges(name, next_step, [*AGENTS, END])
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", next_step, [*AGENTS, END])
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def run_turn(graph, text: str, thread_id: str):
    """Run one user turn, yielding trace events: the plan, specialist tool calls/results, replies."""
    config = {"configurable": {"thread_id": thread_id}}
    # subgraphs=True also streams the specialists' internal tool calls: the trajectory evals will grade.
    for namespace, update in graph.stream({"messages": [("user", text)]}, config,
                                          stream_mode="updates", subgraphs=True):
        agent = namespace[0].split(":")[0] if namespace else None
        for node, data in update.items():
            if not namespace and node == "supervisor":
                yield {"type": "plan", "tasks": data["tasks"]}
            for msg in (data or {}).get("messages", []):
                if agent:
                    for call in getattr(msg, "tool_calls", None) or []:
                        yield {"type": "tool_call", "agent": agent, "name": call["name"], "args": call["args"]}
                    if msg.type == "tool":
                        yield {"type": "tool_result", "agent": agent, "name": msg.name, "content": msg.text}
                else:
                    yield {"type": "message", "agent": msg.name, "content": msg.text}
