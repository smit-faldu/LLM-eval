"""Supervisor multi-agent graph: a LangGraph router dispatching to LangChain tool-calling specialists."""
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

MAX_HOPS = 4
AGENTS = ("order_agent", "product_agent", "policy_agent")

SPECIALISTS = {
    "order_agent": (ORDER_TOOLS, (
        "You are the order specialist for an electronics shop. You look up orders, cancel pending orders and issue"
        " refunds using your tools. Never invent order data. Only say an action succeeded if the tool says so;"
        " if a tool refuses, explain why. Answer only the order-related part of the request."
    )),
    "product_agent": (PRODUCT_TOOLS, (
        "You are the product specialist for an electronics shop. Use your tools to search the catalog and check"
        " stock. Never invent products, prices or stock levels. Answer only the product-related part of the request."
    )),
    "policy_agent": (POLICY_TOOLS, (
        "You are the policy specialist for an electronics shop. Always call search_policy and answer only from the"
        " returned text. If the policy does not cover the question, say so. Answer only the policy part of the request."
    )),
}

SUPERVISOR_PROMPT = """You route customer messages for an electronics shop to specialists:
- order_agent: order status, order history, cancellations, refunds for a specific order.
- product_agent: product search, prices, stock availability.
- policy_agent: general rules about returns, refunds, cancellation, shipping, warranty, payment.

Look at the conversation since the latest user message. Pick the specialist for the next part of the request
that has NOT been answered yet. A request can need several specialists in turn.
Choose FINISH when every part is answered. If you FINISH and no specialist answered the latest user message
(greeting, off-topic), put a short reply in `reply`; otherwise leave `reply` empty."""


class Route(BaseModel):
    next: Literal["order_agent", "product_agent", "policy_agent", "FINISH"]
    reply: str = Field(default="", description="Only when FINISH and no specialist answered.")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next: str
    hops: int


def get_llm() -> ChatOllama:
    # reasoning=False turns off qwen3's <think> output: faster, and keeps it out of answers.
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen3:4b"), temperature=0, num_ctx=8192, reasoning=False)


def build_graph(checkpointer=None):
    llm = get_llm()
    router = llm.with_structured_output(Route)

    def supervisor(state: State) -> dict:
        if state.get("hops", 0) >= MAX_HOPS:
            return {"next": "FINISH"}
        # Trailing human turn: models route more reliably when the prompt ends on a user message.
        route = router.invoke([SystemMessage(SUPERVISOR_PROMPT), *state["messages"],
                               HumanMessage("Which specialist next, or FINISH?")])
        update = {"next": route.next}
        if route.next == "FINISH" and route.reply and isinstance(state["messages"][-1], HumanMessage):
            update["messages"] = [AIMessage(route.reply, name="supervisor")]
        return update

    def make_node(name: str):
        tools, prompt = SPECIALISTS[name]
        agent = create_agent(llm, tools, system_prompt=prompt, name=name)

        def node(state: State) -> dict:
            result = agent.invoke({"messages": state["messages"]})
            answer = result["messages"][-1]
            # Keep only the specialist's final answer in shared history; tool traces stay in the subgraph stream.
            return {"messages": [AIMessage(answer.text, name=name)], "hops": state.get("hops", 0) + 1}

        return node

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    for name in AGENTS:
        graph.add_node(name, make_node(name))
        graph.add_edge(name, "supervisor")
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: END if s["next"] == "FINISH" else s["next"], [*AGENTS, END])
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())
