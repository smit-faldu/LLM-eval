import uuid

from dotenv import load_dotenv

load_dotenv()

from shop.db import DB_PATH, init_db  # noqa: E402
from shop.graph import build_graph  # noqa: E402


def main():
    if not DB_PATH.exists():
        init_db()
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("Shop support. Type 'quit' to exit, 'reset' to reseed the DB.")
    while (text := input("\nyou> ").strip()).lower() != "quit":
        if text.lower() == "reset":
            init_db()
            print("DB reseeded.")
            continue
        if not text:
            continue
        # subgraphs=True also streams the specialists' internal tool calls, handy for seeing the trajectory.
        for namespace, update in graph.stream(
            {"messages": [("user", text)], "hops": 0}, config, stream_mode="updates", subgraphs=True
        ):
            for node, data in update.items():
                for msg in (data or {}).get("messages", []):
                    if namespace and getattr(msg, "tool_calls", None):
                        for call in msg.tool_calls:
                            print(f"  [{namespace[0].split(':')[0]}] tool {call['name']}({call['args']})")
                    elif not namespace:
                        print(f"\n{msg.name}> {msg.text}")
            if not namespace and "supervisor" in update:
                print(f"  [supervisor] -> {update['supervisor']['next']}")


if __name__ == "__main__":
    main()
