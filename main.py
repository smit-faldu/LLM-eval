import uuid

from dotenv import load_dotenv

load_dotenv()

from shop.db import DB_PATH, init_db  # noqa: E402
from shop.graph import build_graph, run_turn  # noqa: E402


def main():
    if not DB_PATH.exists():
        init_db()
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    print("Shop support. Type 'quit' to exit, 'reset' to reseed the DB.")
    while (text := input("\nyou> ").strip()).lower() != "quit":
        if text.lower() == "reset":
            init_db()
            print("DB reseeded.")
            continue
        if not text:
            continue
        for event in run_turn(graph, text, thread_id):
            match event["type"]:
                case "route":
                    print(f"  [supervisor] -> {event['next']}")
                case "tool_call":
                    print(f"  [{event['agent']}] tool {event['name']}({event['args']})")
                case "message":
                    print(f"\n{event['agent']}> {event['content']}")


if __name__ == "__main__":
    main()
