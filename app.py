"""FastAPI backend: browse every DB table, read policies, chat with the agent. UI served at /."""
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, PlainTextResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from shop.db import DB_PATH, get_conn, init_db, table_names  # noqa: E402
from shop.graph import build_graph, run_turn  # noqa: E402

ROOT = Path(__file__).parent
if not DB_PATH.exists():
    init_db()
graph = build_graph()
app = FastAPI(title="Shop Support Agent")


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/tables")
def tables():
    with get_conn() as conn:
        return [{"name": t, "count": conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]} for t in table_names()]


@app.get("/api/tables/{name}")
def table_rows(name: str):
    # Table name can't be a bound parameter, so only allow names that really exist.
    if name not in table_names():
        raise HTTPException(404, f"Unknown table '{name}'")
    with get_conn() as conn:
        cur = conn.execute(f'SELECT * FROM "{name}" ORDER BY rowid DESC')  # ponytail: no paging, add LIMIT/OFFSET past ~10k rows
        return {"columns": [c[0] for c in cur.description], "rows": [list(r) for r in cur.fetchall()]}


@app.get("/api/policies", response_class=PlainTextResponse)
def policies():
    return (ROOT / "shop" / "policies.md").read_text(encoding="utf-8")


@app.post("/api/chat")
def chat(body: ChatIn):
    thread_id = body.thread_id or str(uuid.uuid4())
    try:
        events = list(run_turn(graph, body.message, thread_id))
    except Exception as e:  # model server down, bad model name, etc. - surface it in the UI
        raise HTTPException(502, f"Agent error: {e}") from e
    return {"thread_id": thread_id, "events": events}


@app.post("/api/reset")
def reset():
    init_db()
    return {"ok": True}
