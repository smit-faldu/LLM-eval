"""Agent tools. Business rules live here, not in prompts, so evals can check the agent can't bypass them."""
import os
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from shop.db import get_conn

REFUND_WINDOW_DAYS = 30


def _order_dict(conn, order_id: int) -> dict | None:
    order = conn.execute(
        "SELECT o.*, c.name AS customer, c.email, c.tier FROM orders o JOIN customers c ON c.id = o.customer_id"
        " WHERE o.id = ?",
        (order_id,),
    ).fetchone()
    if order is None:
        return None
    items = conn.execute(
        "SELECT p.name, i.quantity, i.unit_price FROM order_items i JOIN products p ON p.id = i.product_id"
        " WHERE i.order_id = ?",
        (order_id,),
    ).fetchall()
    refund = conn.execute("SELECT amount, reason, created_date FROM refunds WHERE order_id = ?", (order_id,)).fetchone()
    result = dict(order)
    result["items"] = [dict(i) for i in items]
    result["total"] = round(sum(i["quantity"] * i["unit_price"] for i in items), 2)  # items only, excludes shipping
    result["refund"] = dict(refund) if refund else None
    if order["delivered_date"]:
        result["days_since_delivery"] = (date.today() - date.fromisoformat(order["delivered_date"])).days
    # Computed here so the model never does date math or rule checks itself.
    blocked = _refund_blocked_reason(result)
    result["refund_eligible"] = blocked is None
    if blocked:
        result["refund_blocked_reason"] = blocked
    return result


def _refund_blocked_reason(order: dict) -> str | None:
    """The one place the refund rules live; used by get_order (to inform) and issue_refund (to enforce)."""
    if order["status"] != "delivered":
        return f"status is '{order['status']}', only delivered orders can be refunded"
    if order["refund"]:
        return f"already refunded on {order['refund']['created_date']}"
    if order["days_since_delivery"] > REFUND_WINDOW_DAYS:
        return (f"delivered {order['days_since_delivery']} days ago ({order['delivered_date']}),"
                f" outside the {REFUND_WINDOW_DAYS}-day refund window")
    return None


# ---------- order tools ----------

@tool
def get_order(order_id: int) -> dict | str:
    """Look up one order by its numeric ID: status, dates, items, total, any refund, and refund_eligible
    (with refund_blocked_reason when not eligible). Trust refund_eligible; do not recompute it."""
    with get_conn() as conn:
        return _order_dict(conn, order_id) or f"Order {order_id} not found."


@tool
def list_customer_orders(email: str) -> list[dict] | str:
    """List all orders (id, status, order_date) for a customer email address."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT o.id, o.status, o.order_date FROM orders o JOIN customers c ON c.id = o.customer_id"
            " WHERE lower(c.email) = lower(?) ORDER BY o.order_date DESC",
            (email.strip(),),
        ).fetchall()
    return [dict(r) for r in rows] or f"No orders found for {email}."


@tool
def cancel_order(order_id: int) -> str:
    """Cancel an order. Only orders with status 'pending' can be cancelled."""
    with get_conn() as conn:
        order = conn.execute("SELECT status FROM orders WHERE id = ?", (order_id,)).fetchone()
        if order is None:
            return f"Order {order_id} not found."
        if order["status"] != "pending":
            return f"Cannot cancel order {order_id}: status is '{order['status']}', only pending orders can be cancelled."
        conn.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (order_id,))
    return f"Order {order_id} cancelled. Full refund in 3 business days."


@tool
def issue_refund(order_id: int, reason: str) -> str:
    """Refund a delivered order within 30 days of delivery. Each order can be refunded once."""
    with get_conn() as conn:
        order = _order_dict(conn, order_id)
        if order is None:
            return f"Order {order_id} not found."
        if not order["refund_eligible"]:
            return f"Cannot refund order {order_id}: {order['refund_blocked_reason']}."
        conn.execute(
            "INSERT INTO refunds (order_id, amount, reason, created_date) VALUES (?, ?, ?, ?)",
            (order_id, order["total"], reason, date.today().isoformat()),
        )
    return f"Refund of ${order['total']:.2f} issued for order {order_id}. Arrives in 5-7 business days."


@tool
def get_customer(email: str) -> dict | str:
    """Look up a customer profile by email: name, phone, city, membership tier, joined date."""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM customers WHERE lower(email) = lower(?)", (email.strip(),)).fetchone()
    return dict(row) if row else f"No customer with email {email}."


TICKET_CATEGORIES = ("warranty", "damaged", "shipping", "billing", "other")


@tool
def create_support_ticket(category: str, description: str, order_id: int | None = None,
                          email: str | None = None) -> str:
    """Open a support ticket. category: warranty, damaged, shipping, billing or other.
    Give order_id when the issue is about an order, otherwise the customer's email."""
    if category not in TICKET_CATEGORIES:
        return f"Invalid category '{category}'. Use one of: {', '.join(TICKET_CATEGORIES)}."
    with get_conn() as conn:
        if order_id is not None:
            row = conn.execute("SELECT customer_id FROM orders WHERE id = ?", (order_id,)).fetchone()
            if row is None:
                return f"Order {order_id} not found."
        elif email:
            row = conn.execute("SELECT id AS customer_id FROM customers WHERE lower(email) = lower(?)",
                               (email.strip(),)).fetchone()
            if row is None:
                return f"No customer with email {email}."
        else:
            return "Need an order_id or customer email to open a ticket."
        cur = conn.execute(
            "INSERT INTO support_tickets (customer_id, order_id, category, description, status, created_date)"
            " VALUES (?, ?, ?, ?, 'open', ?)",
            (row["customer_id"], order_id, category, description, date.today().isoformat()),
        )
    return f"Ticket #{cur.lastrowid} opened ({category}). First response within 24 hours."


@tool
def list_customer_tickets(email: str) -> list[dict] | str:
    """List support tickets (id, order_id, category, status, created_date, description) for a customer email."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT t.id, t.order_id, t.category, t.status, t.created_date, t.description FROM support_tickets t"
            " JOIN customers c ON c.id = t.customer_id WHERE lower(c.email) = lower(?) ORDER BY t.created_date DESC",
            (email.strip(),),
        ).fetchall()
    return [dict(r) for r in rows] or f"No tickets found for {email}."


# ---------- product tools ----------

@tool
def search_products(query: str) -> list[dict] | str:
    """Search the catalog by keyword in product name, category or brand.
    Returns id, name, category, brand, price, stock, warranty_months."""
    words = query.lower().split()
    if not words:
        return "Empty query."
    # ponytail: AND of LIKE per word; switch to SQLite FTS5 if the catalog grows
    clause = " AND ".join("(lower(name) LIKE ? OR lower(category) LIKE ? OR lower(brand) LIKE ?)" for _ in words)
    params = [f"%{w}%" for w in words for _ in range(3)]
    with get_conn() as conn:
        rows = conn.execute(f"SELECT * FROM products WHERE {clause}", params).fetchall()
    return [dict(r) for r in rows] or f"No products match '{query}'."


@tool
def check_stock(product_id: int) -> str:
    """Check how many units of a product (by numeric ID) are in stock."""
    with get_conn() as conn:
        row = conn.execute("SELECT name, stock FROM products WHERE id = ?", (product_id,)).fetchone()
    if row is None:
        return f"Product {product_id} not found."
    return f"{row['name']}: {row['stock']} in stock." if row["stock"] else f"{row['name']}: out of stock."


@tool
def get_product_reviews(product_id: int) -> dict | str:
    """Get average rating, review count and recent review comments for a product (by numeric ID)."""
    with get_conn() as conn:
        rows = conn.execute("SELECT rating, comment, created_date FROM reviews WHERE product_id = ?"
                            " ORDER BY created_date DESC", (product_id,)).fetchall()
    if not rows:
        return f"No reviews for product {product_id}."
    return {"average_rating": round(sum(r["rating"] for r in rows) / len(rows), 2), "count": len(rows),
            "recent": [dict(r) for r in rows[:5]]}


# ---------- policy tool (RAG) ----------

POLICY_K = 3  # chunks returned per search; evals/phase2.py experiments with other values


def policy_chunks(chunking: str = "section") -> list[Document]:
    """Split policies.md into retrievable chunks, each tagged with its section for retrieval evals.
    "section": one chunk per "# " heading, heading included (what the agent uses).
    "sentence": one chunk per sentence, no heading.  "sentence+title": one per sentence, heading prepended."""
    text = (Path(__file__).parent / "policies.md").read_text(encoding="utf-8")
    docs = []
    for block in (b.strip() for b in text.split("# ") if b.strip()):
        title, _, body = block.partition("\n")
        title, body = title.strip(), body.strip()
        if chunking == "section":
            parts = [f"{title}\n{body}"]
        else:
            parts = re.split(r"(?<=\.)\s+", body)
            if chunking == "sentence+title":
                parts = [f"{title}: {p}" for p in parts]
        docs += [Document(page_content=p, metadata={"section": title}) for p in parts if p]
    return docs


def build_policy_store(chunking: str = "section", embeddings=None) -> InMemoryVectorStore:
    embeddings = embeddings or OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    return InMemoryVectorStore.from_documents(policy_chunks(chunking), embeddings)


@lru_cache(maxsize=1)
def _policy_store() -> InMemoryVectorStore:
    return build_policy_store()


def format_policy_hits(hits: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['section']}]\n{d.page_content}" for d in hits)


@tool
def search_policy(query: str) -> str:
    """Search store policy documents: returns, refunds, exchanges, cancellation, shipping, late or damaged deliveries,
    warranty, membership tiers, price match, payment, stock, support tickets, privacy."""
    return format_policy_hits(_policy_store().similarity_search(query, k=POLICY_K))


ORDER_TOOLS = [get_order, list_customer_orders, cancel_order, issue_refund,
               get_customer, create_support_ticket, list_customer_tickets]
PRODUCT_TOOLS = [search_products, check_stock, get_product_reviews]
POLICY_TOOLS = [search_policy]
