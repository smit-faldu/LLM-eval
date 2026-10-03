"""Agent tools. Business rules live here, not in prompts, so evals can check the agent can't bypass them."""
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from shop.db import get_conn

REFUND_WINDOW_DAYS = 30


def _order_dict(conn, order_id: int) -> dict | None:
    order = conn.execute(
        "SELECT o.*, c.name AS customer, c.email FROM orders o JOIN customers c ON c.id = o.customer_id"
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
    result["total"] = round(sum(i["quantity"] * i["unit_price"] for i in items), 2)
    result["refund"] = dict(refund) if refund else None
    return result


# ---------- order tools ----------

@tool
def get_order(order_id: int) -> dict | str:
    """Look up one order by its numeric ID: status, dates, items, total and any refund."""
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
        if order["status"] != "delivered":
            return f"Cannot refund order {order_id}: status is '{order['status']}', only delivered orders can be refunded."
        if order["refund"]:
            return f"Cannot refund order {order_id}: already refunded on {order['refund']['created_date']}."
        cutoff = date.today() - timedelta(days=REFUND_WINDOW_DAYS)
        if date.fromisoformat(order["delivered_date"]) < cutoff:
            return (f"Cannot refund order {order_id}: delivered {order['delivered_date']},"
                    f" outside the {REFUND_WINDOW_DAYS}-day refund window.")
        conn.execute(
            "INSERT INTO refunds (order_id, amount, reason, created_date) VALUES (?, ?, ?, ?)",
            (order_id, order["total"], reason, date.today().isoformat()),
        )
    return f"Refund of ${order['total']:.2f} issued for order {order_id}. Arrives in 5-7 business days."


# ---------- product tools ----------

@tool
def search_products(query: str) -> list[dict] | str:
    """Search the catalog by keyword in product name or category. Returns id, name, category, price, stock."""
    words = query.lower().split()
    if not words:
        return "Empty query."
    # ponytail: AND of LIKE per word; switch to SQLite FTS5 if the catalog grows
    clause = " AND ".join("(lower(name) LIKE ? OR lower(category) LIKE ?)" for _ in words)
    params = [p for w in words for p in (f"%{w}%", f"%{w}%")]
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


# ---------- policy tool (RAG) ----------

@lru_cache(maxsize=1)
def _policy_store() -> InMemoryVectorStore:
    text = (Path(__file__).parent / "policies.md").read_text(encoding="utf-8")
    docs = [
        Document(page_content=section.strip(), metadata={"section": section.splitlines()[0].strip()})
        for section in text.split("# ")
        if section.strip()
    ]
    return InMemoryVectorStore.from_documents(docs, GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"))


@tool
def search_policy(query: str) -> str:
    """Search the store policy documents (returns, refunds, cancellation, shipping, warranty, payment, stock)."""
    hits = _policy_store().similarity_search(query, k=3)
    return "\n\n".join(f"[{d.metadata['section']}]\n{d.page_content}" for d in hits)


ORDER_TOOLS = [get_order, list_customer_orders, cancel_order, issue_refund]
PRODUCT_TOOLS = [search_products, check_stock]
POLICY_TOOLS = [search_policy]
