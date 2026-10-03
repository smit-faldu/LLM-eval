"""Offline check of tool business rules (no LLM calls). Run: uv run python test_smoke.py"""
from shop.db import init_db
from shop.tools import cancel_order, check_stock, get_order, issue_refund, list_customer_orders, search_products

init_db()

assert get_order.invoke({"order_id": 1007})["total"] == 74.96
assert "not found" in get_order.invoke({"order_id": 9999})
assert len(list_customer_orders.invoke({"email": "ALICE@example.com"})) == 3

assert "cancelled" in cancel_order.invoke({"order_id": 1005})
assert "only pending" in cancel_order.invoke({"order_id": 1005})  # second cancel blocked
assert "only pending" in cancel_order.invoke({"order_id": 1003})  # shipped

assert "outside" in issue_refund.invoke({"order_id": 1001, "reason": "x"})
assert "outside" in issue_refund.invoke({"order_id": 1008, "reason": "x"})  # 31 days
assert "Refund of $59.99" in issue_refund.invoke({"order_id": 1009, "reason": "x"})  # exactly 30 days
assert "already refunded" in issue_refund.invoke({"order_id": 1009, "reason": "x"})
assert "already refunded" in issue_refund.invoke({"order_id": 1004, "reason": "x"})
assert "only delivered" in issue_refund.invoke({"order_id": 1010, "reason": "x"})

assert [p["id"] for p in search_products.invoke({"query": "wireless headphones"})] == [101]
assert "out of stock" in check_stock.invoke({"product_id": 102})

init_db()  # leave a clean DB behind
print("all tool rules ok")
