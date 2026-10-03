"""SQLite store for the demo shop. init_db() rebuilds from a fixed seed so every run starts identical."""
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "shop.db"

SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL);
CREATE TABLE products (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
    price REAL NOT NULL, stock INTEGER NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
    status TEXT NOT NULL CHECK (status IN ('pending', 'shipped', 'delivered', 'cancelled')),
    order_date TEXT NOT NULL, delivered_date TEXT
);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(id), product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL, unit_price REAL NOT NULL
);
CREATE TABLE refunds (
    id INTEGER PRIMARY KEY, order_id INTEGER UNIQUE NOT NULL REFERENCES orders(id),
    amount REAL NOT NULL, reason TEXT NOT NULL, created_date TEXT NOT NULL
);
"""

CUSTOMERS = [
    (1, "Alice Shah", "alice@example.com"),
    (2, "Bob Mehta", "bob@example.com"),
    (3, "Chen Li", "chen@example.com"),
    (4, "Diana Rossi", "diana@example.com"),
    (5, "Ethan Brown", "ethan@example.com"),
    (6, "Fatima Khan", "fatima@example.com"),
    (7, "George Patel", "george@example.com"),
    (8, "Hana Sato", "hana@example.com"),
]

PRODUCTS = [
    (101, "Wireless Headphones Pro", "audio", 129.99, 12),
    (102, "Bluetooth Speaker Mini", "audio", 39.99, 0),
    (103, "Noise Cancelling Earbuds", "audio", 89.99, 25),
    (104, "Mechanical Keyboard", "computer", 79.99, 8),
    (105, "Wireless Mouse", "computer", 24.99, 40),
    (106, "27-inch 4K Monitor", "computer", 349.99, 3),
    (107, "USB-C Hub 7-in-1", "computer", 34.99, 0),
    (108, "Smartwatch Fit", "wearables", 199.99, 15),
    (109, "Fitness Band Lite", "wearables", 49.99, 30),
    (110, "Phone Case Clear", "phone", 14.99, 100),
    (111, "Fast Charger 30W", "phone", 29.99, 60),
    (112, "Power Bank 20000mAh", "phone", 44.99, 5),
    (113, "Webcam HD 1080p", "computer", 59.99, 9),
    (114, "Gaming Headset RGB", "audio", 69.99, 2),
    (115, "Laptop Stand Aluminium", "computer", 39.99, 18),
]

# (order_id, customer_id, status, order_days_ago, delivered_days_ago | None, [(product_id, qty)])
ORDERS = [
    (1001, 1, "delivered", 60, 55, [(101, 1)]),           # outside 30-day refund window
    (1002, 1, "delivered", 12, 8, [(105, 2)]),            # refundable
    (1003, 2, "shipped", 3, None, [(104, 1)]),
    (1004, 2, "delivered", 20, 15, [(108, 1)]),           # already refunded (see REFUNDS)
    (1005, 3, "pending", 1, None, [(106, 1)]),            # cancellable
    (1006, 3, "cancelled", 10, None, [(102, 1)]),
    (1007, 4, "pending", 0, None, [(110, 3), (111, 1)]),  # cancellable, multi-item
    (1008, 4, "delivered", 40, 31, [(109, 1)]),           # just outside window
    (1009, 5, "delivered", 35, 30, [(113, 1)]),           # exactly 30 days, refundable
    (1010, 5, "shipped", 5, None, [(112, 2)]),
    (1011, 6, "delivered", 7, 2, [(103, 1), (105, 1)]),   # refundable, multi-item
    (1012, 6, "pending", 2, None, [(114, 1)]),
    (1013, 7, "delivered", 90, 85, [(115, 1)]),
    (1014, 7, "shipped", 4, None, [(101, 1), (111, 2)]),
    (1015, 8, "delivered", 14, 10, [(104, 1)]),
    (1016, 8, "cancelled", 6, None, [(107, 1)]),
    (1017, 1, "pending", 0, None, [(112, 1)]),
    (1018, 2, "delivered", 25, 21, [(110, 2)]),
    (1019, 3, "shipped", 2, None, [(109, 2)]),
    (1020, 4, "delivered", 9, 5, [(106, 1)]),
]

REFUNDS = [(1, 1004, 199.99, "Defective strap", 14)]  # (id, order_id, amount, reason, days_ago)


@contextmanager
def get_conn():
    """Yield a connection; commit on success, roll back on error, always close."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Drop and recreate the DB. Dates are relative to today so refund-window cases stay valid."""
    DB_PATH.unlink(missing_ok=True)
    today = date.today()
    ago = lambda d: None if d is None else (today - timedelta(days=d)).isoformat()
    prices = {p[0]: p[3] for p in PRODUCTS}
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?)", CUSTOMERS)
        conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", PRODUCTS)
        for oid, cid, status, od, dd, items in ORDERS:
            conn.execute("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", (oid, cid, status, ago(od), ago(dd)))
            conn.executemany(
                "INSERT INTO order_items VALUES (?, ?, ?, ?)",
                [(oid, pid, qty, prices[pid]) for pid, qty in items],
            )
        conn.executemany(
            "INSERT INTO refunds VALUES (?, ?, ?, ?, ?)",
            [(rid, oid, amt, reason, ago(d)) for rid, oid, amt, reason, d in REFUNDS],
        )
