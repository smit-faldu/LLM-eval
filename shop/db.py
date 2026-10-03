"""SQLite store for the demo shop. init_db() rebuilds from a fixed seed so every run starts identical.

Customers 1-8 and orders 1001-1020 are hand-written edge cases (refund window, double refund, cancellable...)
that tests and evals rely on. The rest is generated from a seeded RNG for realistic volume.
"""
import random
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "shop.db"

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, phone TEXT NOT NULL,
    city TEXT NOT NULL, tier TEXT NOT NULL CHECK (tier IN ('regular', 'silver', 'gold')), joined_date TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL, brand TEXT NOT NULL,
    price REAL NOT NULL, stock INTEGER NOT NULL, warranty_months INTEGER NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
    status TEXT NOT NULL CHECK (status IN ('pending', 'shipped', 'delivered', 'cancelled')),
    order_date TEXT NOT NULL, delivered_date TEXT,
    shipping_method TEXT NOT NULL CHECK (shipping_method IN ('standard', 'express')),
    shipping_fee REAL NOT NULL, tracking_number TEXT
);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(id), product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL, unit_price REAL NOT NULL
);
CREATE TABLE refunds (
    id INTEGER PRIMARY KEY, order_id INTEGER UNIQUE NOT NULL REFERENCES orders(id),
    amount REAL NOT NULL, reason TEXT NOT NULL, created_date TEXT NOT NULL
);
CREATE TABLE support_tickets (
    id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id), order_id INTEGER REFERENCES orders(id),
    category TEXT NOT NULL CHECK (category IN ('warranty', 'damaged', 'shipping', 'billing', 'other')),
    description TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('open', 'in_progress', 'resolved')),
    created_date TEXT NOT NULL
);
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY, product_id INTEGER NOT NULL REFERENCES products(id),
    customer_id INTEGER NOT NULL REFERENCES customers(id), rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment TEXT NOT NULL, created_date TEXT NOT NULL
);
"""

# (id, name, email, phone, city, tier, joined_days_ago)
CUSTOMERS = [
    (1, "Alice Shah", "alice@example.com", "+91-98200-11001", "Mumbai", "gold", 900),
    (2, "Bob Mehta", "bob@example.com", "+91-98200-11002", "Pune", "regular", 400),
    (3, "Chen Li", "chen@example.com", "+91-98200-11003", "Bengaluru", "silver", 650),
    (4, "Diana Rossi", "diana@example.com", "+91-98200-11004", "Delhi", "regular", 200),
    (5, "Ethan Brown", "ethan@example.com", "+91-98200-11005", "Hyderabad", "gold", 1200),
    (6, "Fatima Khan", "fatima@example.com", "+91-98200-11006", "Chennai", "silver", 300),
    (7, "George Patel", "george@example.com", "+91-98200-11007", "Ahmedabad", "regular", 150),
    (8, "Hana Sato", "hana@example.com", "+91-98200-11008", "Kolkata", "regular", 60),
]

# (id, name, category, brand, price, stock, warranty_months)
PRODUCTS = [
    (101, "Wireless Headphones Pro", "audio", "Sonic", 129.99, 12, 12),
    (102, "Bluetooth Speaker Mini", "audio", "Sonic", 39.99, 0, 12),
    (103, "Noise Cancelling Earbuds", "audio", "Aura", 89.99, 25, 12),
    (104, "Mechanical Keyboard", "computer", "KeyCraft", 79.99, 8, 24),
    (105, "Wireless Mouse", "computer", "KeyCraft", 24.99, 40, 12),
    (106, "27-inch 4K Monitor", "computer", "Visio", 349.99, 3, 36),
    (107, "USB-C Hub 7-in-1", "computer", "Linkr", 34.99, 0, 12),
    (108, "Smartwatch Fit", "wearables", "Pulse", 199.99, 15, 12),
    (109, "Fitness Band Lite", "wearables", "Pulse", 49.99, 30, 12),
    (110, "Phone Case Clear", "phone", "Shieldo", 14.99, 100, 6),
    (111, "Fast Charger 30W", "phone", "Voltix", 29.99, 60, 12),
    (112, "Power Bank 20000mAh", "phone", "Voltix", 44.99, 5, 12),
    (113, "Webcam HD 1080p", "computer", "Visio", 59.99, 9, 12),
    (114, "Gaming Headset RGB", "audio", "Aura", 69.99, 2, 12),
    (115, "Laptop Stand Aluminium", "computer", "Linkr", 39.99, 18, 6),
    (116, "Soundbar 2.1 Home", "audio", "Sonic", 249.99, 7, 24),
    (117, "Studio Monitor Speakers", "audio", "Aura", 179.99, 4, 24),
    (118, "Portable Party Speaker", "audio", "Sonic", 119.99, 0, 12),
    (119, "Over-Ear Kids Headphones", "audio", "Aura", 29.99, 50, 6),
    (120, "Ergonomic Vertical Mouse", "computer", "KeyCraft", 39.99, 22, 12),
    (121, "Wireless Keyboard Slim", "computer", "KeyCraft", 49.99, 14, 12),
    (122, "34-inch Ultrawide Monitor", "computer", "Visio", 499.99, 2, 36),
    (123, "1TB Portable SSD", "storage", "Datavault", 99.99, 35, 36),
    (124, "2TB External Hard Drive", "storage", "Datavault", 74.99, 20, 24),
    (125, "128GB USB Flash Drive", "storage", "Datavault", 15.99, 120, 12),
    (126, "256GB microSD Card", "storage", "Datavault", 27.99, 80, 12),
    (127, "Wi-Fi 6 Router", "networking", "Linkr", 129.99, 10, 24),
    (128, "Mesh Wi-Fi System 3-Pack", "networking", "Linkr", 279.99, 0, 24),
    (129, "Gigabit Ethernet Switch 8-Port", "networking", "Linkr", 39.99, 25, 24),
    (130, "Smartwatch Ultra", "wearables", "Pulse", 399.99, 6, 24),
    (131, "Kids GPS Watch", "wearables", "Pulse", 79.99, 11, 12),
    (132, "Smart Ring", "wearables", "Aura", 249.99, 0, 12),
    (133, "Wireless Charging Pad", "phone", "Voltix", 24.99, 45, 12),
    (134, "Car Phone Mount", "phone", "Shieldo", 19.99, 70, 6),
    (135, "Tempered Glass Screen Protector", "phone", "Shieldo", 9.99, 200, 0),
    (136, "65W GaN Laptop Charger", "phone", "Voltix", 49.99, 28, 18),
    (137, "Smart Bulb Color 4-Pack", "smart-home", "Lumo", 54.99, 33, 12),
    (138, "Smart Plug 2-Pack", "smart-home", "Lumo", 24.99, 60, 12),
    (139, "Video Doorbell", "smart-home", "Lumo", 149.99, 8, 24),
    (140, "Indoor Security Camera", "smart-home", "Lumo", 59.99, 16, 12),
    (141, "Robot Vacuum Basic", "smart-home", "Cleanix", 229.99, 3, 24),
    (142, "Smart Thermostat", "smart-home", "Lumo", 169.99, 5, 24),
    (143, "Game Controller Wireless", "gaming", "Arcadia", 59.99, 19, 12),
    (144, "Gaming Mouse 16K DPI", "gaming", "Arcadia", 54.99, 13, 24),
    (145, "Gaming Chair Pro", "gaming", "Arcadia", 289.99, 1, 24),
    (146, "Streaming Microphone USB", "gaming", "Aura", 99.99, 9, 12),
    (147, "E-Reader 6-inch", "tablets", "Pagewise", 129.99, 12, 12),
    (148, "10-inch Android Tablet", "tablets", "Pagewise", 219.99, 7, 12),
    (149, "Stylus Pen Universal", "tablets", "Pagewise", 29.99, 40, 6),
    (150, "Action Camera 4K", "cameras", "Visio", 199.99, 6, 12),
    (151, "Camera Tripod Travel", "cameras", "Visio", 44.99, 21, 12),
    (152, "Ring Light 12-inch", "cameras", "Lumo", 34.99, 30, 6),
]

# (order_id, customer_id, status, order_days_ago, delivered_days_ago | None, [(product_id, qty)])
EDGE_ORDERS = [
    (1001, 1, "delivered", 60, 55, [(101, 1)]),           # outside 30-day refund window
    (1002, 1, "delivered", 12, 8, [(105, 2)]),            # refundable
    (1003, 2, "shipped", 3, None, [(104, 1)]),
    (1004, 2, "delivered", 20, 15, [(108, 1)]),           # already refunded (see EDGE_REFUNDS)
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
EDGE_EXPRESS = {1003, 1005, 1014}  # these ship express; everything else hand-written is standard

EDGE_REFUNDS = [(1004, 199.99, "Defective strap", 14)]  # (order_id, amount, reason, days_ago)

# (customer_id, order_id | None, category, description, status, days_ago)
EDGE_TICKETS = [
    (7, 1013, "warranty", "Laptop stand hinge cracked after normal use.", "in_progress", 5),
    (2, 1003, "shipping", "Tracking has not updated for two days.", "open", 1),
    (5, None, "billing", "Charged twice for my last order, please check.", "resolved", 40),
]

FIRST = ["Aarav", "Isha", "Kabir", "Meera", "Rohan", "Sara", "Vikram", "Zoya", "Liam", "Olivia", "Noah", "Emma",
         "Arjun", "Priya", "Daniel", "Sofia", "Karan", "Nisha", "Lucas", "Mia", "Dev", "Ananya", "Ryan", "Tara",
         "Omar", "Leela", "Nikhil", "Ritu", "Samuel", "Kavya", "Aditya", "Neha"]
LAST = ["Iyer", "Gupta", "Singh", "Nair", "Kapoor", "Das", "Joshi", "Reddy", "Fernandes", "Malhotra", "Bose",
        "Verma", "Chopra", "Pillai", "Rao", "Sen", "Thomas", "Kulkarni", "Bhatt", "Menon", "Saxena", "Dutta",
        "Ghosh", "Agarwal", "Mishra", "Banerjee", "Pandey", "Shetty", "Jain", "Kaur", "Arora", "Sinha"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Kochi"]
REFUND_REASONS = ["Item not as described", "Changed my mind", "Stopped working", "Wrong size", "Found it cheaper"]
TICKET_TEXT = {
    "warranty": "Device stopped charging after a few months.",
    "damaged": "Package arrived with a cracked screen.",
    "shipping": "Delivery is late and courier is not responding.",
    "billing": "Discount code was not applied at checkout.",
    "other": "Need an invoice with my company GST number.",
}
REVIEW_TEXT = {
    5: ["Excellent, exactly as described.", "Love it, works perfectly.", "Best purchase this year."],
    4: ["Good value for money.", "Works well, packaging could be better.", "Solid product, minor quirks."],
    3: ["Average, does the job.", "Okay but battery life is short.", "Decent, expected more."],
    2: ["Build quality feels cheap.", "Stopped working properly after weeks."],
    1: ["Arrived broken.", "Does not work at all, very disappointed."],
}


def _shipping_fee(method: str, subtotal: float, tier: str) -> float:
    """Mirrors the shipping policy in policies.md."""
    if method == "express":
        return 0.0 if tier == "gold" else 12.99
    return 0.0 if subtotal >= 50 or tier in ("silver", "gold") else 4.99


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


def _seed_rows():
    """Build every table's rows. Dates are days-ago offsets, resolved against today by init_db."""
    rng = random.Random(42)
    customers = list(CUSTOMERS)
    for i, first in enumerate(FIRST):
        cid, last = 9 + i, LAST[(i * 7) % len(LAST)]
        tier = rng.choices(["regular", "silver", "gold"], [6, 3, 1])[0]
        customers.append((cid, f"{first} {last}", f"{first}.{last}@example.com".lower(),
                          f"+91-9{rng.randint(100000000, 999999999)}", rng.choice(CITIES), tier, rng.randint(10, 1500)))
    tiers = {c[0]: c[5] for c in customers}
    prices = {p[0]: p[4] for p in PRODUCTS}
    in_stock = [p[0] for p in PRODUCTS if p[5] > 0]

    orders = [(oid, cid, st, od, dd, items, "express" if oid in EDGE_EXPRESS else "standard")
              for oid, cid, st, od, dd, items in EDGE_ORDERS]
    for oid in range(1021, 1141):
        status = rng.choices(["delivered", "shipped", "pending", "cancelled"], [65, 15, 10, 10])[0]
        od = {"pending": rng.randint(0, 2), "shipped": rng.randint(1, 6)}.get(status, rng.randint(7, 150))
        dd = od - rng.randint(2, 6) if status == "delivered" else None
        items = [(pid, rng.choice([1, 1, 1, 2])) for pid in rng.sample(in_stock, rng.choice([1, 1, 2, 3]))]
        method = rng.choices(["standard", "express"], [8, 2])[0]
        orders.append((oid, rng.randint(9, 8 + len(FIRST)), status, od, dd, items, method))

    order_rows, item_rows = [], []
    for oid, cid, status, od, dd, items, method in orders:
        subtotal = sum(prices[pid] * qty for pid, qty in items)
        tracking = f"TRK{oid}{cid:03d}IN" if status in ("shipped", "delivered") else None
        order_rows.append((oid, cid, status, od, dd, method, _shipping_fee(method, subtotal, tiers[cid]), tracking))
        item_rows += [(oid, pid, qty, prices[pid]) for pid, qty in items]

    generated = [o for o in orders if o[0] > 1020]
    delivered = [o for o in generated if o[2] == "delivered"]
    refunds = list(EDGE_REFUNDS)
    for oid, _, _, _, dd, items, _ in rng.sample(delivered, 12):
        amount = round(sum(prices[pid] * qty for pid, qty in items), 2)
        refunds.append((oid, amount, rng.choice(REFUND_REASONS), max(dd - rng.randint(1, 10), 0)))

    tickets = list(EDGE_TICKETS)
    for oid, cid, _, _, dd, _, _ in rng.sample(delivered, 18):
        category = rng.choice(list(TICKET_TEXT))
        status = rng.choices(["open", "in_progress", "resolved"], [3, 2, 5])[0]
        tickets.append((cid, oid, category, TICKET_TEXT[category], status, max(dd - rng.randint(0, 20), 0)))

    reviews = []
    for oid, cid, _, _, dd, items, _ in rng.sample(delivered, 50):
        pid = items[0][0]
        rating = rng.choices([5, 4, 3, 2, 1], [40, 30, 15, 10, 5])[0]
        reviews.append((pid, cid, rating, rng.choice(REVIEW_TEXT[rating]), max(dd - rng.randint(1, 15), 0)))

    return customers, order_rows, item_rows, refunds, tickets, reviews


def init_db() -> None:
    """Drop and recreate the DB. Dates are relative to today so refund-window cases stay valid."""
    DB_PATH.unlink(missing_ok=True)
    today = date.today()
    ago = lambda d: None if d is None else (today - timedelta(days=d)).isoformat()
    customers, orders, items, refunds, tickets, reviews = _seed_rows()
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)",
                         [(*c[:6], ago(c[6])) for c in customers])
        conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", PRODUCTS)
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         [(o[0], o[1], o[2], ago(o[3]), ago(o[4]), *o[5:]) for o in orders])
        conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?)", items)
        conn.executemany("INSERT INTO refunds (order_id, amount, reason, created_date) VALUES (?, ?, ?, ?)",
                         [(*r[:3], ago(r[3])) for r in refunds])
        conn.executemany("INSERT INTO support_tickets (customer_id, order_id, category, description, status,"
                         " created_date) VALUES (?, ?, ?, ?, ?, ?)", [(*t[:5], ago(t[5])) for t in tickets])
        conn.executemany("INSERT INTO reviews (product_id, customer_id, rating, comment, created_date)"
                         " VALUES (?, ?, ?, ?, ?)", [(*r[:4], ago(r[4])) for r in reviews])


def table_names() -> list[str]:
    with get_conn() as conn:
        return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY rowid")]
