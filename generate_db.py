"""
generate_db.py
==============
Autonomous Dynamic Personalization & Recommendation Engine
----------------------------------------------------------
Generates a synthetic SQLite database (recommender.db) with:
  - 1,000 video-game items (items table)
  - Synthetic user clickstream / session logs (clickstream table)

Run:
    python generate_db.py
"""

import sqlite3
import json
import random
import uuid
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure Unicode output works on all Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Output path ────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "recommender.db"

# ══════════════════════════════════════════════════════════════════════════════
# DATA ASSETS
# ══════════════════════════════════════════════════════════════════════════════

GENRES = [
    "Action", "Adventure", "RPG", "Strategy", "Simulation",
    "Sports", "Racing", "Horror", "Puzzle", "Platformer",
    "Fighting", "Shooter", "MMORPG", "Roguelike", "Sandbox",
    "Stealth", "Survival", "Visual Novel", "Rhythm", "Card Game",
]

TAG_POOL = [
    "open-world", "multiplayer", "co-op", "PvP", "story-rich", "pixel-art",
    "anime", "sci-fi", "fantasy", "post-apocalyptic", "cyberpunk", "western",
    "medieval", "space", "underwater", "steampunk", "horror", "comedy",
    "dark", "casual", "hardcore", "indie", "AAA", "retro", "4K",
    "VR-support", "controller-support", "moddable", "procedural", "permadeath",
    "turn-based", "real-time", "isometric", "first-person", "third-person",
    "2D", "3D", "top-down", "side-scroller", "sandbox", "crafting", "building",
    "farming", "fishing", "dating-sim", "dungeon-crawler", "hack-and-slash",
    "bullet-hell", "metroidvania", "soulslike", "JRPG", "WRPG", "tower-defense",
]

WORD_PARTS_A = [
    "Dark", "Eternal", "Shadow", "Crystal", "Iron", "Lost", "Ancient",
    "Cyber", "Neon", "Storm", "Blood", "Void", "Star", "Moon", "Fire",
    "Frost", "Chaos", "Arcane", "Primal", "Hollow", "Sacred", "Wrath",
    "Fallen", "Broken", "Forgotten", "Dying", "Blazing", "Silent", "Shattered",
    "Rogue", "Nexus", "Omega", "Alpha", "Phantom", "Hyper", "Ultra",
]

WORD_PARTS_B = [
    "World", "Age", "Quest", "Chronicles", "Legacy", "Realm", "Empire",
    "Arena", "Odyssey", "Saga", "Forge", "Rift", "Gate", "Edge", "Strike",
    "Rising", "Dawn", "Dusk", "Fall", "Wars", "Legends", "Protocol",
    "Nexus", "Cycle", "Code", "Breach", "Command", "Bastion", "Vault",
    "Engine", "Abyss", "Horizon", "Haven", "Syndicate", "Uprising",
]

SUFFIXES = [
    "", "", "", "",   # majority have no suffix (weighted)
    "II", "III", "Remastered", "Origins", "Reloaded", "Redux",
    "Zero", "Evolved", "Reborn", "Aftermath", "Reckoning",
]

PUBLISHERS = [
    "Pixel Forge Studios", "Nova Interactive", "Quantum Realm Games",
    "SkyBound Interactive", "DarkHorse Entertainment", "Lunar Arc",
    "Thunder Peak", "Voidwalker Games", "Eclipse Digital", "Ironclad Studios",
    "Meridian Games", "Prism Works", "Starfall Interactive", "Nebula Play",
    "Arcane Soft", "HorizonX", "RedShift Games", "WarpZone Studios",
]


def random_title(used: set) -> str:
    """Generate a unique game title."""
    for _ in range(1000):
        a = random.choice(WORD_PARTS_A)
        b = random.choice(WORD_PARTS_B)
        s = random.choice(SUFFIXES)
        title = f"{a} {b}" if not s else f"{a} {b}: {s}"
        if title not in used:
            used.add(title)
            return title
    # Fallback: append uuid fragment
    title = f"{random.choice(WORD_PARTS_A)} {random.choice(WORD_PARTS_B)} {uuid.uuid4().hex[:4].upper()}"
    used.add(title)
    return title


def build_items(n: int = 1000) -> list[dict]:
    """Build n synthetic game item records."""
    used_titles: set[str] = set()
    items = []

    for i in range(1, n + 1):
        genre = random.choice(GENRES)

        # 2–6 tags, genre-seeded then random
        n_tags = random.randint(2, 6)
        genre_tag = genre.lower().replace(" ", "-")
        tags = [genre_tag]
        pool_copy = [t for t in TAG_POOL if t != genre_tag]
        random.shuffle(pool_copy)
        tags += pool_copy[: n_tags - 1]

        # Price tiers: free / budget / mid / premium / collector
        tier = random.choices(
            ["free", "budget", "mid", "premium", "collector"],
            weights=[5, 20, 40, 30, 5],
        )[0]
        price_map = {
            "free": 0.0,
            "budget": round(random.uniform(0.99, 9.99), 2),
            "mid": round(random.uniform(14.99, 29.99), 2),
            "premium": round(random.uniform(39.99, 59.99), 2),
            "collector": round(random.uniform(69.99, 99.99), 2),
        }

        # Rating biased towards 3.0-4.8 (realistic bell curve)
        rating = round(min(5.0, max(1.0, random.gauss(3.8, 0.7))), 1)

        # Release date: somewhere in the last 15 years
        days_ago = random.randint(0, 365 * 15)
        release_date = (datetime(2026, 5, 19) - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        items.append(
            {
                "item_id": i,
                "title": random_title(used_titles),
                "genre": genre,
                "tags": json.dumps(tags),
                "price": price_map[tier],
                "rating": rating,
                "publisher": random.choice(PUBLISHERS),
                "release_date": release_date,
                "downloads": random.randint(500, 10_000_000),
            }
        )

    return items


# ══════════════════════════════════════════════════════════════════════════════
# CLICKSTREAM GENERATION
# ══════════════════════════════════════════════════════════════════════════════

N_USERS = 500
N_ITEMS = 1000
SESSIONS_PER_USER_MIN = 2
SESSIONS_PER_USER_MAX = 15
EVENTS_PER_SESSION_MIN = 3
EVENTS_PER_SESSION_MAX = 25

# Users have genre affinities (simulates real preference clusters)
USER_GENRE_CLUSTERS = {}


def _assign_user_genres():
    """Assign 1-3 preferred genres per user (hidden affinity)."""
    for uid in range(1, N_USERS + 1):
        n_pref = random.randint(1, 3)
        USER_GENRE_CLUSTERS[uid] = random.sample(GENRES, n_pref)


def _genre_to_item_ids(items: list[dict]) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for item in items:
        mapping.setdefault(item["genre"], []).append(item["item_id"])
    return mapping


def build_clickstream(items: list[dict]) -> list[dict]:
    """
    Generate chronological session-based clickstream events.

    Click probability is higher when:
      - item genre matches user affinity
      - item rating > 4.0
      - item is free/budget
    """
    _assign_user_genres()
    genre_items = _genre_to_item_ids(items)
    item_lookup = {it["item_id"]: it for it in items}

    all_item_ids = [it["item_id"] for it in items]
    events: list[dict] = []

    base_time = datetime(2025, 1, 1)

    for uid in range(1, N_USERS + 1):
        user_genres = USER_GENRE_CLUSTERS[uid]
        # Preferred items for this user
        preferred_items: list[int] = []
        for g in user_genres:
            preferred_items.extend(genre_items.get(g, []))

        n_sessions = random.randint(SESSIONS_PER_USER_MIN, SESSIONS_PER_USER_MAX)
        session_start = base_time + timedelta(days=random.randint(0, 60))

        for _ in range(n_sessions):
            session_id = str(uuid.uuid4())
            n_events = random.randint(EVENTS_PER_SESSION_MIN, EVENTS_PER_SESSION_MAX)

            # Build candidate pool: 70% from preferred, 30% random exploration
            n_preferred = int(n_events * 0.7)
            n_random = n_events - n_preferred

            preferred_pool = (
                random.choices(preferred_items, k=n_preferred) if preferred_items else []
            )
            random_pool = random.choices(all_item_ids, k=n_random)

            session_items = preferred_pool + random_pool
            random.shuffle(session_items)

            event_time = session_start

            for item_id in session_items:
                item = item_lookup[item_id]

                # Click probability heuristic
                click_prob = 0.25  # baseline
                if item["genre"] in user_genres:
                    click_prob += 0.35
                if item["rating"] >= 4.0:
                    click_prob += 0.15
                if item["price"] == 0.0:
                    click_prob += 0.10
                elif item["price"] < 15.0:
                    click_prob += 0.05
                click_prob = min(click_prob, 0.95)

                clicked = 1 if random.random() < click_prob else 0

                events.append(
                    {
                        "user_id": uid,
                        "session_id": session_id,
                        "item_id": item_id,
                        "timestamp": event_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "clicked": clicked,
                    }
                )

                # Advance time within session (5s–3min between events)
                event_time += timedelta(seconds=random.randint(5, 180))

            # Next session starts 1 hour – 30 days later
            session_start += timedelta(hours=random.randint(1, 24 * 30))

    # Sort globally by (user_id, timestamp) for chronological order
    events.sort(key=lambda e: (e["user_id"], e["timestamp"]))
    return events


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE CREATION
# ══════════════════════════════════════════════════════════════════════════════

DDL_ITEMS = """
CREATE TABLE IF NOT EXISTS items (
    item_id      INTEGER PRIMARY KEY,
    title        TEXT    NOT NULL UNIQUE,
    genre        TEXT    NOT NULL,
    tags         TEXT    NOT NULL,   -- JSON array string
    price        REAL    NOT NULL DEFAULT 0.0,
    rating       REAL    NOT NULL,   -- 1.0 – 5.0
    publisher    TEXT    NOT NULL,
    release_date TEXT    NOT NULL,   -- YYYY-MM-DD
    downloads    INTEGER NOT NULL DEFAULT 0
);
"""

DDL_CLICKSTREAM = """
CREATE TABLE IF NOT EXISTS clickstream (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    session_id  TEXT    NOT NULL,
    item_id     INTEGER NOT NULL,
    timestamp   TEXT    NOT NULL,   -- YYYY-MM-DD HH:MM:SS (UTC)
    clicked     INTEGER NOT NULL CHECK(clicked IN (0, 1)),
    FOREIGN KEY (item_id) REFERENCES items(item_id)
);
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_clickstream_user    ON clickstream(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_clickstream_item    ON clickstream(item_id);",
    "CREATE INDEX IF NOT EXISTS idx_clickstream_session ON clickstream(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_clickstream_ts      ON clickstream(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_items_genre         ON items(genre);",
]


def create_database(items: list[dict], events: list[dict]) -> None:
    """Create recommender.db and populate it."""

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[INFO] Removed existing {DB_PATH.name}")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Enable WAL for better write performance
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    # Schema
    cur.execute(DDL_ITEMS)
    cur.execute(DDL_CLICKSTREAM)
    for idx_sql in DDL_INDEXES:
        cur.execute(idx_sql)

    # Items
    print(f"[INFO] Inserting {len(items):,} items …")
    cur.executemany(
        """
        INSERT INTO items
            (item_id, title, genre, tags, price, rating, publisher, release_date, downloads)
        VALUES
            (:item_id, :title, :genre, :tags, :price, :rating, :publisher, :release_date, :downloads)
        """,
        items,
    )

    # Clickstream
    print(f"[INFO] Inserting {len(events):,} clickstream events …")
    cur.executemany(
        """
        INSERT INTO clickstream
            (user_id, session_id, item_id, timestamp, clicked)
        VALUES
            (:user_id, :session_id, :item_id, :timestamp, :clicked)
        """,
        events,
    )

    con.commit()
    con.close()

    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"[OK]   Database written -> {DB_PATH}  ({size_mb:.2f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  Recommendation Engine — Synthetic Data Generator")
    print("=" * 60)

    print("\n[STEP 1/3] Generating 1,000 game items ...")
    items = build_items(1000)
    print(f"           -> {len(items):,} items built")

    print("\n[STEP 2/3] Generating clickstream logs ...")
    events = build_clickstream(items)
    print(f"           -> {len(events):,} events across {N_USERS} users")

    print("\n[STEP 3/3] Writing SQLite database ...")
    create_database(items, events)

    print("\n[DONE] Run  python verify_db.py  to validate the database.")
    print("=" * 60)


if __name__ == "__main__":
    main()
