"""
verify_db.py
============
Verification script for recommender.db
Checks schema integrity, row counts, data quality, and sample rows.

Run:
    python verify_db.py
"""

import sqlite3
import json
from pathlib import Path
import sys

# Ensure Unicode output works on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = Path(__file__).parent / "recommender.db"

PASS = "PASS"
FAIL = "FAIL"
INFO = "INFO"


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check(condition: bool, msg: str) -> bool:
    status = PASS if condition else FAIL
    print(f"  {status}  {msg}")
    return condition


def main() -> None:
    print("=" * 60)
    print("  recommender.db — Verification Report")
    print("=" * 60)

    # ── File existence ─────────────────────────────────────────────
    section("1. File Check")
    if not check(DB_PATH.exists(), f"Database file exists at {DB_PATH}"):
        print("\n  Run  python generate_db.py  first, then re-run this script.")
        return

    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"  {INFO}  File size: {size_mb:.2f} MB")

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # ── Schema check ───────────────────────────────────────────────
    section("2. Schema Integrity")

    tables = {
        row["name"]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    check("items" in tables, "Table 'items' exists")
    check("clickstream" in tables, "Table 'clickstream' exists")

    indexes = {
        row["name"]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    for expected_idx in [
        "idx_clickstream_user",
        "idx_clickstream_item",
        "idx_clickstream_session",
        "idx_clickstream_ts",
        "idx_items_genre",
    ]:
        check(expected_idx in indexes, f"Index '{expected_idx}' exists")

    # ── Row counts ─────────────────────────────────────────────────
    section("3. Row Counts")

    (n_items,) = cur.execute("SELECT COUNT(*) FROM items").fetchone()
    (n_events,) = cur.execute("SELECT COUNT(*) FROM clickstream").fetchone()
    (n_users,) = cur.execute(
        "SELECT COUNT(DISTINCT user_id) FROM clickstream"
    ).fetchone()
    (n_sessions,) = cur.execute(
        "SELECT COUNT(DISTINCT session_id) FROM clickstream"
    ).fetchone()

    check(n_items == 1000, f"items table has 1,000 rows  (found {n_items:,})")
    check(n_events > 10_000, f"clickstream has >10,000 events  (found {n_events:,})")
    check(n_users >= 100, f"clickstream has ≥100 distinct users  (found {n_users:,})")
    print(f"  {INFO}  Distinct sessions : {n_sessions:,}")
    print(f"  {INFO}  Events per user   : ~{n_events // max(n_users, 1):,}")

    # ── Data quality — items ───────────────────────────────────────
    section("4. Items — Data Quality")

    (dup_titles,) = cur.execute(
        "SELECT COUNT(*) FROM (SELECT title FROM items GROUP BY title HAVING COUNT(*) > 1)"
    ).fetchone()
    check(dup_titles == 0, f"No duplicate titles  (duplicates found: {dup_titles})")

    (bad_rating,) = cur.execute(
        "SELECT COUNT(*) FROM items WHERE rating < 1.0 OR rating > 5.0"
    ).fetchone()
    check(bad_rating == 0, f"All ratings in [1.0, 5.0]  (violations: {bad_rating})")

    (bad_price,) = cur.execute(
        "SELECT COUNT(*) FROM items WHERE price < 0"
    ).fetchone()
    check(bad_price == 0, f"No negative prices  (violations: {bad_price})")

    (null_tags,) = cur.execute(
        "SELECT COUNT(*) FROM items WHERE tags IS NULL OR tags = ''"
    ).fetchone()
    check(null_tags == 0, f"No null/empty tags  (violations: {null_tags})")

    # Validate that all tags columns parse as JSON arrays
    bad_json = 0
    for (tags_str,) in cur.execute("SELECT tags FROM items").fetchall():
        try:
            parsed = json.loads(tags_str)
            if not isinstance(parsed, list):
                bad_json += 1
        except json.JSONDecodeError:
            bad_json += 1
    check(bad_json == 0, f"All tags columns are valid JSON arrays  (errors: {bad_json})")

    genre_counts = cur.execute(
        "SELECT genre, COUNT(*) AS cnt FROM items GROUP BY genre ORDER BY cnt DESC"
    ).fetchall()
    print(f"  {INFO}  Genre distribution ({len(genre_counts)} genres):")
    for row in genre_counts:
        bar = "█" * (row['cnt'] // 5)
        print(f"         {row['genre']:<18} {row['cnt']:>4}  {bar}")

    # ── Data quality — clickstream ─────────────────────────────────
    section("5. Clickstream — Data Quality")

    (bad_click,) = cur.execute(
        "SELECT COUNT(*) FROM clickstream WHERE clicked NOT IN (0, 1)"
    ).fetchone()
    check(bad_click == 0, f"All clicked values are 0 or 1  (violations: {bad_click})")

    (orphan_items,) = cur.execute(
        """
        SELECT COUNT(*) FROM clickstream cs
        LEFT JOIN items i ON cs.item_id = i.item_id
        WHERE i.item_id IS NULL
        """
    ).fetchone()
    check(orphan_items == 0, f"No orphaned item references  (violations: {orphan_items})")

    click_rate = cur.execute(
        "SELECT ROUND(AVG(clicked) * 100, 2) FROM clickstream"
    ).fetchone()[0]
    print(f"  {INFO}  Overall click-through rate: {click_rate}%")

    top_items = cur.execute(
        """
        SELECT i.title, i.genre, COUNT(*) AS views, SUM(cs.clicked) AS clicks
        FROM clickstream cs
        JOIN items i ON cs.item_id = i.item_id
        GROUP BY cs.item_id
        ORDER BY clicks DESC
        LIMIT 5
        """
    ).fetchall()
    print(f"\n  {INFO}  Top 5 most-clicked items:")
    for r in top_items:
        print(f"         [{r['genre']:<12}]  {r['title']:<40}  "
              f"views={r['views']:>5}  clicks={r['clicks']:>4}")

    # ── Temporal sanity ────────────────────────────────────────────
    section("6. Temporal Sanity")

    ts_row = cur.execute(
        "SELECT MIN(timestamp) AS mn, MAX(timestamp) AS mx FROM clickstream"
    ).fetchone()
    print(f"  {INFO}  Earliest event : {ts_row['mn']}")
    print(f"  {INFO}  Latest event   : {ts_row['mx']}")

    (bad_ts,) = cur.execute(
        "SELECT COUNT(*) FROM clickstream WHERE timestamp > '2026-12-31'"
    ).fetchone()
    check(bad_ts == 0, f"No future timestamps beyond 2026  (violations: {bad_ts})")

    # ── Sample rows ────────────────────────────────────────────────
    section("7. Sample Rows")

    print(f"\n  {INFO}  5 random items:")
    rows = cur.execute(
        "SELECT item_id, title, genre, price, rating FROM items ORDER BY RANDOM() LIMIT 5"
    ).fetchall()
    print(f"  {'ID':>5}  {'Title':<40}  {'Genre':<14}  {'Price':>7}  {'Rating':>6}")
    print(f"  {'─'*5}  {'─'*40}  {'─'*14}  {'─'*7}  {'─'*6}")
    for r in rows:
        print(f"  {r['item_id']:>5}  {r['title']:<40}  {r['genre']:<14}  "
              f"${r['price']:>6.2f}  {r['rating']:>6.1f}")

    print(f"\n  {INFO}  5 random clickstream events:")
    evts = cur.execute(
        """
        SELECT cs.event_id, cs.user_id, cs.item_id, i.title,
               cs.timestamp, cs.clicked
        FROM clickstream cs
        JOIN items i ON cs.item_id = i.item_id
        ORDER BY RANDOM()
        LIMIT 5
        """
    ).fetchall()
    print(f"  {'EventID':>7}  {'UserID':>6}  {'ItemID':>6}  {'Title':<30}  "
          f"{'Timestamp':<20}  Clicked")
    print(f"  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*30}  {'─'*20}  {'─'*7}")
    for e in evts:
        print(f"  {e['event_id']:>7}  {e['user_id']:>6}  {e['item_id']:>6}  "
              f"{e['title']:<30}  {e['timestamp']:<20}  {'Yes' if e['clicked'] else 'No ':>7}")

    con.close()

    section("Summary")
    print(f"  {INFO}  Items        : {n_items:,}")
    print(f"  {INFO}  Users        : {n_users:,}")
    print(f"  {INFO}  Sessions     : {n_sessions:,}")
    print(f"  {INFO}  Events       : {n_events:,}")
    print(f"  {INFO}  DB size      : {size_mb:.2f} MB")
    print(f"\n  Verification complete — database looks healthy!")
    print("=" * 60)


if __name__ == "__main__":
    main()
