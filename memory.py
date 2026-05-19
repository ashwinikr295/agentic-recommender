"""
memory.py
=========
Hierarchical Belief-State Memory System
----------------------------------------
Three-layer memory architecture for the Autonomous Recommendation Engine:

  Layer 1 – EventBuffer      : rolling cache of raw session events (clicks/queries)
  Layer 2 – PreferenceMemory : structured JSON beliefs keyed by preference signal
                               {"genre": "Action", "strength": 0.8, "evidence_count": 4}
  Layer 3 – ProfileMemory    : natural-language summary of stable user traits

State transitions
-----------------
  extract()            – drain event buffer -> update preference beliefs
  boost(key, step)     – increment strength + evidence (optimistic signal)
  demote(key, step)    – decrement strength faster (pessimistic update rule)
  forget()             – auto-prune beliefs whose strength has fallen below 0.0

Usage
-----
    mem = HierarchicalBeliefMemory(user_id=42, db_path="recommender.db")
    mem.log_click(item_id=315, clicked=True)
    mem.extract()
    print(mem.profile)
    recs = mem.ranked_preferences(top_k=5)
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH: Path = Path(__file__).parent / "recommender.db"

# Belief update parameters
BOOST_DEFAULT: float = 0.1
DEMOTE_DEFAULT: float = 0.2   # faster erosion → pessimistic update rule
STRENGTH_FLOOR: float = 0.0   # below this → forgotten
STRENGTH_CEIL: float = 1.0
CLICK_BOOST: float = 0.15     # applied per click during extract()
SKIP_DEMOTE: float = 0.05     # applied per non-click during extract()

# Event buffer
BUFFER_CAPACITY: int = 200    # maximum raw events kept in rolling cache


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClickEvent:
    """A single raw interaction event stored in the event buffer."""
    item_id: int
    clicked: bool                              # True = positive signal
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    genre: str | None = None                   # populated on log / during extract
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreferenceEntry:
    """
    A single mutable belief about one user preference dimension.

    Fields
    ------
    key            : Unique identifier (e.g. "genre:Action", "tag:open-world")
    genre          : Human-readable genre label
    strength       : Belief confidence in [0.0, 1.0]
    evidence_count : Total number of signals that have touched this belief
    last_updated   : ISO-8601 timestamp of the most recent update
    tags           : Optional tag hints associated with this preference
    """
    key: str
    genre: str
    strength: float = 0.5
    evidence_count: int = 0
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tags: list[str] = field(default_factory=list)

    # ── Serialisation ──────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PreferenceEntry":
        return cls(**d)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — EVENT BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class EventBuffer:
    """
    Fast rolling cache (deque) for raw session events.

    Capacity is capped at BUFFER_CAPACITY; oldest events are evicted
    automatically when the buffer is full (FIFO).
    """

    def __init__(self, capacity: int = BUFFER_CAPACITY) -> None:
        self._buf: deque[ClickEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────

    def push(self, event: ClickEvent) -> None:
        """Append a new event; oldest is silently dropped when at capacity."""
        with self._lock:
            self._buf.append(event)

    def drain(self) -> list[ClickEvent]:
        """
        Remove and return all events currently in the buffer (consume once).
        Callers are responsible for processing the returned events.
        """
        with self._lock:
            events = list(self._buf)
            self._buf.clear()
        return events

    def peek(self, n: int | None = None) -> list[ClickEvent]:
        """Return (at most n) events without consuming them."""
        with self._lock:
            buf = list(self._buf)
        return buf[-n:] if n is not None else buf

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return f"EventBuffer(size={len(self)}/{self._buf.maxlen})"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — PREFERENCE MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class PreferenceMemory:
    """
    Structured JSON belief store keyed by preference signal strings.

    Keys follow the convention  "<dimension>:<value>", e.g.:
        "genre:Action"
        "tag:open-world"

    Beliefs are mutable via boost() / demote() and are auto-pruned
    when strength falls to STRENGTH_FLOOR via forget().
    """

    def __init__(self) -> None:
        self._beliefs: dict[str, PreferenceEntry] = {}
        self._lock = threading.Lock()

    # ── Core state transitions ─────────────────────────────────────────────

    def boost(self, key: str, step: float = BOOST_DEFAULT) -> PreferenceEntry:
        """
        Increment belief strength by *step* and record evidence.

        The update is optimistic: strength is clipped to STRENGTH_CEIL (1.0)
        but never pushed below its current value by this method.
        """
        with self._lock:
            entry = self._beliefs.get(key)
            if entry is None:
                raise KeyError(f"Belief key '{key}' does not exist. "
                               "Call get_or_create() first.")
            entry.strength = min(STRENGTH_CEIL, entry.strength + step)
            entry.evidence_count += 1
            entry.last_updated = datetime.now(timezone.utc).isoformat()
        return entry

    def demote(self, key: str, step: float = DEMOTE_DEFAULT) -> PreferenceEntry:
        """
        Decrement belief strength by *step* (pessimistic update rule).

        Negative signals erode beliefs faster than positive signals
        build them (demote default 0.2 > boost default 0.1).
        Strength is floored at STRENGTH_FLOOR but NOT removed here;
        call forget() to prune stale beliefs.
        """
        with self._lock:
            entry = self._beliefs.get(key)
            if entry is None:
                raise KeyError(f"Belief key '{key}' does not exist.")
            entry.strength = max(STRENGTH_FLOOR, entry.strength - step)
            entry.evidence_count += 1
            entry.last_updated = datetime.now(timezone.utc).isoformat()
        return entry

    def forget(self) -> list[str]:
        """
        Auto-prune all beliefs whose strength has dropped to STRENGTH_FLOOR.

        Returns the list of keys that were removed.
        """
        with self._lock:
            stale = [k for k, e in self._beliefs.items()
                     if e.strength <= STRENGTH_FLOOR]
            for k in stale:
                del self._beliefs[k]
        return stale

    # ── Helper methods ─────────────────────────────────────────────────────

    def get_or_create(
        self,
        key: str,
        genre: str,
        initial_strength: float = 0.5,
        tags: list[str] | None = None,
    ) -> PreferenceEntry:
        """Return an existing belief or create one with the given defaults."""
        with self._lock:
            if key not in self._beliefs:
                self._beliefs[key] = PreferenceEntry(
                    key=key,
                    genre=genre,
                    strength=initial_strength,
                    tags=tags or [],
                )
        return self._beliefs[key]

    def get(self, key: str) -> PreferenceEntry | None:
        return self._beliefs.get(key)

    def ranked(self, top_k: int | None = None) -> list[PreferenceEntry]:
        """Return beliefs sorted by strength descending."""
        with self._lock:
            ranked = sorted(self._beliefs.values(),
                            key=lambda e: e.strength, reverse=True)
        return ranked[:top_k] if top_k else ranked

    def to_json(self, indent: int = 2) -> str:
        with self._lock:
            data = {k: e.to_dict() for k, e in self._beliefs.items()}
        return json.dumps(data, indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "PreferenceMemory":
        mem = cls()
        data = json.loads(raw)
        for k, d in data.items():
            mem._beliefs[k] = PreferenceEntry.from_dict(d)
        return mem

    def __len__(self) -> int:
        return len(self._beliefs)

    def __repr__(self) -> str:
        return f"PreferenceMemory(beliefs={len(self)})"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — PROFILE MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class ProfileMemory:
    """
    Natural-language distillation of the user's stable personality and goals.

    The profile string is regenerated from the current preference beliefs
    whenever refresh() is called.  It is intentionally human-readable so it
    can be injected directly into an LLM prompt.
    """

    def __init__(self, user_id: int) -> None:
        self.user_id: int = user_id
        self._profile: str = ""
        self._lock = threading.Lock()

    @property
    def text(self) -> str:
        """Current profile string (read-only snapshot)."""
        with self._lock:
            return self._profile

    def refresh(self, preferences: PreferenceMemory) -> str:
        """
        Rebuild the natural-language profile from the current preference beliefs.

        Returns the new profile string.
        """
        ranked = preferences.ranked()
        if not ranked:
            summary = (f"User {self.user_id} has no established preferences yet. "
                       "Their taste profile is still forming.")
        else:
            strong   = [e for e in ranked if e.strength >= 0.7]
            moderate = [e for e in ranked if 0.4 <= e.strength < 0.7]
            weak     = [e for e in ranked if e.strength < 0.4]

            parts: list[str] = [f"User {self.user_id} preference profile:"]

            if strong:
                genres = ", ".join(e.genre for e in strong[:5])
                parts.append(
                    f"Strongly enjoys {genres} "
                    f"(high-confidence beliefs backed by "
                    f"{sum(e.evidence_count for e in strong)} signals)."
                )
            if moderate:
                genres = ", ".join(e.genre for e in moderate[:5])
                parts.append(f"Shows moderate interest in {genres}.")
            if weak:
                genres = ", ".join(e.genre for e in weak[:5])
                parts.append(
                    f"Has weak or fading interest in {genres} "
                    "(may be pruned soon)."
                )

            # Top tags
            all_tags: list[str] = []
            for e in ranked[:10]:
                all_tags.extend(e.tags)
            tag_freq: dict[str, int] = {}
            for t in all_tags:
                tag_freq[t] = tag_freq.get(t, 0) + 1
            top_tags = sorted(tag_freq, key=tag_freq.get, reverse=True)[:6]  # type: ignore[arg-type]
            if top_tags:
                parts.append(
                    f"Recurring content tags: {', '.join(top_tags)}."
                )

            parts.append(
                f"Total active beliefs: {len(ranked)} | "
                f"Strongest: '{ranked[0].genre}' (strength={ranked[0].strength:.2f})."
            )
            summary = " ".join(parts)

        with self._lock:
            self._profile = summary
        return summary

    def __repr__(self) -> str:
        snippet = self._profile[:60] + "..." if len(self._profile) > 60 else self._profile
        return f"ProfileMemory(user_id={self.user_id}, profile='{snippet}')"


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_item_meta(
    item_ids: list[int],
    db_path: Path = DB_PATH,
) -> dict[int, dict[str, Any]]:
    """
    Batch-fetch genre and tags for a list of item IDs from recommender.db.
    Returns {item_id: {"genre": str, "tags": list[str]}}.
    """
    if not item_ids:
        return {}
    placeholders = ",".join("?" * len(item_ids))
    con = sqlite3.connect(db_path)
    rows = con.execute(
        f"SELECT item_id, genre, tags FROM items WHERE item_id IN ({placeholders})",
        item_ids,
    ).fetchall()
    con.close()
    result: dict[int, dict[str, Any]] = {}
    for iid, genre, tags_json in rows:
        try:
            tags = json.loads(tags_json)
        except (json.JSONDecodeError, TypeError):
            tags = []
        result[iid] = {"genre": genre, "tags": tags}
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL — HierarchicalBeliefMemory
# ══════════════════════════════════════════════════════════════════════════════

class HierarchicalBeliefMemory:
    """
    Three-layer Hierarchical Belief-State Memory System.

    Layers
    ------
    event_buffer  (Layer 1) : fast rolling cache of raw events
    preferences   (Layer 2) : structured preference beliefs (JSON)
    profile       (Layer 3) : natural-language user personality string

    State transitions
    -----------------
    log_click(item_id, clicked)       : push a raw event into the buffer
    extract()                         : drain buffer -> update preference beliefs
    boost(key, step=0.1)              : optimistic belief update
    demote(key, step=0.2)             : pessimistic belief update
    forget()                          : prune stale beliefs (strength <= 0.0)

    Persistence
    -----------
    save(path)   / load(path)         : JSON round-trip for the full memory state
    """

    def __init__(
        self,
        user_id: int,
        db_path: Path = DB_PATH,
        buffer_capacity: int = BUFFER_CAPACITY,
    ) -> None:
        self.user_id: int = user_id
        self._db_path: Path = db_path

        self.event_buffer  = EventBuffer(capacity=buffer_capacity)
        self.preferences   = PreferenceMemory()
        self.profile       = ProfileMemory(user_id=user_id)

    # ── Layer 1 public API ─────────────────────────────────────────────────

    def log_click(
        self,
        item_id: int,
        clicked: bool,
        genre: str | None = None,
        tags: list[str] | None = None,
    ) -> ClickEvent:
        """
        Record one interaction event into the rolling event buffer.

        If genre/tags are omitted they are resolved lazily during extract().
        """
        event = ClickEvent(
            item_id=item_id,
            clicked=clicked,
            genre=genre,
            tags=tags or [],
        )
        self.event_buffer.push(event)
        return event

    # ── Layer 2 public API (state transitions) ─────────────────────────────

    def extract(self) -> dict[str, list[str]]:
        """
        Drain the event buffer and update preference beliefs.

        Algorithm
        ---------
        1. Collect all buffered events (buffer is cleared after drain).
        2. Batch-fetch missing item metadata (genre, tags) from the database.
        3. For each event:
           - clicked=True  -> boost genre belief by CLICK_BOOST
           - clicked=False -> demote genre belief by SKIP_DEMOTE
        4. Refresh the natural-language profile from updated beliefs.
        5. Return a summary dict {"boosted": [...], "demoted": [...]}.
        """
        events = self.event_buffer.drain()
        if not events:
            return {"boosted": [], "demoted": []}

        # Resolve missing metadata in one DB round-trip
        missing_ids = [e.item_id for e in events if e.genre is None]
        meta = _fetch_item_meta(missing_ids, self._db_path)
        for e in events:
            if e.genre is None:
                m = meta.get(e.item_id, {})
                e.genre = m.get("genre", "Unknown")
                e.tags  = m.get("tags", [])

        boosted: list[str] = []
        demoted: list[str] = []

        for event in events:
            genre = event.genre or "Unknown"
            key   = f"genre:{genre}"

            # Ensure belief slot exists
            self.preferences.get_or_create(key, genre=genre, tags=event.tags)

            if event.clicked:
                self.preferences.boost(key, step=CLICK_BOOST)
                boosted.append(key)
            else:
                self.preferences.demote(key, step=SKIP_DEMOTE)
                demoted.append(key)

        # Prune and rebuild profile
        self.forget()
        self.profile.refresh(self.preferences)

        return {"boosted": boosted, "demoted": demoted}

    def boost(self, key: str, step: float = BOOST_DEFAULT) -> PreferenceEntry:
        """
        Manually boost a preference belief (optimistic update).

        Parameters
        ----------
        key  : Preference key, e.g. "genre:Action"
        step : Increment amount (default 0.1)
        """
        entry = self.preferences.boost(key, step)
        self.profile.refresh(self.preferences)
        return entry

    def demote(self, key: str, step: float = DEMOTE_DEFAULT) -> PreferenceEntry:
        """
        Manually demote a preference belief (pessimistic update).

        step defaults to 0.2 — intentionally larger than boost (0.1) so that
        negative signals erode beliefs faster than positive ones build them.
        """
        entry = self.preferences.demote(key, step)
        self.forget()
        self.profile.refresh(self.preferences)
        return entry

    def forget(self) -> list[str]:
        """
        Auto-prune stale preference beliefs (strength <= STRENGTH_FLOOR).

        Returns the list of pruned keys.
        """
        pruned = self.preferences.forget()
        if pruned:
            self.profile.refresh(self.preferences)
        return pruned

    # ── Convenience queries ────────────────────────────────────────────────

    def ranked_preferences(self, top_k: int = 10) -> list[PreferenceEntry]:
        """Return top-k preference beliefs sorted by strength descending."""
        return self.preferences.ranked(top_k=top_k)

    def snapshot(self) -> dict[str, Any]:
        """Return a complete in-memory snapshot as a plain dict."""
        return {
            "user_id":      self.user_id,
            "event_buffer": [e.to_dict() for e in self.event_buffer.peek()],
            "preferences":  json.loads(self.preferences.to_json()),
            "profile":      self.profile.text,
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise the full memory state to a JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.snapshot(), fh, indent=2, ensure_ascii=False)
        print(f"[MEMORY] Saved -> {path}  ({path.stat().st_size / 1024:.1f} KB)")

    @classmethod
    def load(cls, path: str | Path, db_path: Path = DB_PATH) -> "HierarchicalBeliefMemory":
        """Restore a HierarchicalBeliefMemory instance from a JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        mem = cls(user_id=data["user_id"], db_path=db_path)

        # Restore event buffer
        for ev in data.get("event_buffer", []):
            mem.event_buffer.push(ClickEvent(**ev))

        # Restore preference beliefs
        for key, entry_dict in data.get("preferences", {}).items():
            mem.preferences._beliefs[key] = PreferenceEntry.from_dict(entry_dict)

        # Restore profile text directly (no need to recompute)
        with mem.profile._lock:
            mem.profile._profile = data.get("profile", "")

        print(f"[MEMORY] Loaded <- {path}")
        return mem

    def __repr__(self) -> str:
        return (
            f"HierarchicalBeliefMemory("
            f"user_id={self.user_id}, "
            f"buffer={len(self.event_buffer)}, "
            f"beliefs={len(self.preferences)}, "
            f"profile_chars={len(self.profile.text)})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# DEMO / SMOKE-TEST
# ══════════════════════════════════════════════════════════════════════════════

def _demo() -> None:
    print("=" * 65)
    print("  Hierarchical Belief-State Memory System — Demo")
    print("=" * 65)

    mem = HierarchicalBeliefMemory(user_id=42, db_path=DB_PATH)
    print(f"\n[1] Initial state: {mem}")

    # Simulate a session: user clicks RPG/Action items, skips Horror
    print("\n[2] Logging 10 session events ...")
    clicks = [
        (315, True),  (42,  True),  (88,  True),  (7,   False),
        (200, True),  (100, True),  (500, False),  (750, True),
        (315, True),  (42,  True),
    ]
    for iid, clicked in clicks:
        mem.log_click(iid, clicked)
    print(f"    Buffer size: {len(mem.event_buffer)}")

    # extract() drains buffer and updates preferences
    print("\n[3] Calling extract() ...")
    result = mem.extract()
    print(f"    Boosted : {result['boosted'][:5]}")
    print(f"    Demoted : {result['demoted'][:5]}")
    print(f"    Beliefs : {len(mem.preferences)}")

    # Show ranked preferences
    print("\n[4] Ranked preferences (top 5):")
    print(f"    {'Key':<25} {'Strength':>8}  {'Evidence':>8}  Genre")
    print(f"    {'─'*25} {'─'*8}  {'─'*8}  {'─'*10}")
    for entry in mem.ranked_preferences(top_k=5):
        print(f"    {entry.key:<25} {entry.strength:>8.3f}  "
              f"{entry.evidence_count:>8}  {entry.genre}")

    # Manual boost/demote
    print("\n[5] Manual boost('genre:RPG', step=0.2) ...")
    key = next((e.key for e in mem.ranked_preferences() if "RPG" in e.key), None)
    if key:
        mem.boost(key, step=0.2)
        e = mem.preferences.get(key)
        print(f"    {key}: strength={e.strength:.3f}")

    print("\n[6] Profile Memory (natural language):")
    print(f"    {mem.profile.text}")

    # Persistence round-trip
    save_path = Path(__file__).parent / f"memory_user{mem.user_id}.json"
    print(f"\n[7] Saving to {save_path.name} ...")
    mem.save(save_path)

    print("\n[8] Loading back ...")
    mem2 = HierarchicalBeliefMemory.load(save_path)
    print(f"    Restored: {mem2}")
    print(f"    Profile : {mem2.profile.text[:80]}...")

    print("\n[DONE]")
    print("=" * 65)


if __name__ == "__main__":
    _demo()
