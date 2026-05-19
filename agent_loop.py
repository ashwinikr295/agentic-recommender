"""
agent_loop.py
=============
Autonomous Dynamic Personalization & Recommendation Engine
----------------------------------------------------------
Step 4: Agentic Control Loop — structured Python StateGraph (ReAct pattern)

Architecture
~~~~~~~~~~~~
A four-node ReAct loop links the SQLite database, PyTorch sequential model,
and Hierarchical Belief-State Memory into a unified agent:

  [START]
     |
     v
  Node 1 — ParseIntent        parse user text -> structured constraints
     |
     v
  Node 2 — RetrieveCandidates get_raw_recommendations() + SQL filtering
     |
     v
  Node 3 — MemoryAlignment    update beliefs, rerank pool
     |
     v
  Node 4 — Generation         LLM -> HTML + reasoning
     |
     v
  [END]

Step Counter guard: if step_count > MAX_STEPS (5), skip to Generation
with a fallback template — prevents infinite loops.

LLM priority
~~~~~~~~~~~~
  1. Claude Sonnet via anthropic  (ANTHROPIC_API_KEY env var)
  2. Gemini Pro via google-generativeai (GOOGLE_API_KEY env var)
  3. Deterministic template fallback (no API key needed)

Run:
    python agent_loop.py
    python agent_loop.py --user 7 --query "Action games under $30"
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Project modules ────────────────────────────────────────────────────────────
from deep_model import get_raw_recommendations
from memory import HierarchicalBeliefMemory, PreferenceEntry

DB_PATH    = Path(__file__).parent / "recommender.db"
MEM_DIR    = Path(__file__).parent
MAX_STEPS  = 5
TOP_K_RAW  = 50   # candidates fetched from the model
TOP_K_FINAL = 10  # items shown to the user


# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentState:
    """
    Mutable container passed through every node in the control loop.
    Each node reads the state, mutates it in-place, and returns it.
    """
    # ── Inputs ─────────────────────────────────────────────────────────────
    user_id: int                                      = 1
    raw_query: str                                    = ""
    media_type: str                                   = "games" ""

    # ── Parsed intent (Node 1) ─────────────────────────────────────────────
    parsed_intent: dict[str, Any]                     = field(default_factory=dict)
    # e.g. {"genre": "Action", "max_price": 30.0, "min_rating": 4.0,
    #        "tags": ["open-world"], "freetext": "Action games under $30"}

    # ── Candidate pool (Node 2) ────────────────────────────────────────────
    raw_candidates: list[tuple[int, float]]           = field(default_factory=list)
    # [(item_id, score), ...]  straight from the deep model

    filtered_candidates: list[dict[str, Any]]         = field(default_factory=list)
    # [{item_id, title, genre, price, rating, score, tags}, ...]

    # ── Reranked pool (Node 3) ─────────────────────────────────────────────
    reranked_candidates: list[dict[str, Any]]         = field(default_factory=list)
    memory_snapshot: dict[str, Any]                   = field(default_factory=dict)

    # ── Final output (Node 4) ──────────────────────────────────────────────
    html_output: str                                  = ""
    reasoning: str                                    = ""

    # ── Control ────────────────────────────────────────────────────────────
    step_count: int                                   = 0
    error: str | None                                 = None
    done: bool                                        = False
    logs: list[str]                                   = field(default_factory=list)
    """Chain-of-thought log captured during execution for UI display."""


# ══════════════════════════════════════════════════════════════════════════════
# MINI STATE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

NodeFn = Callable[[AgentState], AgentState]

class StateGraph:
    """
    Lightweight directed graph executor mirroring LangGraph's API surface.

    Usage
    -----
    graph = StateGraph()
    graph.add_node("parse",    node_parse_intent)
    graph.add_node("retrieve", node_retrieve_candidates)
    graph.add_edge("parse",    "retrieve")
    graph.add_conditional_edge("retrieve", router_fn, {"memory": "memory", "end": END})
    compiled = graph.compile(entry="parse")
    final_state = compiled.invoke(initial_state)
    """

    END = "__END__"

    def __init__(self) -> None:
        self._nodes: dict[str, NodeFn]  = {}
        self._edges: dict[str, str]     = {}          # unconditional
        self._cond:  dict[str, tuple]   = {}          # conditional (fn, mapping)

    def add_node(self, name: str, fn: NodeFn) -> None:
        self._nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self._edges[src] = dst

    def add_conditional_edge(
        self,
        src: str,
        condition_fn: Callable[[AgentState], str],
        mapping: dict[str, str],
    ) -> None:
        self._cond[src] = (condition_fn, mapping)

    def compile(self, entry: str) -> "CompiledGraph":
        return CompiledGraph(entry, self._nodes, self._edges, self._cond)


class CompiledGraph:
    def __init__(self, entry, nodes, edges, cond) -> None:
        self._entry = entry
        self._nodes = nodes
        self._edges = edges
        self._cond  = cond

    def invoke(self, state: AgentState) -> AgentState:
        current = self._entry
        while current != StateGraph.END:
            if current not in self._nodes:
                state.error = f"Unknown node: '{current}'"
                break
            fn = self._nodes[current]
            _log(f"[GRAPH] -> {current.upper()}", state)
            state = fn(state)
            state.step_count += 1

            # Determine next node
            if current in self._cond:
                cond_fn, mapping = self._cond[current]
                route = cond_fn(state)
                current = mapping.get(route, StateGraph.END)
            elif current in self._edges:
                current = self._edges[current]
            else:
                break
        return state


# ══════════════════════════════════════════════════════════════════════════════
# LLM CLIENT  (Claude -> Gemini -> template fallback)
# ══════════════════════════════════════════════════════════════════════════════

class LLMClient:
    """
    Unified LLM abstraction with automatic provider selection.
    Priority: Claude Sonnet 4.5 > Gemini 2.0 Flash > template fallback.
    """

    def __init__(self) -> None:
        self._backend: str = "template"
        self._client:  Any = None
        self._model:   str = ""
        self._init()

    def _init(self) -> None:
        # 1 — Try Anthropic Claude
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            try:
                import anthropic
                self._client  = anthropic.Anthropic(api_key=anthropic_key)
                self._backend = "claude"
                self._model   = "claude-sonnet-4-5"
                print(f"[LLM] Backend: Claude ({self._model})")
                return
            except Exception as exc:
                print(f"[LLM] Claude init failed: {exc}")

        # 2 — Try Gemini
        google_key = os.getenv("GOOGLE_API_KEY", "")
        if google_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_key)
                self._client  = genai.GenerativeModel("gemini-2.0-flash")
                self._backend = "gemini"
                self._model   = "gemini-2.0-flash"
                print(f"[LLM] Backend: Gemini ({self._model})")
                return
            except Exception as exc:
                print(f"[LLM] Gemini init failed: {exc}")

        print("[LLM] No API key found -- using deterministic template fallback")

    def complete(self, system: str, user: str) -> str:
        """Send a prompt and return the text response."""
        try:
            if self._backend == "claude":
                return self._claude(system, user)
            if self._backend == "gemini":
                return self._gemini(system, user)
        except Exception as exc:
            _log(f"[LLM] API call failed ({exc}), falling back to template")
        return ""   # caller uses template on empty string

    def _claude(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    def _gemini(self, system: str, user: str) -> str:
        resp = self._client.generate_content(f"{system}\n\n{user}")
        return resp.text


_LLM = LLMClient()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _log(msg: str, state: "AgentState | None" = None) -> None:
    print(msg)
    if state is not None:
        state.logs.append(msg)


def _fetch_items(item_ids: list[int], media_type: str = "games") -> dict[int, dict]:
    """Batch-fetch item metadata from recommender.db."""
    if not item_ids:
        return {}
    ph  = ",".join("?" * len(item_ids))
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    if media_type == "movies":
        rows = con.execute(
            f"SELECT movie_id, title, genre, tags, price, rating, overview FROM movies "
            f"WHERE movie_id IN ({ph})",
            item_ids,
        ).fetchall()
    else:
        rows = con.execute(
            f"SELECT item_id, title, genre, tags, price, rating FROM items "
            f"WHERE item_id IN ({ph})",
            item_ids,
        ).fetchall()
    con.close()
    result = {}
    for r in rows:
        iid = r[0]
        result[iid] = {
            "item_id": iid,
            "title":   r["title"],
            "genre":   r["genre"],
            "tags":    json.loads(r["tags"]),
            "price":   r["price"],
            "rating":  r["rating"],
        }
        if media_type == "movies" and "overview" in r.keys():
            result[iid]["overview"] = r["overview"]
    return result


def _sql_filter(
    candidates: list[dict],
    intent: dict[str, Any],
) -> list[dict]:
    """
    Apply parsed intent constraints via in-process filtering
    (avoids a second DB round-trip — metadata already loaded).
    """
    genre     = intent.get("genre", "").lower()
    max_price = intent.get("max_price")
    min_rating = intent.get("min_rating", 0.0)
    tags_req  = [t.lower() for t in intent.get("tags", [])]

    out = []
    for item in candidates:
        if genre and item["genre"].lower() != genre:
            continue
        if max_price is not None and item["price"] > max_price:
            continue
        if item["rating"] < min_rating:
            continue
        if tags_req:
            item_tags = [t.lower() for t in item["tags"]]
            if not any(t in item_tags for t in tags_req):
                continue
        out.append(item)
    return out


def _memory_path(user_id: int, media_type: str = "games") -> Path:
    return MEM_DIR / f"memory_user{user_id}_{media_type}.json"


def _load_or_create_memory(user_id: int, media_type: str = "games") -> HierarchicalBeliefMemory:
    p = _memory_path(user_id, media_type)
    if p.exists():
        return HierarchicalBeliefMemory.load(p)
    return HierarchicalBeliefMemory(user_id=user_id, db_path=DB_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — PARSE INTENT
# ══════════════════════════════════════════════════════════════════════════════

_GENRES = {
    "action","adventure","rpg","strategy","simulation","sports","racing",
    "horror","puzzle","platformer","fighting","shooter","mmorpg","roguelike",
    "sandbox","stealth","survival","visual novel","rhythm","card game",
}

def node_parse_intent(state: AgentState) -> AgentState:
    """
    Node 1 — Parse Intent
    Extract genre, max_price, min_rating, and tag constraints
    from raw user text using regex + keyword matching.
    """
    query = state.raw_query.lower()
    intent: dict[str, Any] = {"freetext": state.raw_query}

    # Dynamic genre selection based on media_type
    if state.media_type == "movies":
        genres = {"sci-fi", "horror", "animation", "romance", "comedy", "action", "adventure", "fantasy", "thriller", "drama"}
    else:
        genres = _GENRES

    # Genre detection
    for g in sorted(genres, key=len, reverse=True):
        if g in query:
            # Preserve acronym casing
            intent["genre"] = g.upper() if len(g) <= 4 else g.title()
            break

    # Budget / price
    price_match = re.search(
        r"(?:under|below|less than|<|max|budget)\s*\$?\s*(\d+(?:\.\d+)?)", query
    )
    if price_match:
        intent["max_price"] = float(price_match.group(1))

    # Rating floor
    rating_match = re.search(
        r"(?:rating|rated|score)\s*(?:above|over|>=|>|at least)?\s*(\d(?:\.\d)?)\s*\+?",
        query,
    )
    if rating_match:
        intent["min_rating"] = float(rating_match.group(1))

    # Free games/movies
    if re.search(r"\bfree\b", query):
        intent["max_price"] = 0.0

    # Tags (known tag pool subset based on media_type)
    if state.media_type == "movies":
        tags_pool = ["space", "superhero", "monsters", "survival", "post-apocalyptic", "heist", "family", "history", "paranormal", "combat"]
    else:
        tags_pool = ["open-world","multiplayer","co-op","story-rich","pixel-art","roguelike","indie","casual","hardcore","pvp","soulslike"]
    intent["tags"] = [t for t in tags_pool if t in query]

    # Extract search terms for title match (ignore query verbs, filter words, and genres)
    stop_words = {
        "and", "the", "with", "under", "below", "less", "than", "free", "rating",
        "rated", "above", "over", "games", "game", "movies", "movie", "show", "shows",
        "recommender", "recommend", "please", "search", "find", "get", "me", "some",
        "a", "an", "for", "in", "of", "to", "or", "is", "it", "at", "by", "from",
        "like", "want", "need", "about"
    }
    genre_words = set(intent.get("genre", "").lower().split())
    query_words = [
        w.strip(",.!?\"'") for w in query.split()
        if w.strip(",.!?\"'")
    ]
    search_words = [
        w for w in query_words
        if w not in stop_words and w not in genre_words
    ]
    intent["search_words"] = search_words

    state.parsed_intent = intent
    _log(f"    Intent parsed: {intent}", state)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — RETRIEVE CANDIDATES
# ══════════════════════════════════════════════════════════════════════════════

def node_retrieve_candidates(state: AgentState) -> AgentState:
    """
    Node 2 — Retrieve Candidates (Dual-Track)
    """
    intent     = state.parsed_intent
    genre      = intent.get("genre", "")
    max_price  = intent.get("max_price")
    min_rating = intent.get("min_rating", 0.0)
    tags_req   = [t.lower() for t in intent.get("tags", [])]

    con = sqlite3.connect(DB_PATH)

    # ── Track A: DB-direct ─────────────────────────────────────────────────
    clauses: list[str] = []
    params:  list      = []
    if genre:
        clauses.append("genre = ?")
        params.append(genre)
    if max_price is not None:
        clauses.append("price <= ?")
        params.append(max_price)
    if min_rating:
        clauses.append("rating >= ?")
        params.append(min_rating)

    # Keyword title search
    search_words = intent.get("search_words", [])
    if search_words:
        term_clauses = [f"title LIKE ?" for _ in search_words]
        clauses.append("(" + " OR ".join(term_clauses) + ")")
        params.extend([f"%{w}%" for w in search_words])

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    
    if state.media_type == "movies":
        rows  = con.execute(
            f"SELECT movie_id, title, genre, tags, price, rating, overview "
            f"FROM movies {where} ORDER BY popularity DESC LIMIT ?",
            params + [TOP_K_RAW],
        ).fetchall()
    else:
        rows  = con.execute(
            f"SELECT item_id, title, genre, tags, price, rating "
            f"FROM items {where} ORDER BY rating DESC LIMIT ?",
            params + [TOP_K_RAW],
        ).fetchall()

    db_items: dict[int, dict] = {}
    for r in rows:
        try:
            tags = json.loads(r[3])
        except Exception:
            tags = []
        if tags_req and not any(t in [x.lower() for x in tags] for t in tags_req):
            continue
        item_data = {
            "item_id": r[0], "title": r[1], "genre": r[2],
            "tags": tags, "price": r[4], "rating": r[5],
            "score": r[5] / 5.0,
        }
        if state.media_type == "movies" and len(r) > 6:
            item_data["overview"] = r[6]
        db_items[r[0]] = item_data
    _log(f"    Track A (DB): {len(db_items)} items matching constraints", state)

    # ── Track B: Model / Popularity Track ──────────────────────────────────
    if state.media_type == "movies":
        if genre:
            seed_rows = con.execute(
                "SELECT movie_id, title, genre, tags, price, rating, overview, popularity "
                "FROM movies WHERE genre=? ORDER BY popularity DESC LIMIT ?",
                (genre, TOP_K_RAW),
            ).fetchall()
            seed_label = f"popular-{genre}"
        else:
            mem       = _load_or_create_memory(state.user_id, state.media_type)
            top_prefs = mem.ranked_preferences(top_k=1)
            if top_prefs:
                target_g = top_prefs[0].genre
                seed_rows = con.execute(
                    "SELECT movie_id, title, genre, tags, price, rating, overview, popularity "
                    "FROM movies WHERE genre=? ORDER BY popularity DESC LIMIT ?",
                    (target_g, TOP_K_RAW),
                ).fetchall()
                seed_label = f"popular-{target_g} (memory)"
            else:
                seed_rows = con.execute(
                    "SELECT movie_id, title, genre, tags, price, rating, overview, popularity "
                    "FROM movies ORDER BY popularity DESC LIMIT ?",
                    (TOP_K_RAW,),
                ).fetchall()
                seed_label = "top-popular"

        model_items: dict[int, dict] = {}
        for r in seed_rows:
            try:
                tags = json.loads(r[3])
            except Exception:
                tags = []
            model_items[r[0]] = {
                "item_id": r[0], "title": r[1], "genre": r[2],
                "tags": tags, "price": r[4], "rating": r[5],
                "score": r[5] / 5.0,
                "overview": r[6],
            }
        state.raw_candidates = [(r[0], r[7]) for r in seed_rows]
        _log(f"    Track B (popularity): {len(model_items)} items (seed: {seed_label})", state)
    else:
        if genre:
            seed_ids = [r[0] for r in con.execute(
                "SELECT item_id FROM items WHERE genre=? ORDER BY rating DESC LIMIT 5",
                (genre,),
            ).fetchall()]
            seed_label = genre
        else:
            mem       = _load_or_create_memory(state.user_id, state.media_type)
            top_prefs = mem.ranked_preferences(top_k=1)
            if top_prefs:
                seed_ids = [r[0] for r in con.execute(
                    "SELECT item_id FROM items WHERE genre=? ORDER BY rating DESC LIMIT 5",
                    (top_prefs[0].genre,),
                ).fetchall()]
                seed_label = top_prefs[0].genre
            else:
                seed_ids  = [r[0] for r in con.execute(
                    "SELECT item_id FROM items ORDER BY rating DESC LIMIT 5"
                ).fetchall()]
                seed_label = "top-rated"

        raw = get_raw_recommendations(seed_ids, top_k=TOP_K_RAW)
        state.raw_candidates = raw
        model_meta = _fetch_items([iid for iid, _ in raw], state.media_type)
        model_items: dict[int, dict] = {}
        for iid, score in raw:
            if iid in model_meta:
                item = dict(model_meta[iid])
                item["score"] = score
                model_items[iid] = item
        _log(f"    Track B (model): {len(model_items)} items (seed: {seed_label})", state)

    con.close()

    # ── Merge: DB items first (directly relevant), then extra model items ─────
    merged: dict[int, dict] = dict(db_items)
    for iid, item in model_items.items():
        if iid not in merged:
            if not genre or item.get("genre", "").lower() == genre.lower():
                merged[iid] = item

    all_items = list(merged.values())

    # Apply soft constraints to merged pool
    filtered = _sql_filter(all_items, intent)
    if not filtered and genre:
        filtered = _sql_filter(all_items, {"genre": genre})
        if filtered:
            _log("    Relaxation: kept genre, dropped price/rating", state)
    if not filtered:
        filtered = all_items
        _log("    Relaxation: returning full merged pool", state)

    state.filtered_candidates = filtered[:TOP_K_FINAL * 3]
    _log(f"    Merged pool: {len(state.filtered_candidates)} items", state)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — MEMORY ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def node_memory_alignment(state: AgentState) -> AgentState:
    """
    Node 3 — Memory Alignment
    """
    mem = _load_or_create_memory(state.user_id, state.media_type)

    for item in state.filtered_candidates:
        mem.log_click(
            item_id=item["item_id"],
            clicked=False,
            genre=item["genre"],
            tags=item["tags"],
        )

    mem.extract()

    search_words = state.parsed_intent.get("search_words", [])

    def affinity(item: dict) -> float:
        key   = f"genre:{item['genre']}"
        entry = mem.preferences.get(key)
        strength = entry.strength if entry else 0.5
        score = item["score"] * (0.5 + strength)
        
        # Title match boost
        if search_words:
            title_lower = item["title"].lower()
            matches = sum(1 for w in search_words if w in title_lower)
            if matches > 0:
                match_ratio = matches / len(search_words)
                score += 10.0 * match_ratio
        return score

    reranked = sorted(state.filtered_candidates, key=affinity, reverse=True)
    state.reranked_candidates = reranked[:TOP_K_FINAL]
    state.memory_snapshot = {
        "profile": mem.profile.text,
        "top_beliefs": [
            {"key": e.key, "strength": e.strength, "evidence": e.evidence_count}
            for e in mem.ranked_preferences(top_k=5)
        ],
    }

    mem.save(_memory_path(state.user_id, state.media_type))
    _log(f"    Reranked to top {len(state.reranked_candidates)} | "
         f"Active beliefs: {len(mem.preferences)}", state)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — GENERATION
# ══════════════════════════════════════════════════════════════════════════════

_HTML_SYSTEM = textwrap.dedent("""\
    You are a recommendation UI generator for a video game store.
    Given a list of game recommendations and a user profile, output:
    1. A short paragraph of conversational reasoning (2-3 sentences) explaining WHY
       these games were selected for this user.
    2. An HTML <section> block with the top games rendered as visually-styled cards
       (use inline styles — no external CSS). Each card must show: title, genre,
       price, rating, and a one-line pitch sentence.
    Output format (strict):
    <reasoning>
    ...your reasoning here...
    </reasoning>
    <html>
    ...your HTML here...
    </html>
""")

def _build_llm_user_prompt(state: AgentState) -> str:
    items_txt = "\n".join(
        f"  {i+1}. {it['title']} ({it['genre']}) | "
        f"${it['price']:.2f} | Rating {it['rating']} | Tags: {', '.join(it['tags'][:3])}"
        for i, it in enumerate(state.reranked_candidates)
    )
    return (
        f"User query: \"{state.raw_query}\"\n\n"
        f"User profile: {state.memory_snapshot.get('profile', 'unknown')}\n\n"
        f"Top recommendations:\n{items_txt}\n\n"
        "Generate the reasoning paragraph and HTML card section."
    )


def _template_html(state: AgentState) -> tuple[str, str]:
    """Deterministic HTML fallback — generates specific reasoning from actual results."""
    items  = state.reranked_candidates
    intent = state.parsed_intent
    query  = state.raw_query
    genre  = intent.get("genre", "")
    is_movie = (state.media_type == "movies")
    media_lbl = "movie" if is_movie else "game"

    if not items:
        return f"No matching {media_lbl}s found.", "<p>No results. Try a different query.</p>"

    top3         = [it["title"] for it in items[:3]]
    genres_seen  = list(dict.fromkeys(it["genre"] for it in items))
    avg_rating   = sum(it["rating"] for it in items) / len(items)
    free_count   = sum(1 for it in items if it["price"] == 0)
    price_cap    = intent.get("max_price")
    rating_floor = intent.get("min_rating")

    parts = []
    if genre:
        parts.append(f'Your query for "{query}" matched {len(items)} {genre} {media_lbl}{"s" if len(items) != 1 else ""}.')
    else:
        g_str = " and ".join(genres_seen[:2]) if len(genres_seen) <= 2 \
                else ", ".join(genres_seen[:2]) + f" and {len(genres_seen)-2} more"
        parts.append(f'Found {len(items)} {media_lbl}{"s" if len(items) != 1 else ""} across {g_str} for "{query}".')

    if len(top3) == 1:
        parts.append(f"Top pick: {top3[0]}.")
    elif len(top3) == 2:
        parts.append(f"Top picks: {top3[0]} and {top3[1]}.")
    else:
        parts.append(f"Top picks: {top3[0]}, {top3[1]}, and {top3[2]}.")

    if rating_floor:
        parts.append(f"All picks meet your rating floor of {rating_floor}★ (avg: {avg_rating:.1f}★).")
    else:
        parts.append(f"Average rating: {avg_rating:.1f}★.")

    if is_movie:
        if free_count:
            parts.append(f"{free_count} movie{'s are' if free_count > 1 else ' is'} free to watch.")
        elif price_cap is not None:
            parts.append(f"All movies are priced at ${price_cap:.0f} or below.")
    else:
        if free_count:
            parts.append(f"{free_count} title{'s are' if free_count > 1 else ' is'} free to play.")
        elif price_cap is not None:
            parts.append(f"All titles are priced at ${price_cap:.0f} or below.")

    reasoning = " ".join(parts)

    cards = ""
    for it in items:
        price_str = "Free" if it["price"] == 0 else (f"Rent ${it['price']:.2f}" if is_movie else f"${it['price']:.2f}")
        stars_val = round(it["rating"])
        full_s    = chr(9733) * stars_val
        empty_s   = chr(9734) * (5 - stars_val)
        tags      = it.get("tags", [])[:3]
        if is_movie:
            pitch = it.get("overview", "")
            if pitch:
                if len(pitch) > 150:
                    pitch = pitch[:147] + "..."
            else:
                pitch = f"A {it['genre'].lower()} movie featuring {', '.join(tags)}."
        else:
            pitch     = (f"A {it['genre'].lower()} experience featuring {', '.join(tags)}."
                         if tags else
                         f"A top-rated {it['genre'].lower()} game with strong player reviews.")
        cards += (
            '<div style="background:#1a1a38;border-radius:12px;padding:16px;margin:8px 0;'
            'border-left:4px solid #7c3aed;font-family:sans-serif;color:#e2e8f0;">'
            f'<div style="font-size:1.05em;font-weight:700;color:#c4b5fd">{it["title"]}</div>'
            f'<div style="font-size:0.82em;color:#94a3b8;margin:4px 0">'
            f'{it["genre"]} &nbsp;|&nbsp; {price_str} &nbsp;|&nbsp; {full_s}{empty_s} {it["rating"]}'
            '</div>'
            f'<div style="font-size:0.87em;margin-top:6px;color:#94a3b8">{pitch}</div>'
            '</div>'
        )

    html = (
        '<section style="max-width:640px;margin:auto">'
        '<h2 style="color:#a78bfa;font-family:sans-serif;margin-bottom:4px">Recommended For You</h2>'
        f'<p style="color:#64748b;font-size:0.8em;margin-top:0">Query: <em>{query}</em></p>'
        f'{cards}</section>'
    )
    return reasoning, html

def _parse_llm_response(text: str) -> tuple[str, str]:
    reasoning = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    html      = re.search(r"<html>(.*?)</html>", text, re.DOTALL)
    return (
        reasoning.group(1).strip() if reasoning else "",
        html.group(1).strip() if html else "",
    )

_HTML_SYSTEM_MOVIES = textwrap.dedent("""\
    You are a recommendation UI generator for a movie store.
    Given a list of movie recommendations and a user profile, output:
    1. A short paragraph of conversational reasoning (2-3 sentences) explaining WHY
       these movies were selected for this user.
    2. An HTML <section> block with the top movies rendered as visually-styled cards
       (use inline styles — no external CSS). Each card must show: title, genre,
       price (or "Rent" / "Free"), rating, and a one-line pitch sentence using its overview.
    Output format (strict):
    <reasoning>
    ...your reasoning here...
    </reasoning>
    <html>
    ...your HTML here...
    </html>
""")

def node_generation(state: AgentState) -> AgentState:
    """
    Node 4 — Generation
    """
    if not state.reranked_candidates:
        media_lbl = "movies" if state.media_type == "movies" else "games"
        state.reasoning   = f"No matching {media_lbl} found for your query."
        state.html_output = "<p>No results found. Try a different query.</p>"
        state.done = True
        return state

    system_prompt = _HTML_SYSTEM_MOVIES if state.media_type == "movies" else _HTML_SYSTEM
    llm_text = _LLM.complete(system_prompt, _build_llm_user_prompt(state))

    if llm_text:
        reasoning, html = _parse_llm_response(llm_text)
        if reasoning and html:
            state.reasoning   = reasoning
            state.html_output = html
            state.done = True
            _log(f"    LLM ({_LLM._backend}) generated output", state)
            return state

    _log("    Using deterministic template fallback", state)
    state.reasoning, state.html_output = _template_html(state)
    state.done = True
    return state

    llm_text = _LLM.complete(_HTML_SYSTEM, _build_llm_user_prompt(state))

    if llm_text:
        reasoning, html = _parse_llm_response(llm_text)
        if reasoning and html:
            state.reasoning   = reasoning
            state.html_output = html
            state.done = True
            _log(f"    LLM ({_LLM._backend}) generated output", state)
            return state

    _log("    Using deterministic template fallback", state)
    state.reasoning, state.html_output = _template_html(state)
    state.done = True
    return state


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING (Step Counter Guard)
# ══════════════════════════════════════════════════════════════════════════════

def _guard_router(state: AgentState) -> str:
    """Skip to generation if step limit reached or error detected."""
    if state.step_count >= MAX_STEPS or state.error:
        _log(f"    [GUARD] Step limit reached ({state.step_count}/{MAX_STEPS}) "
             "-> forcing Generation")
        return "generate"
    return "memory"


# ══════════════════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> "CompiledGraph":
    g = StateGraph()
    g.add_node("parse",    node_parse_intent)
    g.add_node("retrieve", node_retrieve_candidates)
    g.add_node("memory",   node_memory_alignment)
    g.add_node("generate", node_generation)

    g.add_edge("parse", "retrieve")
    g.add_conditional_edge(
        "retrieve",
        _guard_router,
        {"memory": "memory", "generate": "generate"},
    )
    g.add_edge("memory",   "generate")
    g.add_edge("generate", StateGraph.END)

    return g.compile(entry="parse")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(user_id: int, query: str, media_type: str = "games") -> AgentState:
    """
    Execute the full agentic recommendation loop.
    """
    print("=" * 65)
    print(f"  Agentic Recommender Loop ({media_type.upper()})")
    print(f"  user_id={user_id}  query=\"{query}\"")
    print("=" * 65)

    state = AgentState(user_id=user_id, raw_query=query, media_type=media_type)
    graph = build_graph()
    final = graph.invoke(state)

    print(f"\n[RESULT] Steps taken: {final.step_count}")
    print(f"[RESULT] Candidates shown: {len(final.reranked_candidates)}")
    print(f"[RESULT] Reasoning:\n  {final.reasoning}\n")

    return final


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agentic Recommendation Loop")
    parser.add_argument("--user",  type=int, default=42,
                        help="User ID (default 42)")
    parser.add_argument("--query", type=str,
                        default="Action movies with high ratings",
                        help="Natural language query")
    parser.add_argument("--media", type=str, default="games",
                        help="games or movies")
    args = parser.parse_args()

    final = run_agent(user_id=args.user, query=args.query, media_type=args.media)

    # Write HTML output to file for inspection
    out_path = Path(__file__).parent / f"recommendations_user{args.user}.html"
    out_path.write_text(
        f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<title>Recommendations — User {args.user}</title>
<style>body{{background:#0f0f1a;padding:24px}}</style>
</head><body>
<p style="color:#94a3b8;font-family:sans-serif">{final.reasoning}</p>
{final.html_output}
</body></html>""",
        encoding="utf-8",
    )
    print(f"[OUTPUT] HTML saved -> {out_path}")
    print("=" * 65)
