"""
app.py
======
Autonomous Dynamic Personalization & Recommendation Engine
----------------------------------------------------------
Step 5: Interactive Streamlit Web Interface

Three visible sections
~~~~~~~~~~~~~~~~~~~~~~
  1. Chat Interface         — natural-language query input with conversation history
  2. Dynamic Page Layout    — rendered recommendation cards (HTML) with icons & prices
  3. Agent Mind Dashboard   — live chain-of-thought execution log (expandable)

Run:
    streamlit run app.py
"""

import sys
import streamlit as st
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Agentic Recommender",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project modules ────────────────────────────────────────────────────────────
from agent_loop import run_agent, AgentState, MAX_STEPS

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"]       { font-family: 'Inter', sans-serif; }
.stApp                            { background: #0b0f19 !important; }
p, li, span, label, div          { color: #e2e8f0; }

/* ── Streamlit native overrides ── */
[data-testid="stHeader"]          { background: #0b0f19; border-bottom: 1px solid #1e293b; }
[data-testid="stToolbar"]         { background: #0b0f19; }
[data-testid="stMainBlockContainer"] { background: #0b0f19; }
h1,h2,h3,h4                      { color: #f8fafc !important; }
.stCaption, .st-emotion-cache-16idsys p { color: #94a3b8 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"]         { background: #111827 !important; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label   { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2      { color: #f8fafc !important; }

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #334155 !important;
    color: #fff !important;
    border-color: #475569 !important;
}

/* Selectbox — trigger box */
[data-testid="stSelectbox"] label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f8fafc !important;
    border-radius: 8px !important;
}
/* Selected value text inside the trigger */
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] div {
    color: #f8fafc !important;
}

/* Selectbox — dropdown popup list */
[data-baseweb="popover"] > div,
[data-baseweb="menu"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

/* Tooltip / help popup styling */
div[data-baseweb="tooltip"],
div[role="tooltip"],
div[data-testid="stTooltipContent"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-baseweb="tooltip"] div,
div[role="tooltip"] div,
div[data-testid="stTooltipContent"] div {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
div[data-baseweb="tooltip"] p,
div[data-baseweb="tooltip"] span,
div[role="tooltip"] p,
div[role="tooltip"] span,
div[data-testid="stTooltipContent"] p,
div[data-testid="stTooltipContent"] span {
    color: #ffffff !important;
}
/* Each option row */
[data-baseweb="menu"] li,
[role="option"] {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    font-size: 0.88rem !important;
}
/* Hover */
[data-baseweb="menu"] li:hover,
[role="option"]:hover {
    background: #334155 !important;
    color: #ffffff !important;
    cursor: pointer;
}
/* Selected highlight */
[role="option"][aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] {
    background: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #64748b !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #151e2e;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"]  { color: #94a3b8 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.07em; }
[data-testid="stMetricValue"]  { color: #38bdf8 !important; font-size: 1.5rem !important; font-weight: 700 !important; }

/* ── Progress bars ── */
[data-testid="stProgressBar"] > div { background: #1e293b !important; border-radius: 99px; }
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #2563eb, #38bdf8) !important; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #2563eb, #38bdf8) !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #151e2e !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}
/* Summary row */
[data-testid="stExpander"] > details > summary {
    background: #1e293b !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    color: #f8fafc !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em;
}
[data-testid="stExpander"] > details > summary:hover {
    background: #334155 !important;
    color: #fff !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: #f8fafc !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary svg { stroke: #38bdf8 !important; fill: #38bdf8 !important; }

/* ── Divider ── */
hr { border-color: #1e293b !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #2563eb !important; }

/* ── Main content area buttons (Load more, etc.) ── */
[data-testid="stMainBlockContainer"] .stButton > button,
.stButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 9px 16px !important;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
    border-color: #60a5fa !important;
    color: #ffffff !important;
    transform: none;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
}
/* Preserve sidebar button override */
[data-testid="stSidebar"] .stButton > button {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #334155 !important;
    color: #fff !important;
    border-color: #475569 !important;
    transform: none;
    box-shadow: none !important;
}

/* ─────────────────────────────────────────────
   CUSTOM COMPONENT CLASSES
───────────────────────────────────────────── */

/* Section headers */
.section-header {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #38bdf8;
    padding: 8px 0 6px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Game card */
.game-card {
    background: #151e2e;
    border-radius: 10px;
    padding: 16px 18px;
    margin: 10px 0;
    border: 1px solid #1e293b;
    border-left: 3px solid #2563eb;
    box-shadow: none;
    transition: border-color 0.15s ease;
    position: relative;
    overflow: hidden;
}
.game-card:hover {
    border-color: #3b82f6;
}
.card-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 6px;
    line-height: 1.3;
}
.card-meta {
    font-size: 0.8rem;
    margin-bottom: 10px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
}
.badge {
    border-radius: 6px;
    padding: 2px 9px;
    font-size: 0.73rem;
    font-weight: 500;
    white-space: nowrap;
    border: 1px solid;
}
.badge-genre  { background: #1e293b; border-color: #334155; color: #cbd5e1; }
.badge-price  { background: #064e3b; border-color: #059669; color: #34d399; }
.badge-rating { background: #451a03; border-color: #d97706; color: #fbbf24; }
.card-pitch   { font-size: 0.84rem; color: #94a3b8; line-height: 1.55; }
.rank-pill {
    position: absolute;
    top: 14px; right: 16px;
    background: #2563eb;
    color: #fff;
    font-size: 0.7rem;
    font-weight: 700;
    border-radius: 6px;
    padding: 2px 9px;
    letter-spacing: 0.04em;
}

/* Chat bubbles */
.user-bubble {
    background: #2563eb;
    color: #fff;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    max-width: 84%;
    margin-left: auto;
    font-size: 0.9rem;
    font-weight: 500;
    line-height: 1.5;
}
.agent-bubble {
    background: #151e2e;
    color: #e2e8f0;
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 0;
    max-width: 84%;
    font-size: 0.88rem;
    border: 1px solid #1e293b;
    line-height: 1.55;
}

/* Reasoning box */
.reasoning-box {
    background: #151e2e;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.6;
    margin-bottom: 14px;
}

/* Execution log */
.log-graph  { color: #38bdf8; font-weight: 600; }
.log-data   { color: #34d399; }
.log-warn   { color: #fbbf24; }
.log-info   { color: #94a3b8; }

/* Stars */
.stars { color: #fbbf24; letter-spacing: 2px; }

/* Portal Cards */
.portal-card {
    background: #151e2e;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 24px;
    text-align: left;
    transition: border-color 0.15s ease;
    margin-bottom: 12px;
}
.portal-card:hover {
    border-color: #2563eb;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

def _init_session() -> None:
    defaults = {
        "messages":       [],   # [{role, content}]
        "last_state":     None, # AgentState from latest run
        "running":        False,
        "cards_shown":    4,    # how many recommendation cards are visible
        "custom_personas": {},  # name -> user_id for user-created profiles
        "current_page":   "home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

GENRE_ICONS = {
    # Games
    "Action": "", "Adventure": "", "RPG": "", "Strategy": "",
    "Simulation": "", "Sports": "", "Racing": "", "Horror": "",
    "Puzzle": "", "Platformer": "", "Fighting": "", "Shooter": "",
    "MMORPG": "", "Roguelike": "", "Sandbox": "", "Stealth": "",
    "Survival": "", "Visual Novel": "", "Rhythm": "", "Card Game": "",
    # Movie genres
    "Sci-Fi": "", "Animation": "", "Romance": "", "Comedy": "",
    "Fantasy": "", "Thriller": "", "Drama": "",
}

def _genre_icon(genre: str) -> str:
    return GENRE_ICONS.get(genre, "")

def _stars(rating: float) -> str:
    full  = int(round(rating))
    empty = 5 - full
    return "★" * full + "☆" * empty

def _price_str(price: float) -> str:
    return "Free" if price == 0.0 else f"${price:.2f}"

def _render_card(item: dict, rank: int, media_type: str = "games") -> str:
    is_movie = (media_type == "movies")
    price = "Free" if item["price"] == 0.0 else (f"Rent ${item['price']:.2f}" if is_movie else f"${item['price']:.2f}")
    stars = _stars(item["rating"])
    tags  = ", ".join(item.get("tags", [])[:3])
    
    if is_movie:
        pitch = item.get("overview", "")
        if pitch:
            if len(pitch) > 150:
                pitch = pitch[:147] + "..."
        else:
            pitch = f"A {item['genre'].lower()} movie featuring {tags}."
    else:
        pitch = f"A {item['genre'].lower()} experience"
        if tags:
            pitch += f" featuring {tags}"
        pitch += f". {'High-value pick' if item['price'] == 0 else 'Well-reviewed title'} with strong player sentiment."
        
    return f"""
<div class="game-card">
  <span class="rank-pill">#{rank}</span>
  <div class="card-title">{item['title']}</div>
  <div class="card-meta">
    <span class="badge badge-genre">{item['genre']}</span>
    <span class="badge badge-price">{price}</span>
    <span class="badge badge-rating"><span class="stars">{stars}</span> {item['rating']}</span>
  </div>
  <div class="card-pitch">
    {pitch}
  </div>
</div>"""

def _render_log_line(line: str) -> str:
    line_esc = line.replace("<", "&lt;").replace(">", "&gt;")
    if "[GRAPH]" in line:
        return f'<span class="log-graph">{line_esc}</span>'
    if "Model returned" in line or "Filtered" in line or "Reranked" in line:
        return f'<span class="log-data">{line_esc}</span>'
    if "Relaxation" in line or "GUARD" in line or "fallback" in line:
        return f'<span class="log-warn">{line_esc}</span>'
    return f'<span class="log-info">{line_esc}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Personalization Engine")
    st.markdown("*Clickstream Analytics & Recommender System*")
    st.divider()
    
    # Back to home button
    if st.session_state["current_page"] != "home":
        if st.button("Back to Home", use_container_width=True, key="back_to_home_btn"):
            st.session_state["current_page"] = "home"
            st.session_state["messages"] = []
            st.session_state["last_state"] = None
            st.rerun()
        st.divider()

    # ── User Profile ──────────────────────────────────────────────────────────
    st.markdown("**User Profile**")
    user_name = st.text_input(
        "User Profile Name",
        value="",
        placeholder="Enter your name (e.g. Rahul, Priya)...",
        key="user_profile_name_input",
        label_visibility="collapsed",
    )
    clean_name = user_name.strip() if user_name.strip() else "Guest User"
    user_id = 1 + (abs(hash(clean_name.lower())) % 500)
    st.caption(f"Active Profile ID: `User #{user_id}` · `{clean_name}`")

    # Only show these options if not on Home Page
    if st.session_state["current_page"] != "home":
        st.divider()
        st.markdown("**Quick Queries**")
        if st.session_state["current_page"] == "movies":
            quick_queries = [
                "Sci-Fi movies with high ratings",
                "Free horror movies",
                "Action movies under 4 dollars",
                "Thriller movies with rating above 4",
                "Popular romance movies",
            ]
        else:
            quick_queries = [
                "Games with high ratings",
                "Free horror games",
                "Multiplayer action games",
                "Puzzle games",
                "Adventure games", 
            ]
        for q in quick_queries:
            if st.button(q, use_container_width=True, key=f"quick_{q}"):
                st.session_state["prefill_query"] = q

        st.divider()
        st.markdown(f"**Analytics & Model Config**")
        st.caption(f"Clickstream events: `58,508` · Baseline CTR: `56.98%`")
        if st.session_state["current_page"] == "movies":
            st.caption("Database: `recommender.db` · Model: `PopularityBaseline`")
        else:
            st.caption("Database: `recommender.db` · Model: `GRUSeqRec`")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state["messages"]   = []
            st.session_state["last_state"] = None
            st.rerun()

    # ── Developer Profile ───────────────────────────────────────────────────
    st.divider()
    st.markdown("**Developer Profile**")
    
    # Load developer configuration dynamically to allow user-level editing and bypass filters
    import json
    try:
        with open("developer_config.json", "r", encoding="utf-8") as f:
            dev_config = json.load(f)
    except Exception:
        dev_config = {
            "name": "Ashwini Kumar",
            "role": "Data Analyst | AI & Data Science Engineer",
            "github": "https://github.com/ashwinikr295",
            "linkedin": "https://www.linkedin.com/in/ashwini-kumar-6928a527a/"
        }

    st.markdown(
        f"""
        <div style="background: #151e2e; 
                    padding: 14px; border-radius: 10px; border: 1px solid #1e293b; 
                    font-family: 'Inter', sans-serif;
                    margin-top: 5px;">
          <div style="font-weight: 700; color: #f8fafc; font-size: 0.95rem; margin-bottom: 2px;">{dev_config.get('name', 'Ashwini Kumar')}</div>
          <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 500; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em;">{dev_config.get('role', 'Data Analyst | AI & Data Science Engineer')}</div>
          <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.85rem;">
            <a href="{dev_config.get('github', 'https://github.com/ashwinikr295')}" target="_blank" style="color: #cbd5e1; text-decoration: none; display: flex; align-items: center; gap: 8px;">
              <span style="border-bottom: 1px dashed #3b82f6;">GitHub Portfolio</span>
            </a>
            <a href="{dev_config.get('linkedin', 'https://www.linkedin.com/in/ashwini-kumar-6928a527a/')}" target="_blank" style="color: #cbd5e1; text-decoration: none; display: flex; align-items: center; gap: 8px;">
              <span style="border-bottom: 1px dashed #3b82f6;">LinkedIn Profile</span>
            </a>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT  — three columns / sections
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT  — Routing between Home and Portals
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state["current_page"] == "home":
    st.markdown("""
    <div style="padding: 24px 0 24px;">
      <h1 style="font-size: 2rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">
        Recommendation Portals
      </h1>
      <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 24px;">
        Select an engine below to start exploring personalized recommendations.
      </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("""
        <div class="portal-card">
          <h3 style="margin: 0 0 8px 0; color: #f1f5f9; font-weight: 600; font-size: 1.25rem;">Games Portal</h3>
          <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5; margin-bottom: 16px;">
            Personalized video game recommendations trained on user interaction patterns and live genre interests.
          </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Games Portal", key="launch_games", use_container_width=True):
            st.session_state["current_page"] = "games"
            st.session_state["messages"] = []
            st.session_state["last_state"] = None
            st.rerun()
            
    with col2:
        st.markdown("""
        <div class="portal-card">
          <h3 style="margin: 0 0 8px 0; color: #f1f5f9; font-weight: 600; font-size: 1.25rem;">Movies Portal</h3>
          <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.5; margin-bottom: 16px;">
            Search and filter a catalog of 8,000+ movies by genre, rating, and rental price.
          </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Movies Portal", key="launch_movies", use_container_width=True):
            st.session_state["current_page"] = "movies"
            st.session_state["messages"] = []
            st.session_state["last_state"] = None
            st.rerun()
else:
    active_page = st.session_state["current_page"]
    is_movie = (active_page == "movies")
    media_lbl = "movie" if is_movie else "game"
    
    col_chat, col_recs = st.columns([1, 1], gap="large")
    
    # ── SECTION 1 — CHAT INTERFACE ──────────────────────────────────────────
    with col_chat:
        portal_title = "Movie Recommender Chat" if is_movie else "Game Recommender Chat"
        st.markdown(f'<div class="section-header">{portal_title}</div>',
                    unsafe_allow_html=True)
    
        # Conversation history
        chat_box = st.container(height=480)
        with chat_box:
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-bubble">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="agent-bubble">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
    
        # Prefill from quick-query buttons
        prefill = st.session_state.pop("prefill_query", "")
    
        chat_placeholder = 'e.g. "Recommend a sci-fi movie under $4"' if is_movie else 'e.g. "Recommend a classic RPG under $15"'
        query = st.chat_input(
            placeholder=chat_placeholder,
            key="chat_input",
        )
        if prefill and not query:
            query = prefill
    
        # ── Run agent on new query ──────────────────────────────────────────────
        if query:
            st.session_state["messages"].append({"role": "user", "content": query})
    
            with st.spinner("Agent thinking..."):
                try:
                    state: AgentState = run_agent(user_id=user_id, query=query, media_type=active_page)
                    st.session_state["last_state"] = state
    
                    # ── Build a dynamic, query-specific reply ──────────────────
                    n      = len(state.reranked_candidates)
                    genre  = state.parsed_intent.get("genre", "")
                    price  = state.parsed_intent.get("max_price")
                    rating = state.parsed_intent.get("min_rating")
    
                    if n == 0:
                        reply = f"I couldn't find any {media_lbl}s matching that query. Try adjusting the genre or removing price/rating filters."
                    else:
                        top_titles  = [it["title"] for it in state.reranked_candidates[:3]]
                        genres_seen = list(dict.fromkeys(
                            it["genre"] for it in state.reranked_candidates
                        ))[:3]
                        avg_rating  = sum(it["rating"] for it in state.reranked_candidates) / n
    
                        # Opening line — vary based on what was found
                        if genre:
                            opener = f"Found **{n} {genre} {media_lbl}{'s' if n != 1 else ''}**"
                        else:
                            opener = f"Found **{n} {media_lbl}{'s' if n != 1 else ''}** across {', '.join(genres_seen)}"
    
                        # Constraints summary
                        constraints = []
                        if price is not None:
                            constraints.append(f"priced ≤ ${price:.0f}" if not is_movie else f"rent price ≤ ${price:.0f}")
                        if rating:
                            constraints.append(f"rated ≥ {rating}")
                        constraint_str = " and ".join(constraints)
    
                        # Top picks sentence
                        if len(top_titles) == 1:
                            picks = f"Top pick: **{top_titles[0]}**"
                        elif len(top_titles) == 2:
                            picks = f"Top picks: **{top_titles[0]}** and **{top_titles[1]}**"
                        else:
                            picks = f"Top picks: **{top_titles[0]}**, **{top_titles[1]}**, and **{top_titles[2]}**"
    
                        reply = opener
                        if constraint_str:
                            reply += f" ({constraint_str})"
                        reply += f". {picks}. Avg rating: {avg_rating:.1f}. Check the right panel for details!"
    
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": reply}
                    )
                    # Reset pagination for new query
                    st.session_state["cards_shown"] = 4
                except Exception as exc:
                    st.session_state["messages"].append(
                        {"role": "assistant",
                         "content": f"Error: {exc}"}
                    )
    
            st.rerun()
    
    # ── SECTION 2 — DYNAMIC PAGE LAYOUT (recommendations) ─────────────────────
    with col_recs:
        st.markdown('<div class="section-header">Dynamic Page Layout</div>',
                    unsafe_allow_html=True)
    
        state: AgentState | None = st.session_state.get("last_state")
    
        if state is None:
            st.markdown(f"""
            <div style="text-align:center;padding:60px 20px;color:#3d3d6b">
              <div style="font-size:1rem;margin-top:12px">
                Ask for a {media_lbl} recommendation to see personalized results here.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Reasoning box
            if state.reasoning:
                st.markdown(
                    f'<div class="reasoning-box">{state.reasoning}</div>',
                    unsafe_allow_html=True,
                )
    
            # Intent chips
            intent = state.parsed_intent
            chips = []
            if intent.get("genre"):
                chips.append(f"{intent['genre']}")
            if intent.get("max_price") is not None:
                chips.append(f"≤ ${intent['max_price']:.0f}")
            if intent.get("min_rating"):
                chips.append(f"≥ {intent['min_rating']}")
            if intent.get("tags"):
                chips += [f"{t}" for t in intent["tags"]]
    
            if chips:
                chip_html = " ".join(
                    f'<span class="badge" style="margin:2px">{c}</span>' for c in chips
                )
                st.markdown(chip_html, unsafe_allow_html=True)
                st.markdown("")
    
            # cards with pagination
            if state.reranked_candidates:
                n_show  = st.session_state.get("cards_shown", 4)
                total   = len(state.reranked_candidates)
                visible = state.reranked_candidates[:n_show]
    
                cards_html = "".join(
                    _render_card(item, i + 1, media_type=active_page)
                    for i, item in enumerate(visible)
                )
                st.markdown(cards_html, unsafe_allow_html=True)
    
                remaining = total - n_show
                if remaining > 0:
                    btn_label = f"Load {min(remaining, 6)} more  ({remaining} remaining)"
                    if st.button(btn_label, key="load_more", use_container_width=True):
                        st.session_state["cards_shown"] = n_show + 6
                        st.rerun()
                else:
                    st.caption(f"All {total} results shown.")
            else:
                st.info(f"No matching {media_lbl}s found. Try a broader query.")
    
            # Stats footer
            st.caption(
                f"Steps: {state.step_count} / {MAX_STEPS} · "
                f"Candidates: {len(state.raw_candidates)} raw → "
                f"{len(state.filtered_candidates)} filtered → "
                f"{len(state.reranked_candidates)} shown"
            )
    
    # ── SECTION 3 — AGENT MIND DASHBOARD ──────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header"> Agent Mind Dashboard</div>',
                unsafe_allow_html=True)
    
    state = st.session_state.get("last_state")
    if state is None:
        st.caption("No agent run yet. Submit a query above.")
    else:
        dash_col1, dash_col2, dash_col3, dash_col4 = st.columns(4)
        dash_col1.metric("Steps taken",   f"{state.step_count} / {MAX_STEPS}")
        dash_col2.metric("Intent genre",  state.parsed_intent.get("genre", "—"))
        dash_col3.metric("Candidates",    len(state.reranked_candidates))
        beliefs = state.memory_snapshot.get("top_beliefs", [])
        dash_col4.metric("Active beliefs", len(beliefs))
    
        # Memory belief bars & Chain-of-thought log
        if beliefs:
            mem_col, log_col = st.columns([1, 2])
            with mem_col:
                with st.expander(" Belief State (top 5)", expanded=True):
                    for b in beliefs[:5]:
                        label = b["key"].replace("genre:", "")
                        val   = float(b["strength"])
                        st.progress(
                            val,
                            text=f"{label}  ({val:.2f}, {b['evidence']} signals)",
                        )
            with log_col:
                with st.expander(" Chain-of-Thought Execution Log", expanded=True):
                    if state.logs:
                        log_lines = "\n".join(_render_log_line(l) for l in state.logs)
                        st.markdown(
                            f'<pre style="background:#0a0a18;border-radius:10px;'
                            f'padding:14px;font-size:0.78rem;line-height:1.6;'
                            f'overflow-x:auto;border:1px solid #2d2d4e;'
                            f'font-family:\'JetBrains Mono\',monospace">'
                            f'{log_lines}</pre>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("No logs captured.")
        else:
            with st.expander(" Chain-of-Thought Execution Log", expanded=True):
                if state.logs:
                    log_lines = "\n".join(_render_log_line(l) for l in state.logs)
                    st.markdown(
                        f'<pre style="background:#0a0a18;border-radius:10px;'
                        f'padding:14px;font-size:0.78rem;line-height:1.6;'
                        f'overflow-x:auto;border:1px solid #2d2d4e;'
                        f'font-family:\'JetBrains Mono\',monospace">'
                        f'{log_lines}</pre>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No logs captured.")
    
        # User profile
        profile_text = state.memory_snapshot.get("profile", "")
        if profile_text:
            with st.expander(" Long-Term User Profile"):
                st.markdown(
                    f'<div class="reasoning-box">{profile_text}</div>',
                    unsafe_allow_html=True,
                )