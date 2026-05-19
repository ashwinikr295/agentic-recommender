"""
import_movies.py
================
Parses movies.csv, auto-tags movies with genres/tags based on overview,
and populates the SQLite `movies` table in `recommender.db`.
"""

import os
import sys
import sqlite3
import pandas as pd
import json
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = "recommender.db"
CSV_PATH = "movies.csv"

GENRE_KEYWORDS = {
    "Sci-Fi": ["sci-fi", "science fiction", "space", "alien", "future", "futuristic", "robot", "cyborg", "galaxy", "universe", "time travel", "portal", "cloning", "cyberpunk", "hologram", "spaceship", "technology", "artificial intelligence", "ai"],
    "Horror": ["horror", "scary", "ghost", "haunted", "werewolf", "vampire", "blood", "zombie", "gore", "creature", "terror", "spooky", "evil", "monster", "demon", "slaughter", "paranormal", "witch", "curse"],
    "Animation": ["animation", "animated", "anime", "cartoon", "toy", "disney", "pixar", "dreamworks", "voice of", "family-friendly", "illustration", "stop-motion"],
    "Romance": ["romance", "romantic", "love", "date", "marriage", "wedding", "kiss", "heart", "boyfriend", "girlfriend", "affair", "passion", "couple"],
    "Comedy": ["comedy", "funny", "laugh", "hilarious", "humor", "joke", "sitcom", "parody", "satire", "fun", "amusing", "silly", "goofy"],
    "Action": ["action", "fight", "battle", "war", "soldier", "army", "agent", "assassin", "cop", "police", "chase", "weapon", "gun", "combat", "superhero", "martial arts", "martial", "rescue", "hostage", "armored", "covert", "mission", "explosive", "criminal", "smuggle", "heist"],
    "Adventure": ["adventure", "island", "journey", "explore", "discover", "jungle", "expedition", "treasure", "quest", "travel", "map", "ruins", "wilderness", "navigation", "voyage", "ocean", "shipwreck"],
    "Fantasy": ["fantasy", "magic", "sword", "witchcraft", "wizard", "gods", "mythology", "kingdom", "elven", "fairy", "spell", "dragon", "legendary", "supernatural", "realm", "curse"],
    "Thriller": ["thriller", "mystery", "crime", "suspense", "detective", "investigate", "murder", "killer", "serial killer", "spy", "conspiracy", "kidnap", "psychological", "puzzle", "homicide", "theft", "stolen"],
}

TAG_WORDS = {
    "space": ["space", "galaxy", "stars", "planet", "orbit", "mars"],
    "superhero": ["superhero", "marvel", "dc", "comics", "powers", "vigilante"],
    "monsters": ["monster", "creature", "beast", "dinosaur", "kaiju"],
    "survival": ["survival", "survive", "wilderness", "stranded", "stranded", "escaped"],
    "post-apocalyptic": ["apocalypse", "post-apocalyptic", "dystopian", "ruins", "wasteland"],
    "heist": ["heist", "robbery", "theft", "steal", "armored", "bank"],
    "family": ["family", "kids", "children", "wholesome"],
    "history": ["history", "historical", "ancient", "biography", "true story"],
    "paranormal": ["ghost", "paranormal", "haunted", "demon", "spirit", "exorcism"],
    "combat": ["fight", "martial arts", "kung fu", "battle", "clash"],
}

def determine_genre_and_tags(title, overview):
    text = f"{title} {overview}".lower()
    
    # Calculate score for each genre
    genre_scores = {}
    for genre, keywords in GENRE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            # Word boundary matching
            matches = len(re.findall(r'\b' + re.escape(kw) + r'\b', text))
            score += matches
        if score > 0:
            genre_scores[genre] = score
            
    # Select primary genre
    if genre_scores:
        primary_genre = max(genre_scores, key=genre_scores.get)
    else:
        primary_genre = "Drama"  # Default fallback
        
    # Extract tags
    tags = []
    # Add matched genre names as tags
    for genre, score in genre_scores.items():
        tags.append(genre.lower())
    # Add matched tag keywords
    for tag_name, keywords in TAG_WORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                tags.append(tag_name)
                break
                
    # Deduplicate and limit tags to 5
    tags = list(set(tags))[:5]
    if not tags:
        tags = ["movie", "cinema"]
        
    return primary_genre, tags

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found in current directory.")
        sys.exit(1)
        
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    
    # Fill NA
    df["overview"] = df["overview"].fillna("")
    df["release_date"] = df["release_date"].fillna("2000-01-01")
    
    # Deduplicate titles by appending year if duplicate title exists
    print("Deduplicating titles by appending year...")
    # Calculate year for each
    years = df["release_date"].apply(lambda d: str(d).split("-")[0] if isinstance(d, str) else "2000")
    df["year"] = years
    
    # Identify duplicate titles
    dup_titles = df[df.duplicated(subset=["title"], keep=False)]["title"].unique()
    print(f"Found {len(dup_titles)} duplicate title groups. Appending years to resolve.")
    
    def process_title(row):
        title = row["title"]
        if title in dup_titles:
            return f"{title} ({row['year']})"
        return title
        
    df["processed_title"] = df.apply(process_title, axis=1)
    
    # Double check final title uniqueness
    df = df.drop_duplicates(subset=["processed_title"], keep="first")
    
    # Process Genres and Tags
    print("Auto-tagging genres and tags...")
    items = []
    for idx, row in df.iterrows():
        genre, tags = determine_genre_and_tags(row["processed_title"], row["overview"])
        
        # Rating 0-10 to 1-5 scale (vote_average / 2)
        rating = round(max(1.0, min(5.0, float(row["vote_average"]) / 2.0)), 1)
        
        # Price proxy - let's make it standard $3.99 rental, or free if very old
        year_int = 2000
        try:
            year_int = int(row["year"])
        except:
            pass
        price = 0.0 if year_int < 2010 else 3.99
        
        items.append({
            "movie_id": int(row["id"]),
            "title": row["processed_title"],
            "genre": genre,
            "tags": json.dumps(tags),
            "price": price,
            "rating": rating,
            "release_date": row["release_date"],
            "popularity": float(row["popularity"]),
            "vote_count": int(row["vote_count"]),
            "overview": row["overview"]
        })
        
    print(f"Processed {len(items)} movies. Connecting to database {DB_PATH}...")
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    
    print("Creating table `movies`...")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS movies (
        movie_id     INTEGER PRIMARY KEY,
        title        TEXT    NOT NULL UNIQUE,
        genre        TEXT    NOT NULL,
        tags         TEXT    NOT NULL,   -- JSON array string
        price        REAL    NOT NULL DEFAULT 3.99,
        rating       REAL    NOT NULL,   -- 1.0 - 5.0
        release_date TEXT    NOT NULL,
        popularity   REAL    NOT NULL,
        vote_count   INTEGER NOT NULL,
        overview     TEXT
    );
    """)
    
    cur.execute("CREATE INDEX IF NOT EXISTS idx_movies_genre ON movies(genre);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_movies_popularity ON movies(popularity DESC);")
    
    # Clear existing movies if any
    cur.execute("DELETE FROM movies;")
    
    print("Inserting movie records...")
    cur.executemany("""
    INSERT INTO movies 
        (movie_id, title, genre, tags, price, rating, release_date, popularity, vote_count, overview)
    VALUES
        (:movie_id, :title, :genre, :tags, :price, :rating, :release_date, :popularity, :vote_count, :overview)
    """, items)
    
    con.commit()
    con.close()
    print(f"[OK] Successfully imported {len(items)} movies into table `movies` in `recommender.db`!")

if __name__ == "__main__":
    main()
