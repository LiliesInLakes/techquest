"""
build_movie_db.py
=================
Builds movie_features.db — a clean, high-signal SQLite feature store that
fuses MovieLens 32M user data with TMDB enrichment.

Schema
------
  movies          — one row per movie; identity + raw source fields + canonical content_text
  movie_user_tags — normalized (movielens_id, tag, tag_count); ALL user tags preserved

Canonical derived text field
-----------------------------
  content_text — shared, clean natural-language movie representation suitable as a baseline
                 input for both lexical TF-IDF retrieval and dense semantic embedding.
                 Structure:
                   Genres: <genres>.
                   Keywords: <keywords>.
                   User tags: <top-50 tags by descending frequency>.
                   Overview: <plot synopsis>

  Preserves multi-word phrases (e.g. "time travel", "black hole", "space exploration").
  Natural language prose throughout.

NOT included in content_text
-----------------------------
  tagline, cast, directors, ratings, runtime, year, language, popularity, IDs,
  generated/LLM-created labels.

Usage
-----
    # Full run with default data path (data/ml-32m/)
    python build_movie_db.py

    # Custom dataset directory
    python build_movie_db.py --data-dir path/to/ml-32m

    # Dry-run: MovieLens data only, no TMDB API calls
    python build_movie_db.py --dry-run

    # Limit to first N movies (for quick testing with real API calls)
    python build_movie_db.py --limit 200

    # Fresh rebuild from scratch
    python build_movie_db.py --reset

    # Adjust top-N tags per movie (default: 50, ordered by tag_count desc)
    python build_movie_db.py --top-n-tags 50

Environment
-----------
    TMDB_API_KEY       — read from .env via python-dotenv, or from environment.
                         If not set, TMDB enrichment is skipped (MovieLens data only).
    MOVIELENS_DATA_DIR — optional custom path for MovieLens 32M CSV directory.

Output
------
    movie_features.db   — SQLite database (project root)
    build_progress.json — checkpoint for resuming interrupted runs
"""

import os
import sys
import json
import time
import asyncio
import logging
import sqlite3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required.  pip install pandas", file=sys.stderr)
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.resolve()
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ml-32m"
DB_PATH       = PROJECT_ROOT / "movie_features.db"
CHECKPOINT    = PROJECT_ROOT / "build_progress.json"

# ── TMDB ───────────────────────────────────────────────────────────────────────
TMDB_BASE        = "https://api.themoviedb.org/3"
TMDB_CONCURRENCY = 40
TMDB_RETRY_MAX   = 3
CHECKPOINT_EVERY = 500

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MOVIELENS LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_movielens(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge MovieLens CSVs from the specified data directory.

    Requires:
        movies.csv, links.csv, ratings.csv, tags.csv

    Returns a tuple of (df, tag_freq):
        df: DataFrame with columns:
            movielens_id, title, genres, avg_rating, rating_count,
            user_tags_with_counts, tmdb_id
        tag_freq: DataFrame with columns:
            movielens_id, tag, tag_count (ALL unique pairs preserved)
    """
    required_files = ["movies.csv", "links.csv", "ratings.csv", "tags.csv"]
    missing = [f for f in required_files if not (data_dir / f).is_file()]

    if missing:
        log.error(
            f"\n{'=' * 68}\n"
            f"ERROR: MovieLens 32M dataset not found at: {data_dir.resolve()}\n"
            f"Missing required file(s): {', '.join(missing)}\n\n"
            f"Setup instructions:\n"
            f"  1. Download MovieLens 32M from: https://grouplens.org/datasets/movielens/32m/\n"
            f"  2. Extract the CSV files into: {data_dir.resolve()}\n"
            f"  3. Required CSVs: {', '.join(required_files)}\n"
            f"  Or specify a custom location with: --data-dir <path>\n"
            f"{'=' * 68}"
        )
        sys.exit(1)

    log.info(f"Loading MovieLens files from: {data_dir} ...")

    # ── movies ──────────────────────────────────────────────────────────────
    movies = pd.read_csv(
        data_dir / "movies.csv",
        dtype={"movieId": int, "title": str, "genres": str},
    )
    movies.rename(columns={"movieId": "movielens_id"}, inplace=True)
    movies["genres"] = movies["genres"].fillna("").str.replace(
        "(no genres listed)", "", regex=False
    )

    # ── links ───────────────────────────────────────────────────────────────
    links = pd.read_csv(data_dir / "links.csv", dtype={"movieId": int})
    links.rename(columns={"movieId": "movielens_id"}, inplace=True)
    links["tmdb_id"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    links = links[["movielens_id", "tmdb_id"]]

    # ── ratings ─────────────────────────────────────────────────────────────
    log.info("Aggregating ratings (large file, ~30 s) ...")
    rating_chunks = []
    for chunk in pd.read_csv(
        data_dir / "ratings.csv",
        usecols=["movieId", "rating"],
        dtype={"movieId": int, "rating": float},
        chunksize=2_000_000,
    ):
        rating_chunks.append(
            chunk.groupby("movieId")["rating"].agg(["sum", "count"])
        )
    ratings_agg = pd.concat(rating_chunks).groupby(level=0).sum()
    ratings_agg["avg_rating"]   = (ratings_agg["sum"] / ratings_agg["count"]).round(4)
    ratings_agg["rating_count"] = ratings_agg["count"].astype(int)
    ratings_agg = ratings_agg[["avg_rating", "rating_count"]].reset_index()
    ratings_agg.rename(columns={"movieId": "movielens_id"}, inplace=True)

    # ── tags: compute (movieId, tag, count) ─────────────────────────────────
    log.info("Aggregating user tags with frequency counts ...")
    tags_raw = pd.read_csv(
        data_dir / "tags.csv",
        usecols=["movieId", "tag"],
        dtype={"movieId": int, "tag": str},
    )
    tags_raw.dropna(subset=["tag"], inplace=True)
    tags_raw["tag"] = tags_raw["tag"].str.strip().str.lower()
    tags_raw = tags_raw[tags_raw["tag"] != ""]

    # Compute frequency per (movie, tag) pair — preserves ALL unique pairs
    tag_freq = (
        tags_raw.groupby(["movieId", "tag"])
        .size()
        .reset_index(name="tag_count")
    )
    tag_freq.rename(columns={"movieId": "movielens_id"}, inplace=True)

    # Build per-movie: list of (tag, count) sorted by count desc
    tag_by_movie = (
        tag_freq.sort_values(["movielens_id", "tag_count"], ascending=[True, False])
        .groupby("movielens_id")
        .apply(lambda g: list(zip(g["tag"], g["tag_count"])), include_groups=False)
        .reset_index()
    )
    tag_by_movie.columns = ["movielens_id", "user_tags_with_counts"]

    # ── merge ────────────────────────────────────────────────────────────────
    log.info("Merging tables ...")
    df = movies.merge(links, on="movielens_id", how="left")
    df = df.merge(ratings_agg, on="movielens_id", how="left")
    df = df.merge(tag_by_movie, on="movielens_id", how="left")

    df["user_tags_with_counts"] = df["user_tags_with_counts"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    log.info(f"MovieLens merged: {len(df):,} movies")
    return df, tag_freq


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TAG SORTING & CANONICAL CONTENT_TEXT CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_top_n_tags(user_tags_with_counts: List[tuple], top_n: int) -> List[str]:
    """
    Explicitly sort user tag tuples by tag_count descending (with tag string tie-break)
    and return the top-N cleaned tag strings.
    """
    if not user_tags_with_counts:
        return []

    sorted_pairs = sorted(
        user_tags_with_counts,
        key=lambda x: (int(x[1]) if len(x) > 1 and x[1] is not None else 1, str(x[0])),
        reverse=True,
    )
    return [str(tag).strip().lower() for tag, _ in sorted_pairs[:top_n] if str(tag).strip()]


def build_content_text(
    genres: str,
    tmdb_keywords: List[str],
    user_tags_with_counts: List[tuple],  # [(tag, count), ...]
    overview: str,
    top_n: int = 50,
) -> Optional[str]:
    """
    Build canonical content representation for BOTH lexical search (CountVectorizer/TF-IDF)
    and semantic embeddings (sentence-transformers).

    Sources (in order):
      1. Genres: "Genres: action, sci-fi, thriller."
      2. TMDB keywords: "Keywords: spacecraft, black hole, time travel."
      3. MovieLens user tags: "User tags: space, emotional, mind-bending, epic." (top-N by tag_count desc)
      4. TMDB overview: "Overview: The adventures of a group of explorers..."

    Preserves multi-word phrases and natural prose throughout.
    Returns None if no content is available.
    """
    parts: List[str] = []

    # 1. Genres
    if genres and genres.strip():
        genre_list = [g.strip().lower() for g in genres.split("|") if g.strip()]
        if genre_list:
            parts.append(f"Genres: {', '.join(genre_list)}.")

    # 2. TMDB keywords
    if tmdb_keywords:
        kw_list = [kw.strip().lower() for kw in tmdb_keywords if kw.strip()]
        if kw_list:
            parts.append(f"Keywords: {', '.join(kw_list)}.")

    # 3. Top-N user tags by descending frequency
    top_tags = extract_top_n_tags(user_tags_with_counts, top_n)
    if top_tags:
        parts.append(f"User tags: {', '.join(top_tags)}.")

    # 4. TMDB overview
    if overview and overview.strip():
        parts.append(f"Overview: {overview.strip()}")

    result = " ".join(parts).strip()
    return result if result else None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SQLITE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

DDL = """
CREATE TABLE IF NOT EXISTS movies (
    movielens_id        INTEGER PRIMARY KEY,
    tmdb_id             INTEGER,
    title               TEXT,
    genres              TEXT,           -- pipe-separated: "Action|Sci-Fi"
    avg_rating          REAL,           -- MovieLens mean rating (NULL if unrated)
    rating_count        INTEGER,        -- MovieLens number of ratings (NULL if unrated)
    overview            TEXT,           -- TMDB plot summary (NULL if unavailable)
    tmdb_keywords       TEXT,           -- JSON array of keyword strings
    movielens_user_tags TEXT,           -- JSON array: top-N tags by frequency (convenience)
    content_text        TEXT,           -- canonical derived natural-language text
    tmdb_fetched        INTEGER DEFAULT 0  -- 0=pending, 1=success, 2=skipped/failed
);

CREATE INDEX IF NOT EXISTS idx_tmdb_id ON movies (tmdb_id);
CREATE INDEX IF NOT EXISTS idx_rating  ON movies (rating_count);

CREATE TABLE IF NOT EXISTS movie_user_tags (
    movielens_id  INTEGER NOT NULL,
    tag           TEXT    NOT NULL,
    tag_count     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (movielens_id, tag),
    FOREIGN KEY  (movielens_id) REFERENCES movies (movielens_id)
);

CREATE INDEX IF NOT EXISTS idx_mut_movie ON movie_user_tags (movielens_id);
CREATE INDEX IF NOT EXISTS idx_mut_tag   ON movie_user_tags (tag);
"""

DROP_DDL = """
DROP TABLE IF EXISTS movie_user_tags;
DROP TABLE IF EXISTS movies;
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DROP_DDL)
    conn.commit()
    conn.executescript(DDL)
    conn.commit()


def upsert_movies(conn: sqlite3.Connection, rows: List[Dict]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO movies
           (movielens_id, tmdb_id, title, genres, avg_rating, rating_count,
            overview, tmdb_keywords, movielens_user_tags,
            content_text, tmdb_fetched)
           VALUES
           (:movielens_id, :tmdb_id, :title, :genres, :avg_rating, :rating_count,
            :overview, :tmdb_keywords, :movielens_user_tags,
            :content_text, :tmdb_fetched)""",
        rows,
    )
    conn.commit()


def insert_user_tags(conn: sqlite3.Connection, tag_freq: pd.DataFrame) -> None:
    log.info(f"Inserting {len(tag_freq):,} rows into movie_user_tags ...")
    conn.execute("DELETE FROM movie_user_tags")
    conn.executemany(
        "INSERT OR REPLACE INTO movie_user_tags (movielens_id, tag, tag_count) "
        "VALUES (?, ?, ?)",
        tag_freq[["movielens_id", "tag", "tag_count"]].itertuples(index=False, name=None),
    )
    conn.commit()
    log.info("movie_user_tags populated.")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TMDB ASYNC ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

async def _tmdb_fetch(
    session: "aiohttp.ClientSession",
    sem: asyncio.Semaphore,
    api_key: str,
    tmdb_id: int,
) -> Optional[Dict]:
    url = f"{TMDB_BASE}/movie/{tmdb_id}"
    params = {
        "api_key": api_key,
        "language": "en-US",
        "append_to_response": "keywords",
    }

    for attempt in range(1, TMDB_RETRY_MAX + 1):
        async with sem:
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        overview = data.get("overview") or ""
                        kw_data  = data.get("keywords", {}).get("keywords", [])
                        keywords = [k["name"] for k in kw_data if "name" in k]
                        return {
                            "overview": overview.strip(),
                            "keywords": keywords,
                        }
                    elif resp.status == 404:
                        return None
                    elif resp.status == 429:
                        await asyncio.sleep(1.5 * attempt)
                    else:
                        log.warning(f"TMDB HTTP {resp.status} for tmdb_id={tmdb_id} (attempt {attempt})")
                        await asyncio.sleep(0.5 * attempt)
            except Exception as e:
                log.warning(f"TMDB error for {tmdb_id} attempt {attempt}: {e}")
                await asyncio.sleep(0.5 * attempt)

    return None


async def enrich_batch(
    rows: List[Dict],
    api_key: str,
    conn: sqlite3.Connection,
    progress: Dict,
    top_n: int,
) -> None:
    if aiohttp is None:
        log.error("aiohttp not installed. Run: pip install aiohttp")
        return

    sem = asyncio.Semaphore(TMDB_CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=TMDB_CONCURRENCY + 10, ttl_dns_cache=300)

    enriched_buffer: List[Dict] = []
    done_ids: List[int] = list(progress.get("done_ids", []))
    done_id_set: Set[int] = set(done_ids)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def process_one(row: Dict) -> Dict:
            tmdb_id = row.get("tmdb_id")
            if tmdb_id is None:
                return _apply_no_tmdb(row, top_n)

            result = await _tmdb_fetch(session, sem, api_key, tmdb_id)
            if result is None:
                return _apply_tmdb_failed(row, top_n)

            return _apply_tmdb_success(row, result, top_n)

        chunk_size = CHECKPOINT_EVERY
        total = len(rows)

        for i in range(0, total, chunk_size):
            chunk = rows[i : i + chunk_size]
            tasks = [process_one(r) for r in chunk]
            results = await asyncio.gather(*tasks)

            enriched_buffer.extend(results)
            for r in results:
                ml_id = r["movielens_id"]
                if ml_id not in done_id_set:
                    done_ids.append(ml_id)
                    done_id_set.add(ml_id)

            upsert_movies(conn, enriched_buffer)
            enriched_buffer.clear()

            progress["done_ids"] = done_ids
            _save_checkpoint(progress)

            log.info(f"  checkpoint: {min(i + chunk_size, total):,}/{total:,} enriched")

    if enriched_buffer:
        upsert_movies(conn, enriched_buffer)
        log.info(f"  final flush: {len(enriched_buffer)} rows")
        enriched_buffer.clear()

    progress["done_ids"] = done_ids
    _save_checkpoint(progress)


def _apply_tmdb_success(row: Dict, result: Dict, top_n: int) -> Dict:
    tags = row["_user_tags_with_counts"]
    top_tags = extract_top_n_tags(tags, top_n)
    row["overview"]            = result["overview"] or None
    row["tmdb_keywords"]       = json.dumps(result["keywords"]) if result["keywords"] else None
    row["movielens_user_tags"] = json.dumps(top_tags) if top_tags else None
    row["content_text"]        = build_content_text(
        genres=row.get("genres", ""),
        tmdb_keywords=result["keywords"],
        user_tags_with_counts=tags,
        overview=result["overview"],
        top_n=top_n,
    )
    row["tmdb_fetched"]        = 1
    return row


def _apply_tmdb_failed(row: Dict, top_n: int) -> Dict:
    tags = row["_user_tags_with_counts"]
    top_tags = extract_top_n_tags(tags, top_n)
    row["overview"]            = None
    row["tmdb_keywords"]       = None
    row["movielens_user_tags"] = json.dumps(top_tags) if top_tags else None
    row["content_text"]        = build_content_text(
        genres=row.get("genres", ""),
        tmdb_keywords=[],
        user_tags_with_counts=tags,
        overview="",
        top_n=top_n,
    )
    row["tmdb_fetched"]        = 2
    return row


def _apply_no_tmdb(row: Dict, top_n: int) -> Dict:
    tags = row["_user_tags_with_counts"]
    top_tags = extract_top_n_tags(tags, top_n)
    row["overview"]            = None
    row["tmdb_keywords"]       = None
    row["movielens_user_tags"] = json.dumps(top_tags) if top_tags else None
    row["content_text"]        = build_content_text(
        genres=row.get("genres", ""),
        tmdb_keywords=[],
        user_tags_with_counts=tags,
        overview="",
        top_n=top_n,
    )
    row["tmdb_fetched"]        = 2
    return row


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def _load_checkpoint() -> Dict:
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "r") as f:
            return json.load(f)
    return {"done_ids": []}


def _save_checkpoint(progress: Dict) -> None:
    with open(CHECKPOINT, "w") as f:
        json.dump(progress, f)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ROW BUILDER  (MovieLens row → DB dict)
# ══════════════════════════════════════════════════════════════════════════════

def ml_row_to_dict(row: pd.Series, top_n: int) -> Dict:
    tags = row.get("user_tags_with_counts", [])
    if not isinstance(tags, list):
        tags = []

    tmdb_id = row.get("tmdb_id")
    tmdb_id_val = (
        int(tmdb_id)
        if (tmdb_id is not None and not (isinstance(tmdb_id, float) and tmdb_id != tmdb_id))
        else None
    )

    avg_rating   = row.get("avg_rating")
    rating_count = row.get("rating_count")
    top_tags     = extract_top_n_tags(tags, top_n)

    return {
        "movielens_id":        int(row["movielens_id"]),
        "tmdb_id":             tmdb_id_val,
        "title":               str(row.get("title", "")),
        "genres":              str(row.get("genres", "")) or None,
        "avg_rating":          float(avg_rating) if avg_rating is not None and avg_rating == avg_rating else None,
        "rating_count":        int(rating_count) if rating_count is not None and rating_count == rating_count else None,
        "overview":            None,
        "tmdb_keywords":       None,
        "movielens_user_tags": json.dumps(top_tags) if top_tags else None,
        "content_text":        None,   # populated during enrichment
        "tmdb_fetched":        0,
        "_user_tags_with_counts": tags,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    env_data_dir = os.getenv("MOVIELENS_DATA_DIR")
    default_dir = env_data_dir if env_data_dir else str(DEFAULT_DATA_DIR)

    p = argparse.ArgumentParser(description="Build movie_features.db")
    p.add_argument("--data-dir", type=str, default=default_dir,
                   help=f"Directory containing MovieLens 32M CSVs (default: {default_dir})")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip TMDB calls; write MovieLens data only")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N movies (for testing)")
    p.add_argument("--reset", action="store_true",
                   help="Delete existing DB and checkpoint; start fresh")
    p.add_argument("--top-n-tags", type=int, default=50,
                   help="Number of top user tags (by frequency) to include "
                        "in content_text (default: 50)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    top_n = args.top_n_tags
    log.info(f"Using MovieLens data directory: {data_dir}")
    log.info(f"top-n-tags = {top_n}")

    # ── Connect and optionally reset ─────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    if args.reset:
        log.info("--reset: dropping existing tables ...")
        reset_db(conn)
        if CHECKPOINT.exists():
            CHECKPOINT.unlink()
            log.info("Deleted checkpoint file.")
    else:
        init_db(conn)

    # ── Load MovieLens ────────────────────────────────────────────────────────
    df, tag_freq = load_movielens(data_dir)

    if args.limit:
        df = df.head(args.limit)
        log.info(f"--limit {args.limit}: processing {len(df)} movies")
        allowed = set(df["movielens_id"])
        tag_freq = tag_freq[tag_freq["movielens_id"].isin(allowed)]

    # ── Populate normalized tag table ─────────────────────────────────────────
    if args.reset or not _table_has_rows(conn, "movie_user_tags"):
        insert_user_tags(conn, tag_freq)
    else:
        log.info("movie_user_tags already populated — skipping (use --reset to rebuild)")

    # ── Build initial rows from MovieLens ─────────────────────────────────────
    log.info("Building initial movie rows ...")
    all_rows = [ml_row_to_dict(row, top_n) for _, row in df.iterrows()]

    # ── Checkpoint: skip already-enriched movies ──────────────────────────────
    progress = _load_checkpoint()
    done_set: Set[int] = set(progress.get("done_ids", []))

    new_rows = [r for r in all_rows if r["movielens_id"] not in done_set]
    if new_rows:
        log.info(f"Inserting {len(new_rows):,} movie rows (pre-enrichment) ...")
        upsert_movies(conn, new_rows)
    else:
        log.info("All rows already present in DB.")

    # ── TMDB enrichment ───────────────────────────────────────────────────────
    api_key = os.getenv("TMDB_API_KEY", "")

    if args.dry_run:
        log.info("--dry-run: skipping TMDB enrichment")
    elif not api_key or api_key == "${TMDB_API_KEY}":
        log.warning("TMDB_API_KEY not set — skipping enrichment.")
    else:
        rows_to_enrich = [r for r in all_rows if r["movielens_id"] not in done_set]
        if not rows_to_enrich:
            log.info("All movies already enriched (checkpoint complete).")
        else:
            log.info(
                f"Starting TMDB enrichment for {len(rows_to_enrich):,} movies "
                f"({TMDB_CONCURRENCY} concurrent) ..."
            )
            t0 = time.time()
            asyncio.run(enrich_batch(rows_to_enrich, api_key, conn, progress, top_n))
            elapsed = time.time() - t0
            log.info(f"TMDB enrichment done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    conn2 = sqlite3.connect(DB_PATH)
    cur = conn2.cursor()
    total    = cur.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    enriched = cur.execute("SELECT COUNT(*) FROM movies WHERE tmdb_fetched=1").fetchone()[0]
    skipped  = cur.execute("SELECT COUNT(*) FROM movies WHERE tmdb_fetched=2").fetchone()[0]
    pending  = cur.execute("SELECT COUNT(*) FROM movies WHERE tmdb_fetched=0").fetchone()[0]
    tag_rows = cur.execute("SELECT COUNT(*) FROM movie_user_tags").fetchone()[0]
    has_ct   = cur.execute(
        "SELECT COUNT(*) FROM movies WHERE content_text IS NOT NULL"
    ).fetchone()[0]
    conn2.close()

    log.info("=" * 62)
    log.info(f"  Database    : {DB_PATH}")
    log.info(f"  Movies      : {total:,}  total")
    log.info(f"  TMDB ok     : {enriched:,}")
    log.info(f"  TMDB skip   : {skipped:,}  (no tmdb_id or fetch failed)")
    log.info(f"  Pending     : {pending:,}  (run again to continue)")
    log.info(f"  Tags table  : {tag_rows:,}  rows  (movie_user_tags)")
    log.info(f"  content_text: {has_ct:,}  non-null")
    log.info("=" * 62)

    if pending > 0 and not args.dry_run:
        log.info("Re-run the script to continue from checkpoint.")


def _table_has_rows(conn: sqlite3.Connection, table: str) -> bool:
    try:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return n > 0
    except Exception:
        return False


if __name__ == "__main__":
    main()
