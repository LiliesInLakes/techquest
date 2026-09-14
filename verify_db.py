"""
verify_db.py
============
Comprehensive verification of movie_features.db.

Sections:
  1. SCHEMA    — table definitions, column checks, banned column absence
  2. COVERAGE  — total movies, enrichment status, ID & text coverage
  3. TAGS      — movie_user_tags table integrity and frequency distribution
  4. TEXT      — content_text length, token stats, and distribution
  5. LEAKAGE   — detect contamination by fields that must NOT appear
  6. TF-IDF    — lexical search queries on content_text (5 queries)
  7. SEMANTIC  — sentence-transformer cosine similarity on content_text (5 queries)

Usage:
    python verify_db.py
    python verify_db.py --no-semantic    # skip embedding tests (faster)
    python verify_db.py --rating-floor 100  # query-time pre-filter
"""

import sys
import io
import json
import argparse
import sqlite3

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"Missing dependency: {e}. Install: pip install pandas scikit-learn", file=sys.stderr)
    sys.exit(1)

DB = "movie_features.db"
SEP = "=" * 68


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SCHEMA & COVERAGE
# ══════════════════════════════════════════════════════════════════════════════

def check_schema_and_coverage(conn: sqlite3.Connection) -> None:
    section("1. SCHEMA & DATABASE COVERAGE")
    cur = conn.cursor()

    # Tables
    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    print(f"  Tables present: {tables}")

    # Columns in movies table
    movie_cols = [r[1] for r in cur.execute("PRAGMA table_info(movies)").fetchall()]
    print(f"  movies columns: {movie_cols}")

    expected_cols = {
        "movielens_id", "tmdb_id", "title", "genres", "avg_rating", "rating_count",
        "overview", "tmdb_keywords", "movielens_user_tags", "content_text", "tmdb_fetched"
    }
    banned_cols = {"all_tags", "search_text", "semantic_text", "tagline", "cast_top10", "directors", "runtime", "release_date", "popularity"}

    missing = expected_cols - set(movie_cols)
    extra_banned = banned_cols.intersection(set(movie_cols))

    print(f"  Missing required columns: {list(missing) if missing else 'None (ok)'}")
    print(f"  Banned columns present  : {list(extra_banned) if extra_banned else 'None (ok)'}")

    total = cur.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    print(f"\n  Total movie rows        : {total:,}")

    def pct(n):
        return f"{100 * n // total}%" if total else "0%"

    for label, sql in [
        ("TMDB enriched (=1)", "SELECT COUNT(*) FROM movies WHERE tmdb_fetched=1"),
        ("TMDB skipped  (=2)", "SELECT COUNT(*) FROM movies WHERE tmdb_fetched=2"),
        ("Pending       (=0)", "SELECT COUNT(*) FROM movies WHERE tmdb_fetched=0"),
        ("Has tmdb_id",        "SELECT COUNT(*) FROM movies WHERE tmdb_id IS NOT NULL"),
        ("Has overview",       "SELECT COUNT(*) FROM movies WHERE overview IS NOT NULL AND overview != ''"),
        ("Has tmdb_keywords",  "SELECT COUNT(*) FROM movies WHERE tmdb_keywords IS NOT NULL AND tmdb_keywords != 'null'"),
        ("Has user_tags",      "SELECT COUNT(*) FROM movies WHERE movielens_user_tags IS NOT NULL AND movielens_user_tags != '[]'"),
        ("Has content_text",   "SELECT COUNT(*) FROM movies WHERE content_text IS NOT NULL AND content_text != ''"),
        ("Has avg_rating",     "SELECT COUNT(*) FROM movies WHERE avg_rating IS NOT NULL"),
        ("Has rating_count",   "SELECT COUNT(*) FROM movies WHERE rating_count IS NOT NULL"),
    ]:
        n = cur.execute(sql).fetchone()[0]
        print(f"  {label:<28}: {n:>8,}  ({pct(n)})")

    # Duplicates check
    dups = cur.execute(
        "SELECT COUNT(*) FROM (SELECT movielens_id, COUNT(*) c FROM movies GROUP BY movielens_id HAVING c > 1)"
    ).fetchone()[0]
    print(f"\n  Duplicate movielens_id rows : {dups}  {'<<< PROBLEM' if dups else '(ok)'}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  NORMALIZED TAG TABLE
# ══════════════════════════════════════════════════════════════════════════════

def check_tags_table(conn: sqlite3.Connection) -> None:
    section("2. NORMALIZED TAG TABLE (movie_user_tags)")
    cur = conn.cursor()

    total_pairs = cur.execute("SELECT COUNT(*) FROM movie_user_tags").fetchone()[0]
    movies_with_tags = cur.execute("SELECT COUNT(DISTINCT movielens_id) FROM movie_user_tags").fetchone()[0]
    distinct_tags = cur.execute("SELECT COUNT(DISTINCT tag) FROM movie_user_tags").fetchone()[0]

    print(f"  Total (movie, tag) pairs    : {total_pairs:,}")
    print(f"  Movies with >= 1 tag        : {movies_with_tags:,}")
    print(f"  Unique distinct tags        : {distinct_tags:,}")

    # Duplicate pair check
    dups = cur.execute(
        "SELECT COUNT(*) FROM (SELECT movielens_id, tag, COUNT(*) c FROM movie_user_tags GROUP BY movielens_id, tag HAVING c > 1)"
    ).fetchone()[0]
    print(f"  Duplicate (movie,tag) pairs : {dups}  {'<<< PROBLEM' if dups else '(ok)'}")

    # tag_count statistics
    stats = cur.execute(
        "SELECT MIN(tag_count), MAX(tag_count), AVG(tag_count) FROM movie_user_tags"
    ).fetchone()
    print(f"\n  tag_count distribution:")
    print(f"    min    : {stats[0]}")
    print(f"    max    : {stats[1]}")
    print(f"    mean   : {stats[2]:.2f}")

    for threshold in [1, 5, 10]:
        n = cur.execute(f"SELECT COUNT(*) FROM movie_user_tags WHERE tag_count > {threshold}").fetchone()[0]
        print(f"    > {threshold:>2}   : {n:>8,}  ({100*n//total_pairs if total_pairs else 0}%)")

    # Top 5 movie-tag pairs by frequency
    print("\n  Top 5 (movie, tag) by tag_count:")
    top_pairs = cur.execute(
        "SELECT movielens_id, tag, tag_count FROM movie_user_tags ORDER BY tag_count DESC LIMIT 5"
    ).fetchall()
    for ml_id, tag, count in top_pairs:
        print(f"    ML_ID={ml_id:<8} tag_count={count:<6} '{tag}'")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CONTENT_TEXT QUALITY & DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def check_text_distribution(conn: sqlite3.Connection) -> None:
    section("3. CONTENT_TEXT FIELD QUALITY")

    df = pd.read_sql_query(
        "SELECT movielens_id, title, content_text FROM movies",
        conn,
    )

    non_null = df["content_text"].dropna()
    non_null = non_null[non_null.str.strip() != ""]

    print(f"  content_text:")
    print(f"    non-null / non-empty rows : {len(non_null):,}")
    print(f"    null / empty rows         : {len(df) - len(non_null):,}")

    lengths = non_null.str.len()
    tokens = non_null.apply(lambda s: len(s.split()))

    print(f"    char length  — min={lengths.min()}  median={lengths.median():.0f}  mean={lengths.mean():.0f}  max={lengths.max()}")
    print(f"    token count  — min={tokens.min()}  median={tokens.median():.0f}  mean={tokens.mean():.0f}  max={tokens.max()}")

    # Sample movie row (Inception)
    sample = df[df["title"].str.contains("Inception", case=False, na=False)]
    if not sample.empty:
        r = sample.iloc[0]
        print(f"\n  Sample row — {r['title']}:")
        text = str(r['content_text'])
        print(f"    content_text ({len(text.split())} tokens):")
        print(f"      {text[:350]}...")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LEAKAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def check_leakage(conn: sqlite3.Connection) -> None:
    section("4. LEAKAGE & CONTAMINATION DETECTION")

    cur = conn.cursor()
    # Check schema for banned columns
    cols = [r[1] for r in cur.execute("PRAGMA table_info(movies)").fetchall()]
    banned = ["all_tags", "tagline", "cast_top10", "directors", "search_text", "semantic_text"]
    found_banned = [c for c in banned if c in cols]
    if found_banned:
        print(f"  FAIL: Found banned columns in schema: {found_banned}")
    else:
        print(f"  OK: No banned columns present in schema ({', '.join(banned)})")

    # Check content_text for leakage
    print("\n  Scanning content_text for pattern indicators:")
    for pat in ["runtime:", "budget:", "revenue:", "popularity:", "tmdb_id:"]:
        n = cur.execute(
            f"SELECT COUNT(*) FROM movies WHERE content_text LIKE '%{pat}%'"
        ).fetchone()[0]
        print(f"    '{pat}' in content_text : {n} rows")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TF-IDF SANITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

QUERIES = [
    "space exploration black hole time dilation sci-fi",
    "superhero cape justice gotham vigilante",
    "animated toys friendship childhood adventure",
    "soft romantic comforting movie",
    "dark psychological disturbing thriller",
]

def run_tfidf_tests(conn: sqlite3.Connection, rating_floor: int = 100) -> None:
    section(f"5. TF-IDF SANITY TESTS  (rating_count >= {rating_floor} query filter)")

    df = pd.read_sql_query(
        f"""SELECT movielens_id, title, genres, content_text
            FROM movies
            WHERE content_text IS NOT NULL AND content_text != ''
              AND rating_count >= {rating_floor}""",
        conn,
    )
    print(f"  Corpus: {len(df):,} movies (with rating_count >= {rating_floor})")
    print("  Vectorizing content_text (ngram_range=(1,2), max_features=20000) ...")

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=20000,
        sublinear_tf=True,
    )
    mat = vec.fit_transform(df["content_text"])
    print(f"  TF-IDF matrix shape: {mat.shape}")

    for q in QUERIES:
        print(f"\n  Query: \"{q}\"")
        q_vec = vec.transform([q])
        scores = cosine_similarity(q_vec, mat).flatten()
        top_idx = np.argsort(scores)[::-1][:8]

        for rank, idx in enumerate(top_idx, 1):
            row = df.iloc[idx]
            sim = scores[idx]
            print(f"    {rank}. {row['title']:<45} score={sim:.4f}  |  {row['genres']}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SEMANTIC SANITY TESTS (Sentence Transformers)
# ══════════════════════════════════════════════════════════════════════════════

def run_semantic_tests(conn: sqlite3.Connection, rating_floor: int = 1000) -> None:
    section(f"6. SEMANTIC SANITY TESTS  (all-MiniLM-L6-v2 cosine similarity)")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not installed. Skipping semantic tests.")
        return

    print("  Loading model: all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    df = pd.read_sql_query(
        f"""SELECT movielens_id, title, genres, content_text
            FROM movies
            WHERE content_text IS NOT NULL AND content_text != ''
              AND rating_count >= {rating_floor}
            LIMIT 5000""",
        conn,
    )
    print(f"  Encoding {len(df):,} movies (rating_count >= {rating_floor}) ...")
    emb = model.encode(df["content_text"].tolist(), show_progress_bar=False, normalize_embeddings=True)
    print(f"  Embeddings shape: {emb.shape}")

    for q in QUERIES:
        print(f"\n  Query: \"{q}\"")
        print("  (Raw cosine similarity [0..1], NOT percentage)")
        q_emb = model.encode([q], normalize_embeddings=True)
        scores = (emb @ q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:8]

        for rank, idx in enumerate(top_idx, 1):
            row = df.iloc[idx]
            sim = scores[idx]
            print(f"    {rank}. {row['title']:<45} cos_sim={sim:.4f}  |  {row['genres']}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify movie_features.db")
    parser.add_argument("--no-semantic", action="store_true", help="Skip sentence-transformer tests")
    parser.add_argument("--rating-floor", type=int, default=100, help="Rating count floor for TF-IDF tests")
    args = parser.parse_args()

    conn = sqlite3.connect(DB)

    check_schema_and_coverage(conn)
    check_tags_table(conn)
    check_text_distribution(conn)
    check_leakage(conn)
    run_tfidf_tests(conn, rating_floor=args.rating_floor)

    if not args.no_semantic:
        run_semantic_tests(conn)

    conn.close()
    print(f"\n{SEP}")
    print("  Verification complete.")
    print(SEP)


if __name__ == "__main__":
    main()
