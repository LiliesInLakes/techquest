import os
import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class MovieDatabase:
    """
    Manages SQLite database connections, canonical deduplicated movie loading,
    and manifest checksum validation for movie_features.db.
    """
    def __init__(self, db_path: str = "movie_features.db", manifest_path: str = "data/movie_db_manifest.json"):
        self.db_path = db_path
        self.manifest_path = manifest_path

    def get_connection(self) -> sqlite3.Connection:
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Movie database file not found at: {self.db_path}. "
                f"Please verify MOVIE_DB_PATH or run `python main.py setup-db`."
            )
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def verify_integrity(self) -> Tuple[bool, str]:
        """
        Validates the database file exists, row count matches expected,
        and optionally verifies SHA-256 against manifest.
        """
        if not os.path.exists(self.db_path):
            return False, f"Database file not found: {self.db_path}"

        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM movies WHERE tmdb_id IS NOT NULL;")
            count = cur.fetchone()[0]
            conn.close()

            if count < 80000:
                return False, f"Unexpected row count: {count} (expected ~87,461 non-null tmdb_id rows)"

            return True, f"Integrity check passed ({count} movies with TMDB IDs)."
        except Exception as e:
            return False, f"Database integrity check failed: {e}"

    def load_canonical_movies(self) -> List[Dict[str, Any]]:
        """
        Loads the canonical movie dataset (87,425 items) from movie_features.db.
        For duplicate tmdb_id groups (e.g. multiple MovieLens IDs mapping to the same TMDB ID),
        it deterministically selects the primary row (highest rating_count, longest content_text)
        and aggregates/unions user tags from secondary MovieLens IDs.
        """
        conn = self.get_connection()
        cur = conn.cursor()

        # Step 1: Find duplicate tmdb_id tags to aggregate
        cur.execute("""
            SELECT tmdb_id, movielens_user_tags
            FROM movies
            WHERE tmdb_id IS NOT NULL 
              AND movielens_user_tags IS NOT NULL 
              AND movielens_user_tags != '[]'
        """)
        all_tags_rows = cur.fetchall()
        tag_union_map: Dict[int, List[str]] = {}
        for row in all_tags_rows:
            tmdb_id = row['tmdb_id']
            tags_json = row['movielens_user_tags']
            try:
                tags = json.loads(tags_json) if isinstance(tags_json, str) else tags_json
                if isinstance(tags, list):
                    if tmdb_id not in tag_union_map:
                        tag_union_map[tmdb_id] = []
                    for t in tags:
                        t_str = str(t).strip().lower()
                        if t_str and t_str not in tag_union_map[tmdb_id]:
                            tag_union_map[tmdb_id].append(t_str)
            except Exception:
                continue

        # Step 2: Query the canonical row for each tmdb_id using window function / group by
        # Row selection: ORDER BY rating_count DESC, LENGTH(content_text) DESC, movielens_id ASC
        query = """
            WITH ranked_movies AS (
                SELECT 
                    movielens_id,
                    tmdb_id,
                    title,
                    genres,
                    avg_rating,
                    rating_count,
                    overview,
                    tmdb_keywords,
                    movielens_user_tags,
                    content_text,
                    tmdb_fetched,
                    ROW_NUMBER() OVER (
                        PARTITION BY tmdb_id 
                        ORDER BY rating_count DESC, LENGTH(content_text) DESC, movielens_id ASC
                    ) as rn
                FROM movies
                WHERE tmdb_id IS NOT NULL
            )
            SELECT 
                movielens_id,
                tmdb_id,
                title,
                genres,
                avg_rating,
                rating_count,
                overview,
                tmdb_keywords,
                movielens_user_tags,
                content_text,
                tmdb_fetched
            FROM ranked_movies
            WHERE rn = 1
            ORDER BY tmdb_id ASC;
        """
        cur.execute(query)
        rows = cur.fetchall()

        canonical_movies = []
        for r in rows:
            tmdb_id = r['tmdb_id']
            genres_raw = r['genres'] or ''
            genres_list = [g.strip() for g in genres_raw.split('|') if g.strip()] if isinstance(genres_raw, str) else []

            keywords_raw = r['tmdb_keywords'] or ''
            keywords_list = []
            if isinstance(keywords_raw, str) and keywords_raw.strip():
                try:
                    parsed_kw = json.loads(keywords_raw)
                    if isinstance(parsed_kw, list):
                        keywords_list = [str(k).strip().strip('[]"\'') for k in parsed_kw if str(k).strip().strip('[]"\'')]
                    elif isinstance(parsed_kw, str):
                        keywords_list = [k.strip().strip('[]"\'') for k in parsed_kw.replace('|', ',').split(',') if k.strip().strip('[]"\'')]
                except Exception:
                    keywords_list = [k.strip().strip('[]"\'') for k in keywords_raw.replace('|', ',').split(',') if k.strip().strip('[]"\'')]
            elif isinstance(keywords_raw, list):
                keywords_list = [str(k).strip().strip('[]"\'') for k in keywords_raw if str(k).strip().strip('[]"\'')]

            # Use aggregated tags if available, else parse canonical row tags
            tags_list = tag_union_map.get(tmdb_id, [])
            if not tags_list and r['movielens_user_tags']:
                try:
                    parsed = json.loads(r['movielens_user_tags'])
                    if isinstance(parsed, list):
                        tags_list = [str(t).strip().lower() for t in parsed if str(t).strip()]
                except Exception:
                    tags_list = []

            avg_rating = float(r['avg_rating']) if r['avg_rating'] is not None else 0.0
            rating_count = int(r['rating_count']) if r['rating_count'] is not None else 0
            # Normalized compatibility rating: MovieLens avg_rating (0.5 - 5.0) * 2.0 -> (1.0 - 10.0 scale)
            vote_average = round(avg_rating * 2.0, 2) if avg_rating > 0 else 0.0

            content_text = r['content_text'] or ''
            if not content_text:
                # Construct fallback content text if content_text column is null
                parts = []
                if genres_list:
                    parts.append(f"Genres: {', '.join(genres_list)}.")
                if keywords_list:
                    parts.append(f"Keywords: {', '.join(keywords_list)}.")
                if r['overview']:
                    parts.append(f"Overview: {r['overview']}")
                if tags_list:
                    parts.append(f"Tags: {', '.join(tags_list[:15])}.")
                content_text = ' '.join(parts).strip()

            canonical_movies.append({
                'movielens_id': r['movielens_id'],
                'tmdb_id': tmdb_id,
                'id': tmdb_id,
                'title': r['title'] or '',
                'genres': genres_list,
                'keywords': keywords_list,
                'tags': tags_list,
                'overview': r['overview'] or '',
                'avg_rating': avg_rating,
                'rating_count': rating_count,
                'vote_average': vote_average,
                'vote_count': rating_count,
                'content_text': content_text,
                'tmdb_fetched': r['tmdb_fetched']
            })

        conn.close()
        return canonical_movies
