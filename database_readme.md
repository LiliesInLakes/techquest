# Movie Features Database Integration (`movie_features.db`)

## Overview

`movie_features.db` is the single authoritative offline movie recommendation corpus for TechQuest. It contains comprehensive metadata and features derived from the MovieLens (32M dataset) and TMDB.

## Database Summary & Canonical Corpus

| Property | Value |
|---|---|
| **Database File** | `movie_features.db` (SQLite 3) |
| **File Size** | 256,065,536 bytes (~244.2 MB) |
| **Raw Rows** | 87,585 |
| **Null TMDB IDs** | 124 (excluded from canonical corpus) |
| **Duplicate TMDB ID Groups** | 36 groups (48 rows) |
| **Canonical Movie Count** | **87,425 unique TMDB movies** |
| **Embedding Model** | `multi-qa-mpnet-base-dot-v1` (768 dimensions, L2-normalized) |
| **Embedding Matrix** | 87,425 × 768 float32 (`data/processed/movie_embeddings.npy`, ~268.5 MB in RAM) |
| **TF-IDF Vocabulary** | 5,000 features (`CountVectorizer` + `TfidfVectorizer`) |
| **Similarity Query Latency** | ~11.5 ms on-demand dot-product across all 87.4k movies |

## Canonical Deduplication Policy

When multiple records share the same `tmdb_id`:
1. The primary record is selected deterministically by:
   `ORDER BY rating_count DESC, LENGTH(content_text) DESC, movielens_id ASC`
2. Distinct MovieLens user tags (`movielens_tags`) across all records in the duplicate group are unioned and appended to the primary record's `content_text` and `movielens_tags` field.
3. Raw metadata (titles, original release dates, ratings, overviews) is retained separately in the database columns.
4. This guarantees a strict 1-to-1 mapping between `tmdb_id` and feature matrix / embedding index rows.

## Candidate Filtering & Live TMDB Interaction

1. **Live Candidate Discovery**: TMDB API discovers candidate movies matching explicit user hard constraints (explicit genres, release dates, runtime, language, certifications, providers).
2. **SQLite Membership Filter**: If TMDB returns a candidate whose `tmdb_id` is **absent** from `movie_features.db`, it is **strictly excluded** before ranking.
3. **No Dynamic Runtime Embeddings**: Features are never computed ad-hoc for out-of-database movies, preventing latency spikes and semantic drift.
4. **Coverage Telemetry**: The pipeline logs exact candidate counts before and after database filtering:
   `[Candidate Discovery] TMDB: X | SQLite: Y | Excluded: Z | Coverage: W%`
5. **Multi-Page Discovery Top-up**: If SQLite filtering reduces candidates below the target threshold (50), the recommender fetches additional TMDB discovery pages (up to 5 pages) to ensure adequate candidate density.

## Offline Fallback Mode

When TMDB API is unreachable or `TMDB_API_KEY` is not provided:
- Candidates are drawn directly from `movie_features.db` using SQL filtering that honors all extracted hard constraints (genres, release year range, runtime, language, rating floor).
- Candidates are sorted by `rating_count DESC, avg_rating DESC`.

## Schema Reference

```sql
CREATE TABLE movie_features (
    movielens_id INTEGER PRIMARY KEY,
    tmdb_id REAL,
    title TEXT,
    release_date TEXT,
    genres TEXT,
    overview TEXT,
    avg_rating REAL,
    rating_count INTEGER,
    tmdb_vote_average REAL,
    tmdb_vote_count REAL,
    popularity REAL,
    poster_path TEXT,
    backdrop_path TEXT,
    original_language TEXT,
    runtime REAL,
    directors TEXT,
    cast_members TEXT,
    keywords TEXT,
    movielens_tags TEXT,
    content_text TEXT,
    tmdb_fetched INTEGER,
    enrichment_status TEXT,
    last_updated TEXT
);
CREATE INDEX idx_mf_tmdb_id ON movie_features(tmdb_id);
```

## Runtime Asset Distribution & Setup

Prebuilt runtime artifacts (`movie_features.db`, 768-d embeddings, sparse TF-IDF matrices) are distributed via GitHub Releases.

To download and verify them:
```bash
python scripts/setup_assets.py
```

To verify the integrity of currently installed assets:
```bash
python scripts/setup_assets.py --verify-only
```

### Local Precomputation & Rebuilding (Optional)

To rebuild the feature artifacts locally from `movie_features.db`:
```bash
python main.py precompute
```

This generates:
- `data/processed/movie_embeddings.npy` (87,425 × 768 float32 matrix)
- `data/processed/movie_id_map.json` (TMDB ID to matrix index mapping)
- `data/processed/count_vectorizer.pkl` (5,000-feature CountVectorizer)
- `data/processed/tfidf_vectorizer.pkl` (5,000-feature TfidfVectorizer)
- `data/processed/count_matrix.npz` (87,425 × 5,000 sparse count matrix)
- `data/processed/tfidf_matrix.npz` (87,425 × 5,000 sparse TF-IDF matrix)
- `data/processed/keyword_feature_embeddings.npy` (Vocabulary embeddings)
- `data/processed/keyword_embeddings.pkl` (Keyword dictionary)
- `data/processed/precompute_manifest.json` (Artifact checksums and build metadata)

Precomputation is fully resumable: checkpoints are saved after every batch in `data/processed/checkpoints/`.

## Retired Legacy Files

The following legacy data files are completely superseded and retired from runtime:
- `data/tmdb_5000_movies.csv` (legacy TMDB 5000 corpus)
- `data/tmdb_5000_credits.csv` (legacy credits)
- `data/processed/semantic_index.pkl` (retired in favor of fast ~11.5ms on-demand dot-product)
