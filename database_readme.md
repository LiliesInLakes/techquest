# Movie Feature Database Documentation (`movie_features.db`)

## Overview

`movie_features.db` is a clean, compact, high-signal SQLite feature store designed for movie retrieval, filtering, and semantic search. It fuses:
1. **MovieLens 32M**: User-generated ratings (volume and average) and comprehensive user tag applications.
2. **TMDB (The Movie Database)**: Plot overviews and keyword metadata.

It provides a single canonical textual representation — `content_text` — alongside structured raw metadata and a complete normalized tag frequency table.

---

## 1. Database Schema

```
movie_features.db
├── movies (table)
│   ├── movielens_id        INTEGER PRIMARY KEY   -- MovieLens 32M movie ID
│   ├── tmdb_id             INTEGER               -- The Movie Database ID (indexed)
│   ├── title               TEXT                  -- Movie title with release year
│   ├── genres              TEXT                  -- Pipe-separated genres ("Action|Sci-Fi")
│   ├── avg_rating          REAL                  -- Mean rating from MovieLens ratings.csv (NULL if unrated)
│   ├── rating_count        INTEGER               -- Total user ratings count (NULL if unrated, indexed)
│   ├── overview            TEXT                  -- TMDB synopsis / plot description
│   ├── tmdb_keywords       TEXT                  -- JSON array of TMDB keywords
│   ├── movielens_user_tags TEXT                  -- JSON array of top-50 tags by tag_count descending
│   ├── content_text        TEXT                  -- Canonical natural-language text representation
│   └── tmdb_fetched        INTEGER DEFAULT 0     -- 0=pending, 1=success, 2=skipped/fallback
│
└── movie_user_tags (table)
    ├── movielens_id        INTEGER NOT NULL      -- FK -> movies.movielens_id
    ├── tag                 TEXT NOT NULL         -- Cleaned lowercased tag
    └── tag_count           INTEGER NOT NULL      -- Total times applied in MovieLens tags.csv
```

### Indexes
- `idx_tmdb_id` on `movies(tmdb_id)`
- `idx_rating` on `movies(rating_count)`
- `idx_mut_movie` on `movie_user_tags(movielens_id)`
- `idx_mut_tag` on `movie_user_tags(tag)`
- `sqlite_autoindex_movie_user_tags_1` PRIMARY KEY `(movielens_id, tag)`

---

## 2. Summary Statistics (Actual Verified Build)

| Metric | Count | % of Total | Notes |
| :--- | :--- | :--- | :--- |
| **Total Movies** | **87,585** | 100% | Full MovieLens 32M dataset |
| **TMDB Enriched** | 86,250 | 98.5% | Enriched with plot & keywords |
| **TMDB Fallback / Skipped** | 1,335 | 1.5% | No TMDB ID or API 404 |
| **Has TMDB ID** | 87,461 | 99.9% | Mapped via links.csv |
| **Has Overview** | 86,013 | 98.2% | TMDB plot synopsis |
| **Has TMDB Keywords** | 62,830 | 71.7% | Clean descriptive keywords |
| **Has User Tags** | 51,323 | 58.6% | Movies with ≥1 user tag |
| **Has `content_text`** | 87,380 | 99.8% | Only 205 movies with 0 text |
| **Has `avg_rating` & `rating_count`** | 84,432 | 96.4% | Preserved as structured metadata |
| **Total `movie_user_tags` Rows** | **1,071,041** | — | 100% of unique (movie, tag) pairs |

---

## 3. Field Specifications & Construction

### A. Canonical `content_text`
`content_text` is a shared, clean natural-language movie representation suitable as a baseline input for both lexical TF-IDF retrieval and dense semantic embedding.

#### Construction Order:
1. **Genres**: `"Genres: action, sci-fi, thriller."`
2. **TMDB Keywords**: `"Keywords: spacecraft, black hole, time travel."`
3. **MovieLens User Tags**: `"User tags: alternate reality, thought-provoking, mind-bending."` (Top-50 sorted by `tag_count DESC`)
4. **TMDB Overview**: `"Overview: A thief who steals corporate secrets through dream-sharing technology..."`

#### Formatting Rules:
- Multi-word phrases are preserved naturally (e.g. `"black hole"`, `"time travel"`, `"coming of age"`).
- No artificial dashed tokens (`action-adventure-superhero`).
- No noise fields: Taglines, cast, directors, ratings, dates, runtime, language, popularity, and numeric IDs are strictly omitted.
- Graceful degradation: If a section is missing, it is omitted without inserting empty placeholders.

### B. Raw Fields Kept Independently
To allow downstream systems to build specialized representations without rebuilding the database, all raw source fields remain available:
- `genres` (raw pipe-separated string)
- `overview` (raw TMDB prose)
- `tmdb_keywords` (JSON array of strings)
- `movielens_user_tags` (JSON array of top-50 strings)
- `avg_rating` & `rating_count` (MovieLens rating statistics)

### C. Normalized Tag Store (`movie_user_tags`)
- Contains all **1,071,041** unique `(movielens_id, tag)` pairs from MovieLens `tags.csv`.
- `tag_count` records the exact number of times users applied that tag to the movie (ranging from 1 to 603).
- No single-use tags are discarded from this table.

---

## 4. Source Data & Setup Instructions

The database builder requires the **MovieLens 32M** dataset released by GroupLens Research.

### A. Required Source Files
Download the dataset archive from GroupLens:
- **Download URL**: [https://grouplens.org/datasets/movielens/32m/](https://grouplens.org/datasets/movielens/32m/)
- **Direct Archive**: `https://files.grouplens.org/datasets/movielens/ml-32m.zip` (~290 MB compressed, ~911 MB uncompressed)

Extract the four required CSV files into `data/ml-32m/`:
```
data/
└── ml-32m/
    ├── movies.csv      (4.05 MB — movie titles and genres)
    ├── links.csv       (1.86 MB — MovieLens to TMDB ID mappings)
    ├── ratings.csv     (836.45 MB — 32M user ratings)
    └── tags.csv        (68.99 MB — 2M user tag applications)
```

*(Note: `genome-tags.csv` and `genome-scores.csv` are not part of the 32M release and are not needed).*

### B. Configuring Dataset Path
By default, `build_movie_db.py` looks in `data/ml-32m/`. You can specify a custom location:
1. **Via CLI argument**:
   ```bash
   python build_movie_db.py --data-dir /path/to/extracted/ml-32m
   ```
2. **Via Environment Variable**:
   ```bash
   export MOVIELENS_DATA_DIR=/path/to/extracted/ml-32m
   ```

If the required files are missing, the builder terminates with a clear diagnostic message and download instructions.

---

## 5. Usage in Downstream Systems

### Lexical Retrieval (TF-IDF / CountVectorizer)
```python
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

conn = sqlite3.connect("movie_features.db")
df = pd.read_sql_query(
    "SELECT movielens_id, title, genres, content_text FROM movies WHERE content_text IS NOT NULL",
    conn
)

# Use bigrams to capture natural multi-word phrases (e.g. "time travel", "black hole")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=25000, sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(df["content_text"])
```

### Semantic Embedding (Sentence Transformers)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["content_text"].tolist(), show_progress_bar=True, normalize_embeddings=True)
```

---

## 6. Notes on Short Documents & Downstream Handling

Lexical matching (TF-IDF) on raw text can occasionally score short documents highly if a query token appears in a sparse document.
- The database faithfully stores all movies regardless of popularity or length.
- Downstream retrieval pipelines should handle short document bias via:
  - Query-time popularity filters (e.g. `rating_count >= 50`)
  - Minimum token length thresholds
  - Two-stage re-ranking combining lexical and semantic signals

---

## 7. Build & Verification Commands

### Rebuilding from Scratch
```bash
# Clean rebuild (MovieLens CSVs + TMDB API)
python build_movie_db.py --reset

# Using custom dataset path
python build_movie_db.py --reset --data-dir data/ml-32m

# Fast dry-run (MovieLens CSVs only, no TMDB API calls)
python build_movie_db.py --dry-run
```

### Running Verification
```bash
# Run full diagnostics, leakage detection, TF-IDF tests, and sentence-transformer tests
python verify_db.py
```
