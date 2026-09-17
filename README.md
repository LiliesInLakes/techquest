# Movie Recommender Agent — In-Depth Guide

> **One-line pitch:** An agentic movie-recommendation system blending live TMDB discovery, authoritative SQLite corpus (87,425 canonical movies), offline feature engineering (5,000-d TF-IDF + 768-d `multi-qa-mpnet-base-dot-v1` dense embeddings), two-stage ranking with age-relative popularity, and LLM explanations. It can be driven via **Web UI**, **CLI**, **MCP server** (Model Context Protocol), or **A2A agent** (Agent-to-Agent protocol).

If you have never seen this codebase before, read this file top-to-bottom. By the end you should be able to set it up, run it, trace a request through every layer, and extend it.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Repository Layout](#4-repository-layout)
5. [Data & Offline Artifacts](#5-data--offline-artifacts)
6. [Configuration](#6-configuration)
7. [Core Recommendation Pipeline (End-to-End Workflow)](#7-core-recommendation-pipeline-end-to-end-workflow)
8. [Module Deep-Dives](#8-module-deep-dives)
9. [Interfaces: UI, CLI, MCP, A2A](#9-interfaces-ui-cli-mcp-a2a)
10. [Setup From Scratch](#10-setup-from-scratch)
11. [Usage Cookbook](#11-usage-cookbook)
12. [Testing](#12-testing)
13. [Fallback & Offline Behavior](#13-fallback--offline-behavior)
14. [How to Extend It](#14-how-to-extend-it)
15. [Glossary & Limitations](#15-glossary--limitations)

---

## 1. What This System Does

Given a user ID, watch history (`[{movie_id, rating}]`), subscriptions, region/language, and optionally a natural-language request like *"mind-bending 90s sci-fi thriller"*, return a ranked recommendation with an AI explanation (*"Because you enjoyed..."*).

The orchestrator is `MovieRecommender` (`src/pipeline/recommender.py`):
1. Loads or creates the user's multi-dimensional profile (genres, user tags, themes, tones, dislikes).
2. Parses the natural-language query, strictly separating explicit hard constraints for TMDB from soft concepts for semantic ranking.
3. Discovers live candidates via TMDB and filters them against the authoritative SQLite canonical corpus (`movie_features.db`).
4. Scores candidates through Stage 1 ranking (quality + TF-IDF + semantic + age-relative popularity) and Stage 2 personalization (history boost + dislike penalties).
5. Applies diversity filtering and generates a decisive verdict/explanation via Kado LLM or template engine.

---

## 2. High-Level Architecture

```
                    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌─────────────┐
                    │  Web UI      │      │  CLI         │      │ MCP Server   │      │ A2A Agent   │
  User / Agent ─────▶│  (:8080)     │      │  (main.py)   │      │ (FastAPI     │────▶│ (3 caps)    │
   or External App  │  Showcase    │      │  Interactive │      │  :8000)      │      │             │
                    └──────┬───────┘      └──────┬───────┘      └──────┬───────┘      └──────┬──────┘
                           │                     │                     │                     │
                           ▼                     ▼                     ▼                     ▼
                ┌──────────────────────────────────────────────────────────────────────────────────┐
                │                          Pipeline: MovieRecommender                              │
                │  1. UserMemoryService: Multi-Dimensional Profile (Genres, Tags, Themes, Tones)   │
                │  2. NLFeatureExtractor: Hard vs Soft Constraint Separation (KADO / regex)        │
                │  3. TMDBClient: Live Discovery + Provider ID resolution                          │
                │  4. SQLite Filter: 87,425 Canonical Movie Membership Filter (movie_features.db)  │
                │  5. Vectorization: multi-qa-mpnet-base-dot-v1 (768-dim) + TF-IDF (5,000-dim)    │
                │  6. MovieRanker.rank_movies_stage1: 0.30 Quality + 0.25 TF-IDF + 0.35 Semantic   │
                │     + 0.10 Age-Adjusted Cohort Popularity (No Recency Bias)                      │
                │  7. MovieRanker.rank_movies_stage2: History boost + Multi-feature Neg Penalties  │
                │  8. DiversityFilter: Genre and director frequency limits                         │
                │  9. Single Final Recommendation (top_k=1): #1 Pick + AI Verdict & Rationale      │
                │ 10. Rejection Replacement Engine: Skip → Learn Dislikes → Instant Fresh Pick    │
                └──────────────┬─────────────────────────┬─────────────────────────┬───────────────┘
                               │                         │                         │
                               ▼                         ▼                         ▼
                        ┌────────────┐            ┌──────────────┐          ┌────────────────────┐
                        │ TMDB API   │            │ SQLite Corpus│          │ LLM (Kado OpenAI-  │
                        │ (Live)     │            │ movie_       │          │ compatible /       │
                        │ Discover / │            │ features.db  │          │ DeepSeek fallback) │
                        │ Providers  │            │ (87,425 recs)│          │ Verdict & Reasons  │
                        └────────────┘            └──────────────┘          └────────────────────┘
```

**Key architectural principles:**
- **Explicit hard constraints vs. soft semantic preferences:** TMDB `discover` is called only with parameters that represent explicit hard constraints (`with_cast`, `with_crew`, `primary_release_year`, `primary_release_date`, `with_runtime`, `language`, `region`, `with_watch_providers`, `include_adult`, and explicit genre requirements `with_genres` / `without_genres`). Inferred/soft concepts (such as "dark", "mind-bending", "atmospheric", "space") are omitted from TMDB API queries and evaluated downstream via dense embeddings and TF-IDF matching.
- **Authoritative SQLite membership:** Any candidate returned by TMDB that is not indexed in `movie_features.db` is strictly excluded before ranking, preventing semantic drift and runtime latency spikes.
- **Age-relative popularity (anti-dominance):** Lifetime vote volume is normalized against release-year cohorts ($Y \pm 3$ / $Y \pm 5$ years) using $\log(1 + \text{count})$ before computing percentile ranks. Popularity accounts for exactly 10% of Stage 1 ranking.

---

## 3. Tech Stack

| Area | Libraries / Services |
|---|---|
| Language / runtime | Python 3.10+, `asyncio`, `aiohttp` (TMDB), `uvicorn` + `FastAPI` (UI & MCP server), `sqlite3` |
| Data | `movie_features.db` (SQLite 3, 87,425 canonical movies derived from MovieLens 32M + TMDB), `pandas`, `numpy`, `scipy` |
| Classical features | `scikit-learn` `CountVectorizer` + `TfidfVectorizer` (1–2 grams, 5000 feats, `min_df=2`, `max_df=0.95`, english stopwords) |
| Semantic | `sentence-transformers` (`multi-qa-mpnet-base-dot-v1`, 768-dim L2-normalized float32), `torch`, `nltk` WordNet synonyms |
| LLM | OpenAI-compatible client pointed at KADO (`https://awesome.kado.so/openai/v1`, model `kado`) with backwards-compatible fallback |
| Memory / State | Dual-branch SQLite event ledger (`data/memory.db`) with 7-day half-life decay, asymptotic updates, and negative preference learning |
| Config / env | `pyyaml` (`config.yaml`), `python-dotenv` (`.env`) with `KADO_API_KEY`, `KADO_BASE_URL`, `KADO_MODEL`, `TMDB_API_KEY` |
| Protocols | `mcp` types (`Tool`, `TextContent`, `CallToolResult`), A2A protocol dataclasses |
| Tests | `pytest`, `pytest-asyncio`, `unittest.mock` |

---

## 4. Repository Layout

```
TechQuest/
├── main.py                    # CLI entry point: setup-db, precompute, ui, recommend, mcp, a2a
├── mcp_server.py              # FastAPI MCP server (:8000): 4 tools for agent integration
├── a2a_protocol.py            # A2A agent protocol implementation (3 capabilities)
├── movie_features.db          # Authoritative SQLite database (87,425 canonical TMDB movies)
├── config.yaml                # Tunable hyperparameters (ranking weights, thresholds, stage sizes)
├── requirements.txt           # Core Python dependencies
├── .env / .env.example        # KADO_API_KEY, KADO_BASE_URL, KADO_MODEL, TMDB_API_KEY
├── hist.json                  # Sample watch history for manual CLI testing
├── database_readme.md         # Database schema and canonical deduplication reference
├── ARCHITECTURE.md            # Detailed system architecture, mathematical formulas, and flowcharts
├── src/
│   ├── ui/
│   │   ├── web_app.py         # FastAPI UI server on :8080 with user persistence & feedback loop
│   │   └── static/            # Frontend assets (HTML, CSS, JS)
│   ├── db/
│   │   └── connection.py      # SQLite MovieDatabase manager & canonical row loader
│   ├── tmdb/
│   │   └── client.py          # Async TMDBClient (discover, providers, details, credits)
│   ├── features/
│   │   ├── matrix_builder.py  # FeatureMatrixBuilder (content_text, TF-IDF, 768-d MPNet embeddings)
│   │   ├── semantic_search.py # SemanticKeywordSearch + HybridKeywordMatcher
│   │   └── preference_extractor.py # Watch history → UserPreferenceProfile
│   ├── memory/
│   │   ├── service.py         # UserMemoryService orchestrator
│   │   ├── engine.py          # MemoryEngine (time decay, asymptotic additions, beta balancing)
│   │   └── storage.py         # SQLite memory ledger storage
│   ├── ranking/
│   │   └── ranker.py          # MovieRanker (Stage 1 linear scoring, Stage 2 personalization, DiversityFilter)
│   ├── llm/
│   │   ├── reasoning.py       # DeepSeekReasoner (Kado LLM) + ReasoningTemplateEngine
│   │   └── nl_feature_extractor.py # NL query parser (hard vs soft constraint separation)
│   └── pipeline/
│       └── recommender.py     # MovieRecommender pipeline orchestrator
├── tests/
│   ├── test_recommender.py    # Unit tests: extractor, ranker, semantic search, diversity
│   ├── test_database_integration.py # Unit tests: DB integrity, canonical counts, candidate filtering
│   ├── test_memory.py         # Unit tests: decay, beta transition, negative penalties
│   ├── test_feature_extraction.py # Unit tests: constraint separation, genre normalization
│   ├── test_cleanup_regressions.py # Regression tests: UI persistence, providers, languages
│   └── test_age_relative_popularity.py # Unit tests: age-relative popularity, cohort statistics
└── data/
    ├── memory.db              # SQLite user memory event ledger
    ├── movie_db_manifest.json # Canonical corpus checksums and statistics
    └── processed/             # Precomputed artifacts:
                               #   movie_embeddings.npy (87.4k × 768), movie_id_map.json,
                               #   count_vectorizer.pkl, tfidf_vectorizer.pkl, feature_matrix.pkl
```

---

## 5. Data & Offline Artifacts

**Authoritative Corpus (`movie_features.db`):** 87,585 records derived from MovieLens 32M and TMDB, deduplicated deterministically to **87,425 canonical TMDB movies** (124 null TMDB IDs excluded; 36 duplicate TMDB ID groups consolidated with user tag unioning).

**Precomputation (`python main.py precompute`):**
1. Loads canonical rows from `movie_features.db`, unioning distinct MovieLens user tags and building the unified `content_text`.
2. Fits `CountVectorizer` and `TfidfVectorizer` (5,000 features, 1–2 grams, English stopwords).
3. Encodes texts with `multi-qa-mpnet-base-dot-v1` into 768-dimensional $L_2$-normalized `float32` vectors with atomic checkpointing in `data/processed/checkpoints/`.
4. Saves `movie_embeddings.npy` (87,425 × 768, ~268.5 MB), `movie_id_map.json`, `count_vectorizer.pkl`, `tfidf_vectorizer.pkl`, `feature_matrix.pkl`, and `precompute_manifest.json`.

*Note: The old 4,803-movie legacy CSV files and `semantic_index.pkl` are retired. Runtime similarity uses fast ~11.5 ms on-demand dot-product matrix multiplication.*

---

## 6. Configuration

`config.yaml` controls runtime parameters:

```yaml
tmdb:
  base_url: "https://api.themoviedb.org/3"
  image_base_url: "https://image.tmdb.org/t/p/w500"

kado:
  base_url: "https://awesome.kado.so/openai/v1"
  model: "kado"

recommendation:
  top_k_movies: 100
  final_recommendations: 1
  imdb_weight: 0.30
  tfidf_weight: 0.25
  semantic_weight: 0.35
  popularity_weight: 0.10
  stage1_candidates: 100
  stage1_max_tmdb_results: 200
  history_boost_factor: 0.30
  max_same_genre: 3
  max_same_director: 2
```

---

## 7. Core Recommendation Pipeline (End-to-End Workflow)

```
watch_history + NL query + subscriptions/region/language
  │
  ▼
[1] UserMemoryService loads user profile and negative preference penalties.
  │
  ▼
[2] NLFeatureExtractor parses query:
  │   • Explicit hard constraints → TMDB discovery parameters.
  │   • Soft concepts (mood, plot, themes) → semantic search query.
  │
  ▼
[3] TMDBClient discovers live candidates matching explicit filters.
  │   • SQLite Membership Filter: excludes any movie absent from movie_features.db.
  │   • Top-up: fetches additional TMDB pages (up to 5) if candidates < 50.
  │   • Fallback: queries SQLite movie_features.db directly if TMDB is offline.
  │
  ▼
[4] Query Vectorization: matrix_builder.transform_query(query_text)
  │   → (count_vec, tfidf_vec, 768-d MPNet embedding).
  │
  ▼
[5] Stage 1 Ranking: MovieRanker.rank_movies_stage1()
  │   stage1_score = 0.30 * rating_quality
  │                + 0.25 * tfidf_relevance
  │                + 0.35 * semantic_relevance
  │                + 0.10 * age_adjusted_popularity
  │
  │   Where age_adjusted_popularity:
  │     • 0.70 * cohort_percentile(log1p(votes) in Y ± 3 / Y ± 5 cohort)
  │     • + 0.30 * (log1p(votes) / log1p(global_max_votes))
  │   (Exact sum = 1.00; no recency bias, no double-counting).
  │
  ▼
[6] Stage 2 Ranking: MovieRanker.rank_movies_stage2()
  │   • History boost: + 0.30 * content_sim + 0.15 * keyword_overlap.
  │   • Blended negative penalty: penalizes disliked genres, tags, and themes.
  │   • final_score = stage1_score * (1 + boost) * penalty.
  │
  ▼
[7] DiversityFilter: caps max same genre (3) and max same director (2).
  │
  ▼
[8] Single Top Pick & Verdict:
  │   • Delivers #1 recommendation with AI rationale via Kado LLM or templates.
  │   • User can accept (watch/watchlist/rate) or skip ("Not For Me"), feeding
  │     the negative preference learner for instant re-roll.
```

---

## 8. Module Deep-Dives

### 8.1 `src/ranking/ranker.py` — Scoring & Diversity
- **Exact Stage 1 Linear Weights:**
  - `RATING_WEIGHT = 0.30`
  - `TFIDF_WEIGHT = 0.25`
  - `SEMANTIC_WEIGHT = 0.35`
  - `POPULARITY_WEIGHT = 0.10`
- **Age-Relative Popularity Algorithm:**
  - Uses release year extracted from `release_date`, `release_year`, or `(YYYY)` in title.
  - Precomputes cohort index over all 87,425 SQLite movies at startup ($O(1)$ lookup during scoring).
  - Primary cohort: $Y \pm 3$ years; secondary cohort: $Y \pm 5$ years (used if cohort $< 20$ items).
  - Calculates percentile rank of $\log(1 + \text{rating\_count})$ within the cohort.
  - Combines: $0.70 \times \text{cohort\_percentile} + 0.30 \times \text{global\_popularity}$.
  - Fallback: missing year or sparse cohort falls back safely to global log popularity; null/zero vote count returns 0.0 deterministically.
- **Stage 2 Personalization:** Computes history content similarity and keyword overlap boosts, scaled by multi-feature negative preference penalties.
- **DiversityFilter:** Greedy anti-clustering filter enforcing $\le 3$ same genre and $\le 2$ same director.

### 8.2 `src/llm/nl_feature_extractor.py` — Constraint Separation
- Distinguishes explicit hard constraints (`explicit_genres`, `actors`, `directors`, `release_year_min/max`, `min_runtime_minutes`, `original_language`, `include_adult`) from soft semantic preferences (`mood`, `theme_or_plot`, `reference_movies`, `semantic_search_query`).
- Generates TMDB `/discover/movie` parameters using ONLY explicit constraints.
- Resolves actor and director names to numeric TMDB Person IDs via `search_person()`.
- Normalizes colloquial genre aliases (`sci-fi` $\to$ Science Fiction, `romcom` $\to$ Comedy, `spooky` $\to$ Horror).

### 8.3 `src/memory/` — Multi-Dimensional Memory Subsystem
- `UserMemoryService` manages user profiles stored in `data/memory.db`.
- Short-term memory decays exponentially with a 7-day half-life ($\lambda \approx 1.1458 \times 10^{-6} \text{ s}^{-1}$).
- Long-term memory aggregates preferences asymptotically ($P \leftarrow P + (1 - P) \cdot w$) with Bayesian confidence shrinkage.
- Smooth Beta dynamically balances short-term exploration vs. long-term exploitation.
- Dislike learning: "Not For Me" skips penalize the features of the rejected movie to avoid repeated similar recommendations.

### 8.4 `src/ui/web_app.py` — Web Application
- Single-movie hero showcase with backdrop, poster, match percentage, and AI verdict.
- User ID persistence across browser refreshes via `localStorage` (defaults to `demo`).
- Interactive feedback buttons: "Watched", "Add to Watchlist", "Rate (1–5 Stars)", and "Not For Me / Skip".
- Factual streaming provider badges (no fabricated availability).
- Refresh does not trigger an accidental recommendation.

---

## 9. Interfaces: UI, CLI, MCP, A2A

### 9.1 Web UI (`python main.py ui`)
Starts FastAPI web server on `http://127.0.0.1:8080`.
- Single hero recommendation card with natural language input.
- Real-time skip and re-roll loop with negative feedback adaptation.
- Taste dashboard and interaction history ledger.

### 9.2 CLI (`python main.py`)
```bash
# Setup and verify database
python main.py setup-db

# Run one-time offline precomputation
python main.py precompute

# Interactive recommendations
python main.py recommend --user-id demo --natural-language "great thriller" --top-k 5

# Rejection interaction
python main.py reject --user-id demo --movie-id 77
```

### 9.3 MCP Server (`python main.py mcp`)
Runs FastAPI server on `http://127.0.0.1:8000` exposing 4 MCP tools:
- `search_movies`: Search TMDB/SQLite catalog with structured filters.
- `get_recommendations`: Full two-stage recommendation pipeline with NL query support.
- `get_movie_details`: Fetch movie metadata, credits, providers, and similar films.
- `analyze_user_preferences`: Summarize user taste profile and history.

### 9.4 A2A Agent (`python main.py a2a`)
Implements Agent-to-Agent protocol (`MovieRecommenderA2AAgent`, ID `movie-recommender-001`) with 3 capabilities.

---

## 10. Setup From Scratch

### 10.1 Quick Start (Recommended — Download Prebuilt Assets)

1. **Clone & Virtual Environment Setup:**
   ```bash
   git clone https://github.com/LiliesInLakes/techquest.git
   cd techquest

   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   pip install fastapi uvicorn aiohttp mcp pydantic pytest pytest-asyncio
   ```

2. **Download & Verify Prebuilt Runtime Assets:**
   ```bash
   # Downloads movie_features.db, 768-d MPNet embeddings, & TF-IDF matrices from GitHub Release:
   python scripts/setup_assets.py

   # Or verify integrity of locally placed assets:
   python scripts/setup_assets.py --verify-only
   ```

3. **Configure API Keys (`.env`):**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your `TMDB_API_KEY`, `KADO_API_KEY`, `KADO_BASE_URL`, and `KADO_MODEL`.

4. **Launch Web UI:**
   ```bash
   python main.py ui
   ```
   Open `http://127.0.0.1:8080` in your browser.

---

### 10.2 Alternative: Rebuild & Precompute From Scratch (Advanced)

If you prefer to rebuild all artifacts from the database:
1. Validate database integrity:
   ```bash
   python main.py setup-db
   ```
2. Run offline precomputation (generates 87.4k 768-d embeddings and TF-IDF matrices):
   ```bash
   python main.py precompute
   ```
   *(Or run [`colab_precompute.ipynb`](file:///c:/Users/lenovo/Desktop/kado_tech/techquest/colab_precompute.ipynb) on Google Colab GPU for fast build in ~10 mins).*


---

## 11. Usage Cookbook

```bash
# Get recommendations with natural language and subscriptions
python main.py recommend --user-id u1 --watch-history hist.json --top-k 5 \
  --natural-language "dark psychological thriller with intense pacing" \
  --subscriptions netflix --region US

# Run MCP server
python main.py mcp
```

---

## 12. Testing

Run the comprehensive pytest suite:
```bash
pytest tests/ -v
```

Test suites include:
- `test_age_relative_popularity.py`: Exact Stage 1 weights (0.30, 0.25, 0.35, 0.10 = 1.00), cohort percentile logic, missing year fallback, null/zero counts, no double counting.
- `test_cleanup_regressions.py`: UI user persistence, factual providers, language handling, genre normalization.
- `test_memory.py`: Asymptotic updates, exponential decay, Smooth Beta, blended negative penalties.
- `test_feature_extraction.py`: Hard vs soft constraint separation, TMDB discover parameters.
- `test_database_integration.py`: SQLite 87,425 canonical movies, tag unioning, candidate exclusion filtering.
- `test_recommender.py`: Core orchestrator and ranking pipelines.

---

## 13. Fallback & Offline Behavior

| Component | Missing Key / Failure | Fallback Behavior |
|---|---|---|
| **TMDB API** | No `TMDB_API_KEY` or network failure | Queries `movie_features.db` directly with SQL filters on hard constraints; sorts by `rating_count DESC, avg_rating DESC`. |
| **LLM Provider** | No `KADO_API_KEY` or API error | `NLFeatureExtractor` falls back to regex parser (`confidence=0.5`); `DeepSeekReasoner` falls back to prioritized rule templates (`ReasoningTemplateEngine`). |
| **Release Year** | Missing in candidate metadata | Safely falls back to global log popularity without fabricating a release year. |
| **Rating Count** | NULL or 0 votes | Safely assigns 0.0 popularity score deterministically. |
| **User Profile** | Empty watch history | Zeroed profile; ranking is driven by query relevance, semantic match, and quality. |

---

## 14. How to Extend It

- **Adjust Ranking Weights:** Modify `config.yaml` (`imdb_weight`, `tfidf_weight`, `semantic_weight`, `popularity_weight`).
- **Update Memory Decay:** Modify $\lambda$ in `src/memory/engine.py`.
- **Add MCP Tools / A2A Capabilities:** Add handler methods in `mcp_server.py` and `a2a_protocol.py` delegating to `MovieRecommender`.

---

## 15. Glossary & Limitations

- **Canonical Corpus:** 87,425 unique TMDB movies deduplicated from MovieLens 32M and TMDB in `movie_features.db`.
- **Age-Relative Popularity:** Percentile ranking of $\log(1 + \text{votes})$ within release-year cohorts ($Y \pm 3$ / $Y \pm 5$), preventing classic movies with decades of accumulated votes from overwhelming modern films.
- **Model-Derived Semantic Similarity:** Cosine similarity from `multi-qa-mpnet-base-dot-v1` (768-d), representing conceptual proximity rather than a literal percentage.
- **Known Limitations:**
  - MovieLens dataset temporal coverage reaches ~2023; very recent 2024–2026 films exist in TMDB but will only be recommended if indexed in the canonical database.
  - First-run model initialization requires loading the 768-d SentenceTransformer and embedding matrix (~2–3 seconds on CPU).
  - External LLM API calls add 500–1500 ms latency per recommendation request (mitigated by asynchronous execution and template fallback).
