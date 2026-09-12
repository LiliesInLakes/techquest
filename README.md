# Movie Recommender Agent — In-Depth Guide

> **One-line pitch:** An agentic, B2B2C movie-recommendation system built for **hospitality & airline partners** (seatback screens, hotel TVs, mobile apps). It blends TMDB live discovery, offline feature engineering, semantic search, multi-stage ranking, LLM explanations, and a full business-intelligence layer (partner rules, sessions, churn, actions, analytics events). It can be driven three ways: **CLI**, **MCP server** (Model Context Protocol), or **A2A agent** (Agent-to-Agent protocol).

If you have never seen this codebase before, read this file top-to-bottom. By the end you should be able to set it up, run it, trace a request through every layer, and extend it.


(tldr at end)
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
9. [Business-Intelligence Layer](#9-business-intelligence-layer)
10. [Interfaces: CLI, MCP, A2A](#10-interfaces-cli-mcp-a2a)
11. [Setup From Scratch](#11-setup-from-scratch)
12. [Usage Cookbook](#12-usage-cookbook)
13. [Testing](#13-testing)
14. [Fallback & Offline Behavior](#14-fallback--offline-behavior)
15. [How to Extend It](#15-how-to-extend-it)
16. [Glossary](#16-glossary)

---

## 1. What This System Does

**User-facing:** given a user ID, watch history (`[{movie_id, rating}]`), subscriptions, region/language, and optionally a natural-language request like *"funny 90s space comedy with Tom Hanks"*, return a ranked top-K list of movies, each with a natural-language *reason* ("Because you love…").

**Business-facing (the differentiator):** the same flow can run *in a partner context* (`partner_id` = e.g. `airline_default`, `hotel_default`, `streaming_default`):

- Filter/rank per partner content rules (allowed ratings, max runtime, license regions, competitor excludes, offline-only catalogs).
- Track a **session** (device, party type/size, flight duration, hotel stay, subscription tier, bandwidth, accessibility).
- Score **churn risk** and suggest retention actions.
- **Trigger business actions** (bundle upgrade offer, re-engagement nudge, personalized playlist, …).
- **Ingest analytics events** (plays, skips, impressions, clicks, searches, ratings, …) through a queued background pipeline.

There are two recommender classes for this reason:

| Class | File | Purpose |
|---|---|---|
| `MovieRecommender` | `src/pipeline/recommender.py` | Pure recommendation: profile → candidates → rank → explain. No business side-effects. |
| `BusinessMovieRecommender(MovieRecommender)` | `src/pipeline/recommender.py` | Everything above **plus** rules engine, sessions, churn, actions, event ingestion. Used by `recommend-biz`, churn/action/event CLIs, and the business MCP/A2A tools. |

---

## 2. High-Level Architecture

```
                    ┌──────────────┐      ┌──────────────┐
                    │  CLI (main.py)│      │ MCP server   │      ┌─────────────┐
 User / Partner ───▶│  7 subcommands│      │ (FastAPI :8000)│────▶│ A2A agent   │
  or external agent │               │      │ 11 tools     │      │ 10 caps     │
                    └──────┬────────┘      └──────┬───────┘      └──────┬──────┘
                           │                      │                     │
                           ▼                      ▼                     ▼
              ┌──────────────────────────────────────────────────────────────┐
              │              Pipeline: MovieRecommender /                     │
              │              BusinessMovieRecommender                          │
              │  1. PreferenceExtractor → user profile                        │
              │  2. NLFeatureExtractor  → structured features from NL query   │
              │  3. TMDBClient (or local pool) → Stage-1 candidates           │
              │  4. MovieRanker.stage1 → query-similarity ranking             │
              │  5. MovieRanker.stage2 → watch-history boost                  │
              │  6. DiversityFilter → de-duplicate genres/directors           │
              │  7. DeepSeekReasoner / TemplateEngine → explanations          │
              │  (+ business: RulesEngine, SessionManager, Churn, Actions,    │
              │   Events — only on the Business path)                         │
              └──────┬───────────────┬───────────────────┬───────────────────┘
                     │               │                   │
                     ▼               ▼                   ▼
              ┌────────────┐ ┌──────────────┐ ┌────────────────────┐
              │ TMDB API   │ │ Offline pool │ │ LLM (DeepSeek,      │
              │ (live)     │ │ data/processed│ │ OpenAI-compatible)  │
              │ discover / │ │ feature_matrix│ │ reasoning + NL      │
              │ search /   │ │ .pkl, TF-IDF, │ │ parsing             │
              │ details    │ │ embeddings   │ │                     │
              └────────────┘ └──────────────┘ └────────────────────┘
```

**Key design decisions:**

- **Stage-1 is loose, ranking is strict.** TMDB `discover` is called with only the *primary* genre + `popularity.desc` sorting. Exclusions (e.g. "not horror") are applied in Python (`_passes_exclusions`), and real ordering happens in the ranker. This keeps recall high.
- **Everything degrades gracefully.** No TMDB key → local `data/processed` pool. No DeepSeek key → regex NL parsing + template explanations. TMDB timeout → automatic local-pool fallback.
- **Business logic is layered on, not baked in.** The base recommender never touches partner state; `BusinessMovieRecommender` wraps it with rules/sessions/churn/actions/events.

---

## 3. Tech Stack

| Area | Libraries / Services |
|---|---|
| Language / runtime | Python 3, `asyncio`, `aiohttp` (TMDB), `uvicorn` + `FastAPI` (MCP server) |
| Data | `pandas`, `numpy`, `scipy` |
| Classical features | `scikit-learn` `CountVectorizer` + `TfidfVectorizer` (1–2 grams, 5000 feats, `min_df=2`, `max_df=0.95`, english stopwords) |
| Semantic | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim), `torch`, `nltk` WordNet synonyms |
| LLM | `openai==1.3.7` client pointed at DeepSeek (`https://api.deepseek.com/v1`, model `deepseek-chat`) |
| Config / env | `pyyaml` (`config.yaml`), `python-dotenv` (`.env`) |
| Protocols | `mcp` types (`Tool`, `TextContent`, `CallToolResult`), hand-rolled A2A dataclasses |
| Tests | `pytest`, `pytest-asyncio` style markers, `unittest.mock` |

`requirements.txt` pins: `requests, pandas, numpy, scikit-learn, scipy, pyyaml, python-dotenv, openai, nltk, sentence-transformers, torch, tqdm`. Note: `fastapi, uvicorn, aiohttp, mcp, pytest` are imported by the code but **not** in `requirements.txt` — install them manually (see Setup).

---

## 4. Repository Layout

```
TechQuest/
├── main.py                    # CLI entry point: 7 subcommands (mcp, a2a, precompute,
│                              #   recommend, recommend-biz, assess-churn, trigger-action, ingest-events)
├── mcp_server.py              # FastAPI app (:8000): 11 MCP tools, Pydantic param models,
│                              #   GET /mcp/tools + POST /mcp/tools/{tool_name}
├── a2a_protocol.py            # A2A message/capability dataclasses, abstract A2AAgent,
│                              #   MovieRecommenderA2AAgent (10 capabilities), A2AClient
├── config.yaml                # All tunable knobs (TMDB/DeepSeek endpoints, ranking weights,
│                              #   stage sizes, dataset paths)
├── requirements.txt           # Pinned deps (see caveat above)
├── .env / .env.example        # TMDB_API_KEY, DEEPSEEK_API_KEY (never commit .env)
├── hist.json                  # Sample watch history used for manual testing
├── src/
│   ├── tmdb/client.py         # Async TMDBClient + Movie dataclass (discover/search/details/
│   │                          #   popular/top-rated/keywords/credits/providers)
│   ├── features/
│   │   ├── matrix_builder.py  # Offline: CSVs → combined text → count/TF-IDF/embeddings →
│   │   │                      #   data/processed/*.pkl. Also transform_query() at serve time.
│   │   ├── semantic_search.py # SemanticKeywordSearch (embeddings + WordNet) + HybridKeywordMatcher
│   │   └── preference_extractor.py # Watch history → UserPreferenceProfile (genres/keywords/
│   │                          #   directors/cast/decades/languages, TF-IDF weights, feature vector)
│   ├── ranking/
│   │   └── ranker.py          # MovieRanker (stage1/stage2 scoring) + DiversityFilter
│   ├── llm/
│   │   ├── reasoning.py       # DeepSeekReasoner (per-movie explanations) + ReasoningTemplateEngine fallback
│   │   └── nl_feature_extractor.py # NL query → ExtractedFeatures (genres/keywords/actors/
│   │                          #   directors/decades/mood/rating/exclusions) via DeepSeek or regex fallback
│   ├── pipeline/
│   │   └── recommender.py     # MovieRecommender + BusinessMovieRecommender (orchestrates everything)
│   └── business/
│       ├── rules_engine.py    # PartnerConfig, BusinessRule, RulesEngine + 3 default partners
│       ├── session_context.py # SessionContext, UserSessionProfile, SessionManager (in-memory, TTL)
│       ├── churn_risk.py      # ChurnFeatures, rule-weighted ChurnRiskModel, ChurnRiskService
│       ├── actions.py         # BusinessAction, executors, BusinessActionEngine
│       └── events.py          # AnalyticsEvent/EventType, queued EventIngestion pipeline
├── tests/
│   └── test_recommender.py    # Unit tests: extractor, ranker, semantic search, templates, diversity, pipeline helpers
└── data/
    ├── tmdb_5000_movies.csv / tmdb_5000_credits.csv  # Offline corpus (TMDB 5000)
    ├── genome-tags.csv / genome-scores.csv / links.csv  # MovieLens tag relevance + TMDB↔MovieLens ID map
    ├── processed/             # Build output: feature_matrix.pkl, semantic_index.pkl,
    │                          #   count_vectorizer.pkl, tfidf_vectorizer.pkl
    ├── analytics/ archive/    # Event dumps / misc (not on hot path)
    └── processed/             # (cached matrices — required at runtime)
```

If you only have time to read five files: `src/pipeline/recommender.py`, `src/ranking/ranker.py`, `src/features/matrix_builder.py`, `src/llm/nl_feature_extractor.py`, `src/business/rules_engine.py`.

---

## 5. Data & Offline Artifacts

**Inputs (`data/`):** TMDB-5000 movies+credits, MovieLens `genome-tags.csv` (tag names), `genome-scores.csv` (movie↔tag relevance), `links.csv` (TMDB↔MovieLens ID map).

**Offline build (`python main.py precompute` → `FeatureMatrixBuilder.build_and_save()`):**

1. `load_tmdb_5000()` — merge movies+credits on `id` (dropping duplicate `title`), parse JSON-ish `keywords/genres/cast/crew`, derive `director` + `top_cast` (first 5).
2. `build_combined_text()` — map TMDB ids → MovieLens ids via `links.csv`, take top-20 tags/movie by relevance, join tag names, then build one `combined_features` string per movie: `overview + keywords + genres + director + top_cast + movielens_tags`.
3. `build_matrices()` — fit `CountVectorizer`, `TfidfVectorizer`, and `SentenceTransformer.encode()` over all combined texts; store per-movie metadata (`title/genres/keywords/overview/vote_average/vote_count/popularity/release_date/director/cast/movielens_tags`) in `FeatureMatrix`.
4. `build_semantic_index()` — full cosine-similarity matrix over semantic embeddings; keep top-50 neighbors per movie (`semantic_index.pkl`, used by `get_movie_details().similar_movies`).
5. Pickle all four artifacts to `data/processed/`.

**At serve time** `MovieRecommender.initialize()` tries `FeatureMatrixBuilder.load()` (fast path); only if pickles are missing does it rebuild (slow, needs CSVs + downloads the transformer model). It then builds the keyword search index from all metadata genres+keywords and wires `preference_extractor → ranker → reasoner → nl_extractor`.

---

## 6. Configuration

`config.yaml` (env placeholders `${TMDB_API_KEY}` / `${DEEPSEEK_API_KEY}` are resolved via `.env`, not YAML interpolation — the code reads `os.getenv` first):

```yaml
tmdb: { base_url, image_base_url }
deepseek: { base_url: https://api.deepseek.com/v1, model: deepseek-chat }
recommendation:
  top_k_movies: 100
  final_recommendations: 10
  imdb_weight: 0.3
  tfidf_weight: 0.2
  watch_history_weight: 0.5
  semantic_similarity_threshold: 0.7   # note: ranker.py currently reuses this as semantic_weight
  semantic_weight: 0.3
  stage1_candidates: 100               # how many stage-1 winners advance
  stage1_max_tmdb_results: 200         # cap on raw TMDB pool
  history_boost_factor: 0.3
  max_same_genre: 3                    # DiversityFilter caps (ranker defaults: genre 3, director 2)
  max_same_director: 2
data: { tmdb_5000_path, tmdb_5000_credits_path, movielens_tags_path, movielens_relevance_path }
```

`RankingConfig` defaults add: `content_weight=0.4`, `keyword_overlap_weight=0.3`, `min_vote_count=100`, `popularity_boost=0.1`.

---

## 7. Core Recommendation Pipeline (End-to-End Workflow)

This is the heart of the system. Trace `MovieRecommender.get_recommendations()`:

```
watch_history + NL query + subscriptions/region/language
  │
  ▼
[1] Build ratings dict {movie_id: rating} from history entries
  │   (accepts both `movie_id` and `id` keys; ignores rating 0)
  ▼
[2] PreferenceExtractor.extract_preferences(history, ratings, feature_matrix)
  │   → UserPreferenceProfile: top genres/keywords/directors/cast (rating-weighted
  │     counters, normalized), TF-IDF-style keyword/genre weights (tf × log(N/df)),
  │     avg/std rating, rating histogram, preferred decades/languages, watch_count,
  │     and a rating-weighted mean CountVectorizer feature_vector.
  ▼
[3] NLFeatureExtractor.extract(natural_language, context)   [if NL query present]
  │   → ExtractedFeatures {genres, keywords, actors, directors, decades, mood,
  │     min_rating, exclude_genres/keywords, query_text, confidence}
  │   → map_to_vocabulary(): snap keywords into the CountVectorizer vocab via
  │     semantic nearest-neighbor (threshold 0.7), else keep lowercased token.
  ▼
[4] Stage-1 candidate fetch (_get_candidate_movies_stage1)
  │   • WITH NL features: TMDB discover(genre=<first genre>, popularity.desc,
  │     min_rating, watch providers) → Python exclusion filter
  │     (_passes_exclusions: drop excluded genres / overview keyword hits).
  │     If <50 survive, top up with get_popular_movies(). Cap at stage1_candidates.
  │     Any TMDB failure / no key → _get_local_candidates() (entire feature-matrix
  │     pool minus exclusions, sorted by vote_average+popularity).
  │   • WITHOUT NL features: per top-5 profile genre → discover(vote_average.desc,
  │     min_rating=6.0) × 30 + 50 populars → dedupe → cap 200. Fallback to local pool.
  ▼
[5] Vectorize query: matrix_builder.transform_query(query_text)
  │   query_text = NL query_text OR genres+keywords+mood built from profile.
  │   → (count_vec, tfidf_vec, MiniLM embedding)
  ▼
[6] MovieRanker.rank_movies_stage1(candidates, query_embedding, query_tfidf)
  │   Per movie: imdb (vote_avg/10, halved if votes < 100) + tfidf cosine +
  │   semantic cosine + popularity ((pop/1000 + votes/10000)/2). NO history yet.
  │   Keep top stage1_candidates (default 100).
  ▼
[7] MovieRanker.rank_movies_stage2(stage1_top, profile, extracted, matrix, boost=0.3)
  │   Only inside the NL genre cluster (or all if no NL genres): compute
  │   content_similarity (profile feature_vector ↔ movie count vector, cosine) and
  │   keyword_overlap (profile↔movie genre/keyword overlap + TF-IDF weights).
  │   new_score = stage1_score × (1 + 0.3×content + 0.15×overlap).
  ▼
[8] DiversityFilter.filter() — greedy pass in rank order; skip a movie if any of
  │   its genres already appeared max_same_genre (3) times or any director
  │   max_same_director (2) times.
  ▼
[9] Explanation (_generate_reasonings): DeepSeekReasoner.generate_batch_reasoning()
  │   (async, one chat call per movie: system prompt + movie/profile/factor-matched
  │   prompt → 2–4 sentence "Because you…" text) or ReasoningTemplateEngine fallback
  │   (genre_match > watch_history > high_rating > generic).
  ▼
Return { recommendations: [{rank, movie, reasoning, score}],
         user_profile: {top_genres, top_keywords, avg_rating, watch_count},
         extracted_features }
```

`search_movies()` (no history) is simpler: TMDB `discover` (or local genre/rating filter) → trimmed dicts. `get_movie_details()` merges live TMDB details/credits/keywords/providers with local metadata + `similar_movies` from the semantic index.

---

## 8. Module Deep-Dives

### 8.1 `src/tmdb/client.py` — Live movie source
- `Movie` dataclass: id/title/overview/genres/dates/votes/popularity/language/poster/backdrop + lazy `keywords/credits/watch_providers`.
- `TMDBClient(api_key?, config_path)`: raises `ValueError` if key missing/unexpanded — this is what triggers local-pool mode upstream. Async context manager owns an `aiohttp` session; `_request()` injects `api_key`.
- Methods: `discover_movies()` (genre name → id via cached `/genre/movie/list`, supports rating/vote floors, providers, region), `search_movies()`, `get_movie_details()` (appends `credits,keywords,watch/providers`), `get_movie_keywords/credits`, `get_watch_providers()`, `get_popular_movies()`, `get_top_rated_movies()`. `_parse_movie()` maps `genre_ids`↔names using the cache (deliberately sync-safe; empty genres if cache cold).

### 8.2 `src/features/matrix_builder.py` — Offline brains
Covered in §5. Serve-time API is `transform_query(text) → (count, tfidf, embedding)` and classmethod `load()` returning `(builder, feature_matrix, semantic_index)`.

### 8.3 `src/features/semantic_search.py` — Synonym-aware matching
- `SemanticKeywordSearch`: MiniLM embeddings over the full keyword/genre vocabulary + cosine `find_similar_keywords()` above a threshold (0.65 at init, 0.7 in ranker wiring); WordNet `get_synonyms_wordnet()` with cache; `expand_query_with_synonyms()` unions tokens + WordNet synonyms + embedding neighbors; pickle `save/load`.
- `HybridKeywordMatcher`: pre-embeds all CountVectorizer feature names; `match_user_keywords()` maps each user keyword to top-50 vocab features above threshold; `get_feature_importance_for_keywords()` max-pools scores. Used to bridge free text → model vocabulary.

### 8.4 `src/features/preference_extractor.py` — Who is this user?
Rating-weighted counting (`weight = 0.3 + rating/10×0.7`, 0.5 if unrated) across genres/keywords/directors/cast/decades/languages; TF-IDF-style `keyword_weights/genre_weights`; rating stats + histogram; `compute_content_similarity()` and `compute_keyword_overlap_score()` consumed by stage-2 ranking. Empty history → zeroed `_empty_profile`.

### 8.5 `src/ranking/ranker.py` — Scoring
- `RankingConfig` + `RankedMovie` (all sub-scores + `reasoning_factors` dict passed to the LLM).
- `rank_movies()` (legacy single-pass combining imdb/tfidf/history/semantic/content/overlap+popularity boost) vs. the actually-used `rank_movies_stage1()` (query-only) + `rank_movies_stage2()` (history boost) + `get_top_k()` + `explain_ranking()`.
- Score details: IMDb halved below `min_vote_count`; TF-IDF/semantic are cosines vs. query vectors; popularity is mean of normalized popularity and vote count (plus a double-counted popularity term in stage-1 — known quirk, see code); final single-pass score weights content-combined vs. imdb/tfidf.
- `DiversityFilter(max_same_genre=3, max_same_director=2)`: greedy genre/director caps.

### 8.6 `src/llm/nl_feature_extractor.py` — NL → structured filters
- `ExtractedFeatures` dataclass + `to_dict()`.
- `NLFeatureExtractor`: DeepSeek JSON-mode parse (system prompt constrains to 19 TMDB genres, mood enum, rating/exclusion rules) with `_validate_genres/_validate_decades`, synonym map (`sci-fi→Science Fiction`, `romcom→Comedy`, …). **No key → regex fallback** (genre keyword scan, decade regex, stopword-stripped keywords, mood map, `min_rating=6.0` on "good/great/best"). `build_tmdb_discover_params()` emits loose `discover_movies()` kwargs; `map_to_vocabulary()` grounds keywords in model vocab.

### 8.7 `src/llm/reasoning.py` — NL explanations
- `DeepSeekReasoner`: per-movie prompt fusing movie facts, top profile items, matched genres/keywords, and thresholded factor phrases (`imdb>0.7`, `watch_history>0.5`, …); `temperature=0.7, max_tokens=500`; batch via `asyncio.gather`; LLM failure → `_fallback_reasoning()`; missing key at init → `ValueError` (pipeline catches → templates).
- `ReasoningTemplateEngine`: 7 canned templates with priority genre > history > rating > generic.

---

## 9. Business-Intelligence Layer

Only active on `BusinessMovieRecommender` (and the matching MCP/A2A tools + CLI subcommands). All state is **in-memory** (no DB) — sessions/profiles/actions/events reset on restart.

### 9.1 `src/business/rules_engine.py` — Partner rules
- `PartnerConfig`: `content_filters, ranking_weights, license_regions, partner_content_boost, competitor_exclude, max_runtime_minutes, allowed_content_ratings (default G/PG/PG-13), offline_required, session_limits` + ordered `BusinessRule`s (`condition` supports `eq/neq/gt/gte/lt/lte/in/not_in/contains` + priority + enable flag).
- `RulesEngine.evaluate()` merges global + partner rules → `{filters, weight_adjustments, actions, warnings}`; `apply_content_filters()` prunes candidates; `apply_weight_adjustments()` mutates `ranker.config` on the business path.
- `create_default_rules_engine()` ships **3 partners**: `airline_default` (family-safe, offline, runtime-capped for flights), `hotel_default` (broader, premium upsell), `streaming_default` (widest). Start here when adding a partner.

### 9.2 `src/business/session_context.py` — Who/where is watching?
- `SessionContext`: session/user/partner ids, `DeviceType` (seatback/mobile/tablet/laptop/tv/vr), `PartyType` (solo/couple/family/friends/business) + size, region/language/subs, flight (`duration/number`) or hotel (`room_type, check-in/out`) fields, tier, time/day, offline/bandwidth/screen/audio/accessibility, state machine (`active/paused/ended/expired`).
- `SessionManager` (thread-safe, 24h TTL): `create_session()`, `get_session()`, `end_session(session_id, watch_events)` (rolls events into `UserSessionProfile`: totals, genre/decade/device/party distributions, completion/skip/rewatch/discovery rates, churn/LTV estimates).

### 9.3 `src/business/churn_risk.py` — Will they leave?
- `ChurnFeatures`: recency (days since session/watch), frequency (sessions/watch-time windows), engagement (length/completion/skip/rewatch/discovery), taste diversity (HHI), search health (failure rate), recommendation CTR, subscription (tier/age/payment failures/support tickets), external (competitor app, devices).
- `ChurnRiskModel`: weighted sum over 7 groups (recency .25, frequency .20, engagement .15, satisfaction .15, diversity .10, subscription .10, external .05) with log/linear/inverse normalization; thresholds low .3 / medium .5 / high .7 / critical .85; emits `ChurnPrediction {risk_score, risk_level, top_risk_factors, recommended_actions (reengage/offer/winback/…), confidence, predicted_churn_date, estimated_ltv_loss}`. `ChurnRiskService` adds caching + `get_high_risk_users()`.

### 9.4 `src/business/actions.py` — Do something about it
- `BusinessAction {action_id, action_type (recommend/upgrade/discount/premium/reengage/playlist/…), partner/user, trigger (session_start/churn_risk_high/search_failed/…), priority, payload, reasoning, expires_at, executed/result}`.
- `BusinessActionEngine` + per-type `ActionExecutor`s (`RecommendMovieExecutor` queues a recommendation request; `BundleUpgradeExecutor` prices tier upgrades + benefits + discount; …). `create_default_action_engine(recommender?)` wires defaults. Business recommendations auto-create actions from `rules_engine.get_triggered_actions()` on `session_start`.

### 9.5 `src/business/events.py` — Analytics firehose
- `EventType` (~25 values: session/content/recommendation/search/watchlist/share/download/subscription/device/error).
- `EventIngestion`: bounded `Queue` (10k), batch (100) + interval (30s) background flush thread, pluggable `handlers`, `{received, flushed, dropped, batches, errors}` stats, `ingest/ingest_batch/start/stop`. `build_event()` factory + `create_ingestion_pipeline()` default. Business recommendations auto-emit `RECOMMENDATION_IMPRESSION` per returned movie.

---

## 10. Interfaces: CLI, MCP, A2A

### 10.1 CLI (`main.py`) — 7 subcommands
| Command | What it does | Key flags / example |
|---|---|---|
| `precompute` | Build `data/processed/*.pkl` | `python main.py precompute` |
| `mcp` | Serve FastAPI on `:8000` | `python main.py mcp` |
| `a2a` | Print agent card + capabilities (no server) | `python main.py a2a` |
| `recommend` | Base recommendations / search | `python main.py recommend --user-id u1 --genre Action --region US --subscriptions netflix disney --watch-history hist.json --top-k 10 --natural-language "funny 90s space comedy"` (omit `--watch-history` → plain `search_movies`) |
| `recommend-biz` | Same + partner/session/rules/churn/actions/events | Adds `--partner-id airline_default` (also `hotel_default`, `streaming_default`) |
| `assess-churn` | Churn score for a user+partner | `python main.py assess-churn --user-id u1 --partner-id airline_default [--user-data user.json]` |
| `trigger-action` | Create+execute one business action | `--action-type recommend_movie --trigger session_start --reasoning "..." --payload '{}' --priority medium` |
| `ingest-events` | Batch-ingest analytics JSON | `python main.py ingest-events --events-file events.json` |

Watch-history JSON: `[{"movie_id": 550, "rating": 9.0}, …]`. Sample: `hist.json`.

### 10.2 MCP server (`mcp_server.py`, FastAPI `:8000`)
- `GET /mcp/tools` → tool catalog; `POST /mcp/tools/{tool_name}` → dispatches to 11 tools (Pydantic-validated):
  - Recommendation: `search_movies`, `get_recommendations` (+`partner_id/session_id/natural_language`), `get_movie_details`, `analyze_user_preferences`.
  - Business/session: `create_session` (device/party/flight/hotel/tier/offline…), `end_session`, `get_churn_risk`, `trigger_business_action`, `ingest_events`, `apply_business_rules`, `get_partner_config`.
- Returns `CallToolResult(content=[TextContent(text=<json>)])`; unknown tool → 404, handler exception → 500. CORS fully open.

### 10.3 A2A (`a2a_protocol.py`)
- Primitives: `A2AMessageType` (task_request/response/status, discovery/capabilities, error), `TaskStatus`, `A2AMessage` (uuid, timestamp, correlation), `AgentCapability` (name/desc/input+output JSON-schema/tags), `AgentInfo`, abstract `A2AAgent` (discovery/capability handlers, `register_handler`, `handle_message` with error envelope).
- `MovieRecommenderA2AAgent` (`movie-recommender-001`): 10 capabilities mirroring MCP — `search_movies`, `get_recommendations`, `analyze_preferences`, `create_session`, `end_session`, `assess_churn_risk`, `trigger_business_action`, `ingest_events`, `apply_business_rules`, `get_partner_config` — each with JSON schemas + tags, routed in `_handle_task_request` to the same pipeline/business classes.
- `A2AClient`: `send_request(recipient, task, params, timeout=30)` (future keyed by correlation id — note: demo transport never delivers, so it will time out without a bus), `discover_agent()` stub, `handle_response()`.

---

## 11. Setup From Scratch

1. **Python + venv:** Python 3.10+ recommended.
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   # requirements.txt omits runtime imports — add:
   pip install fastapi uvicorn aiohttp mcp pydantic pytest pytest-asyncio
   # torch sentence-transformers pulls ~GBs; nltk wordnet downloads on first run.
   ```
2. **Keys:**
   ```bash
   cp .env.example .env
   # edit .env: TMDB_API_KEY=...  DEEPSEEK_API_KEY=...
   # Both optional (see §14), but live discovery + LLM quality need them.
   # Get TMDB key: https://www.themoviedb.org/settings/api
   ```
3. **Datasets:** place `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`, `genome-tags.csv`, `genome-scores.csv`, `links.csv` in `data/` (repo already includes them).
4. **Precompute (one-time, slow — embeddings + similarity matrix):**
   ```bash
   python main.py precompute
   # expect data/processed/{feature_matrix,semantic_index,count_vectorizer,tfidf_vectorizer}.pkl
   ```
5. **Run:**
   ```bash
   python main.py recommend --user-id demo --watch-history hist.json --top-k 5 \
     --natural-language "funny 90s space comedy" --subscriptions netflix
   python main.py recommend-biz --user-id demo --partner-id airline_default \
     --watch-history hist.json --top-k 5
   python main.py mcp            # → http://0.0.0.0:8000, try GET /mcp/tools
   python main.py a2a            # prints agent card
   pytest tests/ -v
   ```

---

## 12. Usage Cookbook

```bash
# Search only (no history → search_movies path)
python main.py recommend --user-id u1 --genre Action --region US --language en \
  --subscriptions netflix disney --top-k 10

# Personalized + NL
python main.py recommend --user-id u1 --watch-history hist.json --top-k 10 \
  --natural-language "mind-bending sci-fi with time travel, highly rated"

# Business path (adds session_id + business_actions_triggered to output)
python main.py recommend-biz --user-id u1 --partner-id hotel_default \
  --watch-history hist.json --natural-language "feel-good romantic comedy from the 2000s"

# Churn / actions / events
python main.py assess-churn --user-id u1 --partner-id airline_default
python main.py trigger-action --user-id u1 --partner-id airline_default \
  --action-type offer_bundle_upgrade --trigger churn_risk_high \
  --reasoning "High churn score, basic tier" --payload '{"current_tier":"basic"}'
python main.py ingest-events --events-file events.json
```

```bash
# MCP
curl http://localhost:8000/mcp/tools | jq '.tools[].name'
curl -X POST http://localhost:8000/mcp/tools/search_movies \
  -H 'Content-Type: application/json' \
  -d '{"genre":"Comedy","region":"US","max_results":5}'
```

Output shape (`get_recommendations`): `{recommendations:[{rank, movie:{id,title,overview,genres,vote_average,vote_count,release_date,director,cast}, reasoning, score}], user_profile:{top_genres, top_keywords, avg_rating, watch_count}, extracted_features?, session_id?, business_actions_triggered?}`.

---

## 13. Testing

```bash
pytest tests/ -v
```

`tests/test_recommender.py` covers (with mocks, no API keys needed): empty/populated preference extraction + rating weights; IMDb halving below 100 votes + popularity ordering; semantic index build/similar-query/synonym expansion (asserts 384-dim MiniLM embeddings — will download model on first run); template branches (genre vs. high-rating); diversity caps; `_build_query_from_profile` concatenation; `_summarize_watch_history` ("3 movies… 8.5"). Business modules (rules/churn/actions/events), MCP routes, and A2A handlers have **no** tests — manual CLI/MCP curl is the current verification path.

---

## 14. Fallback & Offline Behavior

| Missing / failure | Behavior |
|---|---|
| No `TMDB_API_KEY` | `TMDBClient` raises at init → pipeline logs to stderr, uses local feature-matrix pool for search + stage-1 candidates (genre/rating filter only; no provider/region enforcement). |
| TMDB request error | Same local-pool fallback per call (`"TMDB unavailable (…), falling back…"`) |
| No `DEEPSEEK_API_KEY` | `DeepSeekReasoner` raises → template explanations; `NLFeatureExtractor` uses regex extraction (lower recall, `confidence=0.5`). Both log to stderr. |
| LLM call failure | Per-movie `_fallback_reasoning()` (genre/rating sentence). |
| No precomputed matrices | `initialize()` rebuilds from CSVs (minutes, needs model download). |
| Empty watch history | Zeroed profile → stage-2 boost is ~0; results ≈ popularity/IMDb/query order. |

---

## 15. How to Extend It

- **New partner:** add a `PartnerConfig` in `create_default_rules_engine()` (`src/business/rules_engine.py:286`) with filters/weights/regions/ratings/runtime/offline/session limits + `BusinessRule`s; verify via `get_partner_config` MCP tool and `recommend-biz --partner-id <new>`.
- **New rule:** append `BusinessRule(rule_id, name, condition{…operators…}, actions, priority, partner_id)`; conditions are evaluated by `BusinessRule.matches()` against the rule context built in `_build_rule_context()` (party/device/flight/hotel/tier/time/offline/bandwidth/accessibility + NL genres/decades/mood).
- **New business action:** add `ActionType` + `ActionExecutor` subclass in `src/business/actions.py`, register in `create_default_action_engine()`; expose via `trigger_business_action` (MCP/A2A/CLI share the same engine call).
- **New analytics event:** add `EventType`, emit with `build_event()`, optionally add an `EventIngestion` handler (persistence, warehouse sync, churn feature updater).
- **Ranking tweak:** prefer `config.yaml` weights/thresholds/stage sizes/diversity caps first; code second (`ranker.py` stage functions, `preference_extractor.py` weighting). Re-run `pytest` + a `recommend-biz` golden query.
- **New MCP/A2A surface:** add Pydantic params + `MCPToolType` + handler in `mcp_server.py`, and a matching `AgentCapability` + `_handle_task_request` branch in `a2a_protocol.py` — both delegate to `src/pipeline/recommender.py` or `src/business/*`, so no pipeline duplication.

---

## 16. Glossary

- **Stage-1 / Stage-2:** query-similarity pre-rank (IMDb+TF-IDF+semantic+popularity) → watch-history re-rank (content+keyword boost). `stage1_candidates` controls the handoff size.
- **TF-IDF / CountVectorizer:** bag-of-words text similarity between the query and each movie's combined text.
- **Semantic embedding:** 384-dim MiniLM vector capturing meaning beyond exact words ("funny" ≈ "hilarious").
- **WordNet:** synonym dictionary used for query expansion.
- **TMDB:** The Movie Database API — live catalog, ratings, providers.
- **DeepSeek:** OpenAI-compatible LLM used for NL parsing + explanation generation.
- **MCP:** Model Context Protocol — exposes typed tools over HTTP for other agents/apps.
- **A2A:** Agent-to-Agent protocol — task/discovery message envelope for inter-agent calls.
- **Partner:** a B2B customer (airline/hotel/streamer) with its own catalog rules.
- **Churn:** user abandoning the partner's service; scored 0–1 with retention playbooks.
- **DiversityFilter:** anti-monotony cap (≤3 same genre, ≤2 same director in a row-walk).

---

*Maintainer notes: keep this file in sync when adding CLI subcommands, MCP tools, A2A capabilities, partners, weights, or data files. Secrets belong in `.env` (git-ignored), never in `config.yaml` or docs.*





## TLDR

**Pipeline (base):** Watch history → user profile (genres/keywords/directors/cast) + NL query → structured filters → TMDB live candidates (or local pool fallback) → Stage-1 rank (IMDb + TF-IDF + semantic + popularity) → Stage-2 boost (watch-history content/keyword overlap) → Diversity filter (≤3 same genre, ≤2 same director) → LLM explanation per movie.

**Business path (`BusinessMovieRecommender`):** Same pipeline + partner rules filter, session tracking, churn-risk scoring, auto business actions, and analytics event ingestion.

**Features:** Personalized top-K recs, natural-language search, semantic/synonym matching, 2-stage ranking, LLM reasons with template fallback, partner content rules (airline/hotel/streaming), session/device/party context, churn prediction + retention actions, queued analytics events, 3 interfaces (CLI 7 cmds, MCP 11 tools, A2A 10 caps), graceful offline mode without API keys.
