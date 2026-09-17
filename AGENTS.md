# TechQuest — Project Context

## What this is
Agentic movie-recommendation system. Blends TMDB live discovery, offline
feature engineering (SQLite `movie_features.db` with 87,425 canonical movies, 5,000-feature TF-IDF + 768-d `multi-qa-mpnet-base-dot-v1` semantic embeddings), two-stage ranking with age-relative popularity, and LLM explanations. Driven via Web UI (FastAPI :8080), CLI, MCP server (FastAPI :8000), or A2A agent.

## How to run
- `python main.py setup-db` — verify SQLite `movie_features.db` integrity and canonical manifest.
- `python main.py precompute` — build `data/processed/movie_embeddings.npy` (87.4k × 768 float32), `count_vectorizer.pkl`, `tfidf_vectorizer.pkl`, and `feature_matrix.pkl`.
- `python main.py ui` — run interactive single-recommendation Web UI on `http://127.0.0.1:8080`.
- `python main.py recommend --user-id demo --watch-history hist.json --top-k 5 --natural-language "..." --subscriptions netflix`
- `python main.py mcp` — FastAPI server on :8000; `python main.py a2a` — prints agent card.
- Tests: `pytest tests/ -v` (use `venv/bin/python` / `venv\Scripts\python`).

## API keys (in `.env`, loaded via python-dotenv)
- `TMDB_API_KEY` → live movie discovery/search/providers (TMDBClient).
- `KADO_API_KEY`, `KADO_BASE_URL`, `KADO_MODEL` → LLM NL extraction + per-movie explanations (pointed at OpenAI-compatible endpoint, e.g. `https://awesome.kado.so/openai/v1`).
- Legacy fallback: `DEEPSEEK_API_KEY` is supported for backwards compatibility.
- Graceful fallback: no TMDB → local `movie_features.db` pool; no LLM key → regex NL parsing + template explanations.
- Verify keys: TMDB `GET https://api.themoviedb.org/3/configuration?api_key=...`; Kado/OpenAI `GET /v1/models` with `Authorization: Bearer ...`.

## Pipeline (MovieRecommender.get_recommendations)
1. Ratings dict from watch history (`movie_id`/`id` + rating>0).
2. `UserMemoryService` / `PreferenceExtractor` → multi-dimensional user profile (genres, tags, themes, tones, TF-IDF weights, negative penalties).
3. `NLFeatureExtractor` → structured filters separating explicit hard constraints (genres, actors, crew, year, language, adult) from soft concepts (mood, theme, semantic search query).
4. Stage-1 candidates: TMDB discover (explicit filters, popularity.desc) + SQLite membership filter (unindexed movies dropped; multi-page top-up up to 5 pages); fallback → local SQLite pool.
5. Vectorize query → (count_vec, tfidf_vec, 768-d MPNet embedding).
6. `MovieRanker.rank_movies_stage1` — 0.30 IMDb/rating quality + 0.25 TF-IDF relevance + 0.35 semantic relevance + 0.10 age-adjusted popularity (cohort percentile in $Y \pm 3$ / $Y \pm 5$ years + global log popularity).
7. `rank_movies_stage2` — watch-history boost (content similarity + keyword overlap) combined with multi-feature negative preference penalties (dislikes/skips).
8. `DiversityFilter` — ≤3 same genre, ≤2 same director.
9. Explanations — Kado LLM reasoner or template engine fallback.

## Key files
- `main.py` — CLI: ui/mcp/a2a/setup-db/precompute/recommend subcommands.
- `src/ui/web_app.py` — FastAPI :8080 single-recommendation showcase UI with localStorage user persistence and skip/feedback loop.
- `mcp_server.py` — FastAPI :8000, 4 tools (search_movies, get_recommendations, get_movie_details, analyze_user_preferences).
- `a2a_protocol.py` — A2A dataclasses + MovieRecommenderA2AAgent (3 capabilities).
- `movie_features.db` — Authoritative SQLite database (87,425 canonical TMDB movies).
- `src/pipeline/recommender.py` — orchestrator `MovieRecommender`.
- `src/tmdb/client.py` — async TMDBClient (aiohttp), raises ValueError if key missing → local-pool mode.
- `src/features/matrix_builder.py` — offline build + `transform_query()` + `load()`.
- `src/features/semantic_search.py` — 768-d MPNet embedding + WordNet synonym matching.
- `src/features/preference_extractor.py` — user profile extraction.
- `src/memory/service.py` — UserMemoryService with dual-branch short/long term and negative preference ledger.
- `src/ranking/ranker.py` — stage1/stage2 scoring + age-relative popularity + DiversityFilter.
- `src/llm/reasoning.py` — Kado / DeepSeek Reasoner + ReasoningTemplateEngine.
- `src/llm/nl_feature_extractor.py` — NL → ExtractedFeatures (hard vs soft constraint separation).
- `config.yaml` — all tunable knobs (weights, stage sizes, diversity caps, API endpoints).
- `hist.json` — sample watch history `[{"movie_id", "rating"}]`.

## Ranking Weights (Stage 1)
- `rating_quality` (IMDb): 0.30
- `tfidf_relevance`: 0.25
- `semantic_relevance`: 0.35
- `age_adjusted_popularity`: 0.10
- Total: 1.00 (strictly linear, no recency bonus, no double counting).
- Age-relative popularity: $0.70 \times \text{cohort\_percentile}(\log(1+\text{votes})) + 0.30 \times \text{global\_popularity}$. Cohort radius: $Y \pm 3$ years (fallback $Y \pm 5$ years, min size 20).