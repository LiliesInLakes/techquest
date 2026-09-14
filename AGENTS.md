# TechQuest — Project Context

## What this is
Agentic movie-recommendation system. Blends TMDB live discovery, offline
feature engineering (TF-IDF + MiniLM semantic embeddings), two-stage ranking,
and LLM explanations. Driven via CLI, MCP server (FastAPI :8000), or A2A agent.

## How to run
- `python main.py precompute` — build `data/processed/*.pkl` (slow, one-time).
- `python main.py recommend --user-id demo --watch-history hist.json --top-k 5 --natural-language "..." --subscriptions netflix`
- `python main.py mcp` — FastAPI server on :8000; `python main.py a2a` — prints agent card.
- Tests: `pytest tests/ -v` (use `venv/bin/python` / activate `venv/`).

## API keys (in `.env`, loaded via python-dotenv)
- `TMDB_API_KEY` → live movie discovery/search (TMDBClient).
- `DEEPSEEK_API_KEY` → LLM NL extraction + per-movie explanations (pointed at `https://api.deepseek.com/v1`, model `deepseek-chat`).
- Graceful fallback: no TMDB → local `data/processed` pool; no DeepSeek → regex NL parsing + template explanations.
- Verify keys: TMDB `GET https://api.themoviedb.org/3/configuration?api_key=...`; DeepSeek `GET https://api.deepseek.com/v1/models` with `Authorization: Bearer ...`.

## Pipeline (MovieRecommender.get_recommendations)
1. Ratings dict from watch history (`movie_id`/`id` + rating>0).
2. `PreferenceExtractor` → user profile (genres/keywords/directors/cast, TF-IDF weights).
3. `NLFeatureExtractor` → structured filters from NL query (DeepSeek or regex fallback).
4. Stage-1 candidates: TMDB discover (first genre, popularity.desc) + exclusion filter; <50 → top up with populars; fail → local pool.
5. Vectorize query → (count, tfidf, MiniLM embedding).
6. `MovieRanker.rank_movies_stage1` — IMDb + tf-idf + semantic + popularity → top stage1_candidates.
7. `rank_movies_stage2` — watch-history boost (content + keyword overlap).
8. `DiversityFilter` — ≤3 same genre, ≤2 same director.
9. Explanations — DeepSeekReasoner (async batch) or template fallback.

## Key files
- `main.py` — CLI: mcp/a2a/precompute/recommend subcommands.
- `mcp_server.py` — FastAPI :8000, 4 tools (search_movies, get_recommendations, get_movie_details, analyze_user_preferences).
- `a2a_protocol.py` — A2A dataclasses + MovieRecommenderA2AAgent (3 capabilities).
- `src/pipeline/recommender.py` — orchestrator `MovieRecommender`.
- `src/tmdb/client.py` — async TMDBClient (aiohttp), raises ValueError if key missing → local-pool mode.
- `src/features/matrix_builder.py` — offline build + `transform_query()` + `load()`.
- `src/features/semantic_search.py` — embedding + WordNet synonym matching.
- `src/features/preference_extractor.py` — user profile extraction.
- `src/ranking/ranker.py` — stage1/stage2 scoring + DiversityFilter.
- `src/llm/reasoning.py` — DeepSeekReasoner + ReasoningTemplateEngine.
- `src/llm/nl_feature_extractor.py` — NL → ExtractedFeatures.
- `config.yaml` — all tunable knobs (weights, stage sizes, diversity caps, API endpoints).
- `hist.json` — sample watch history `[{"movie_id", "rating"}]`.

## Config notes
- `config.yaml` uses `${TMDB_API_KEY}` placeholders but code reads env directly.
- ranker reuses `semantic_similarity_threshold` as `semantic_weight` (known quirk).

## Caveats
- `requirements.txt` omits fastapi/uvicorn/aiohttp/mcp/pydantic/pytest — they are in `venv/`.
- No MCP/A2A tests; only CLI/unit tests exist. Manual curl is the MCP verification path.

## Fixes made (Sep 2026)
- `src/features/matrix_builder.py`: credits CSV keys on `movie_id`, not `id` — renamed to `id` before merge (precompute crashed with KeyError).
- `src/tmdb/client.py`: `with_watch_providers` now resolves provider NAMES (e.g. "netflix") to TMDB numeric IDs via `/watch/providers/movie` instead of passing the raw name (which TMDB rejects → 0 discover results).
- `src/pipeline/recommender.py::_passes_exclusions`: overview can be NaN/float (local-pool metadata); `str`-check before `.lower()` instead of crashing.

## Known issue (as of last run)
- DeepSeek key authenticates (`/models` → 200) but account has **Insufficient Balance** (`402` on chat calls). Effect: NL extraction silently falls back to regex (`confidence 0.5`) and explanations use the template engine. Fix = top up DeepSeek credit. Pipeline still returns results either way.