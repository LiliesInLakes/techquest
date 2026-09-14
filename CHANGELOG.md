# Project Changes & Team Onboarding Guide

This changelog outlines all file-level and directory-level modifications made to the repository, along with the exact dependency setup required for any team member to run the project without version mismatches.

---

## 1. Quick Setup for Team Members

Run these commands from the project root directory (`techquest/`):

```bash
# 1. Activate your virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# (Or on Linux/macOS: source .venv/bin/activate)

# 2. Install all required dependencies
pip install -r requirements.txt

# 3. Configure environment variables (optional for local fallback)
copy .env.example .env
# (Fill in TMDB_API_KEY, OPENROUTER_API_KEY, or DEEPSEEK_API_KEY if using live APIs)

# 4. Verify installation (runs all 32 automated tests)
pytest
```

---

## 2. Required Dependencies (`requirements.txt`)

These dependencies must be installed to ensure parity across all team members:

| Package | Minimum Version | Purpose |
| :--- | :--- | :--- |
| **`fastapi`** | `>=0.110.0` | REST API endpoints (`/users/{user_id}/profile`, `/events`, `/ratings`) |
| **`uvicorn`** | `>=0.28.0` | ASGI web server for FastAPI & MCP server |
| **`mcp`** | `>=1.0.0` | Model Context Protocol SDK exposing agent tools |
| **`aiohttp`** | `>=3.9.0` | Asynchronous HTTP client for TMDB & OpenRouter |
| **`pytest`** | `>=8.0.0` | Test runner framework |
| **`pytest-asyncio`**| `>=0.23.0` | Asynchronous pytest execution plugin |
| **`requests`** | `==2.31.0` | Synchronous HTTP requests |
| **`pandas`** | `>=2.1.4` | Metadata and dataset processing |
| **`numpy`** | `>=1.26.2` | Vector math, EMA, and decay calculations |
| **`scikit-learn`** | `>=1.3.2` | TF-IDF, CountVectorizer, and cosine similarity |
| **`scipy`** | `>=1.11.4` | Sparse matrix calculations (`csr_matrix`) |
| **`sentence-transformers`** | `>=2.7.0` | Sentence embeddings (`all-MiniLM-L6-v2`) |
| **`torch`** | `>=2.2.0` | Neural network backend for sentence transformers |
| **`openai`** | `>=1.3.7` | Client for OpenRouter & DeepSeek LLM reasoning |
| **`nltk`** | `==3.8.1` | Natural language processing tools |
| **`pyyaml`** | `==6.0.1` | Configuration loader for `config.yaml` |
| **`python-dotenv`** | `==1.0.0` | `.env` file environment variable loader |
| **`tqdm`** | `==4.66.1` | Progress bars for batch indexing |

---

## 3. Directory & File Change Log

### `src/memory/` **[NEW DIRECTORY]**
Implements the V1 User Watch Profile / Memory Subsystem (Observable interactions only, zero streaming telemetry assumptions).

- [`src/memory/__init__.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/src/memory/__init__.py) **[NEW]**: Module interface and symbol exports.
- [`src/memory/models.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/src/memory/models.py) **[NEW]**: Data models and Enums:
  - `EventType` (`movie_clicked`, `movie_viewed`, `added_to_watchlist`, `marked_watched`, `rated`).
  - `FeatureType` (`genre`, `keyword`, `director`, `cast`, `decade`, `language`).
  - Dataclasses: `WatchEvent`, `RatingEvent`, `FeaturePreference`, `NegativePreference`, `UserProfileMemory`.
- [`src/memory/decay.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/src/memory/decay.py) **[NEW]**:
  - Exponential time decay with half-life $H = 7.0$ days: $\text{decay}(t) = 2^{-t / 7.0}$.
  - Normalized weighted aggregation for short-term memory.
- [`src/memory/storage.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/src/memory/storage.py) **[NEW]**:
  - SQLite persistent storage engine (`data/memory.db`).
  - Stores raw `watch_events`, `ratings`, and derived `user_profile` state.
  - Queries for 30-min deduplication cooldown and 24-hour feature weight lookups.
- [`src/memory/engine.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/src/memory/engine.py) **[NEW]**:
  - **Asymptotic Addition:** $P_{\text{long-term}}(f) \leftarrow P_{\text{long-term}}(f) + \alpha \cdot \text{Effective\_Weight} \cdot (1.0 - P_{\text{long-term}}(f))$ (No score drops on positive actions).
  - **Smooth Beta Curve:** $\beta(N) = 0.40 + 0.60 \cdot \max(0, 1 - N/10)$ (Transitions from short-term to long-term memory).
  - **Bayesian Confidence:** $\text{confidence}(f) = \frac{N_f}{N_f + 10}$.
  - **Rating Spectrum:**
    - 5★: $+1.0$ (Positive)
    - 4★: $+0.6$ (Positive)
    - 3★: $0.0$ (Neutral: increments $N_f$ and $N$ for confidence, $0.0$ score weight, no negative penalty)
    - 2★: $+0.4$ (Negative penalty)
    - 1★ & 0★: $+1.0$ (Negative penalty)
  - **24-Hour Binge Damping & 30-Min Cooldown:** $\text{Effective\_Weight} = \log_2(1 + \sum \text{Weight}_{24h})$, 30-min repeat events damped by $\times 0.25$.
  - **Profile Replay:** Chronological deterministic reconstruction from raw SQLite logs.
- [`src/memory/service.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/src/memory/service.py) **[NEW]**:
  - `UserMemoryService` API coordinator.
  - `get_combined_preference_profile(user_id)` adapter producing `UserPreferenceProfile` with negative preferences attached for `MovieRanker`.

---

### `src/ranking/` **[MODIFIED]**
- [`src/ranking/ranker.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/src/ranking/ranker.py) **[MODIFIED]**:
  - Implemented **Blended Negative Penalty Formula**:
    $$\text{penalty}(movie) = 1.0 - \min\left(0.9, 0.75 \cdot \max_{f \in movie}(neg(f)) + 0.25 \cdot \frac{1}{|F|}\sum_{f \in movie} neg(f)\right)$$
  - Applied penalty multiplier to stage 2 ranked candidates and exposed `negative_penalty` in `reasoning_factors`.
  - 100% backward-compatible when profiles have no negative preferences.

---

### `src/features/` **[MODIFIED]**
- [`src/features/matrix_builder.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/src/features/matrix_builder.py) **[MODIFIED]**:
  - Added safe column check for `'movie_id'` in `credits_df` to rename to `'id'` before merging with `movies_df`, fixing `KeyError` on raw TMDB 5000 CSV ingestion during `python main.py precompute`.

---

### `src/tmdb/` **[MODIFIED]**
- [`src/tmdb/client.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/src/tmdb/client.py) **[MODIFIED]**:
  - Added `_resolve_watch_provider_ids` helper to map streaming service names (e.g. "netflix", "prime") to numeric TMDB provider IDs via `/watch/providers/movie`.
  - Updated `discover_movies` to resolve provider IDs before dispatching discover queries.

---

### `src/pipeline/` **[MODIFIED]**
- [`src/pipeline/recommender.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/src/pipeline/recommender.py) **[MODIFIED]**:
  - Initialized `UserMemoryService` inside `initialize()`.
  - Added fallback to load from persistent user memory when `watch_history` is empty or omitted.
  - Fixed offline / missing `TMDB_API_KEY` handling (`self.tmdb_client is not None` checks) to avoid `'NoneType' object does not support the asynchronous context manager protocol` crashes.
  - Added safe fallback in `get_movie_details()` to local precomputed feature matrix when TMDB is offline.
  - Added `isinstance(overview, str)` check in `_passes_exclusions` to prevent `AttributeError` crashes on `NaN`/float overview fields in local candidate datasets.

---

### `src/llm/` **[MODIFIED]**
- [`src/llm/nl_feature_extractor.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/src/llm/nl_feature_extractor.py) **[MODIFIED]**:
  - Added OpenRouter model routing (`deepseek/deepseek-chat` / `openrouter/auto`) and fallback to local heuristic keyword matching when API keys are absent.
- [`src/llm/reasoning.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/src/llm/reasoning.py) **[MODIFIED]**:
  - Added OpenRouter compatibility, custom base URL support, and template engine fallback.

---

### Root Application Files **[MODIFIED & NEW]**
- [`mcp_server.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/mcp_server.py) **[MODIFIED]**:
  - Added MCP tools: `record_watch_event`, `record_movie_rating`, `get_user_profile`.
  - Added REST endpoints: `GET /users/{user_id}/profile`, `POST /users/{user_id}/events`, `POST /users/{user_id}/ratings`.
  - Fixed async initialization in `get_movie_details` and `analyze_preferences`.
- [`main.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/main.py) **[MODIFIED]**:
  - Added CLI commands: `profile`, `log-event`, `rate`.
  - Updated `recommend` command to automatically read from memory store when `--watch-history` is omitted.
- [`a2a_protocol.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/a2a_protocol.py) **[MODIFIED]**:
  - Added missing `import asyncio`.
  - Bound async futures to `asyncio.get_running_loop().create_future()`.
  - Fixed correlation ID matching for request/response dispatch.
  - Replaced deprecated `datetime.utcnow()` with `datetime.now(timezone.utc)` for Python 3.12+.
- [`requirements.txt`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/requirements.txt) **[MODIFIED]**:
  - Updated with all required dependencies (`fastapi`, `uvicorn`, `mcp`, `aiohttp`, `pytest`, `pytest-asyncio`).
- [`.gitignore`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/.gitignore) **[MODIFIED]**:
  - Added ignore rules for SQLite databases (`*.db`, `data/*.db`, `data/memory.db`).
- [`.env.example`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/.env.example) **[NEW]**:
  - Environment variable template documenting `TMDB_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_BASE_URL`.

---

### `tests/` **[NEW & UPDATED]**
- [`tests/test_memory.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/tests/test_memory.py) **[NEW]** (17 tests):
  - Unit tests for asymptotic addition, smooth beta transition, blended negative penalty, rating spectrum, neutral 3-star, 24h binge damping, half-life decay, Bayesian confidence, and replay idempotence.
- [`tests/test_a2a.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/tests/test_a2a.py) **[NEW]** (2 tests):
  - Tests async A2A agent discovery and task request handling.
- [`tests/test_recommender.py`](file:///c:/GamC/.vscode/techquest_project/ps/techquest/tests/test_recommender.py) **[PRESERVED]** (13 tests):
  - End-to-end recommender, ranker, and semantic search tests.

**Total Test Coverage: 32 / 32 Passed (100% Pass Rate)**

---

## 4. Common CLI Commands for Team Members

```bash
# Run all unit tests
pytest

# View user profile memory
python main.py profile --user-id <user_id>

# Log an observable watch event
python main.py log-event --user-id <user_id> --movie-id 550 --event marked_watched

# Rate a movie (0 to 5 stars)
python main.py rate --user-id <user_id> --movie-id 550 --rating 5.0

# Get recommendations (using persistent profile memory)
python main.py recommend --user-id <user_id> --top-k 5

# Start FastAPI / MCP server
python main.py mcp
```
