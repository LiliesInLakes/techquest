# TechQuest Movie Recommender — Evaluation Report

**Evaluation Timestamp (UTC)**: `2026-09-15T14:50:49.304019+00:00`
**Benchmark Dataset Version**: `1.0` (25 test cases)
**Candidate Source**: `Live TMDB Discovery API` (`live_tmdb`)
**API Retrieval Timestamp**: `2026-09-15T14:49:33.959819+00:00`

---

## 1. Data & Embedding Integrity

> [!WARNING]
> **Embedding Space Mismatch & Runtime Scoring Status**
> - **Configured Query Embedding Model**: `multi-qa-mpnet-base-dot-v1` (**768-dimensional**)
> - **Stored Movie Embedding Model**: `all-MiniLM-L6-v2` (**384-dimensional**)
> - **Runtime Behavior**: `MovieRanker._compute_semantic_score()` catches `ValueError` due to shape mismatch (`(1, 768)` vs `(1, 384)`) and silently defaults `semantic_score` to `0.0` for all movies.
> - **Runtime Semantic Relevance Status**: **BROKEN / DEFAULT 0.0**. Reported as `UNAVAILABLE / INVALID` below.

### On-Disk Artifact Audit Summary
| Audit Item | Value / Description |
|---|---|
| **FeatureMatrix Movie Count** | `10` movies |
| **Configured Query Model** | `multi-qa-mpnet-base-dot-v1` (768d) |
| **Stored Movie Model** | `all-MiniLM-L6-v2` (384d) |
| **Runtime Semantic Score** | Always `0.0` (Caught `ValueError`) |
| **Candidate Discovery Mode** | `Live TMDB Discovery API` |
| **History Input Coverage** | `88.00%` |
| **Reco Feature Coverage @5** | `36.80%` |
| **Reco Feature Coverage @10** | `29.60%` |

---

## 2. Current System Baseline Evaluation

This table presents the objective measurements of the current recommender **operating exactly as-is**.

| Metric Dimension | Metric | @5 Value | @10 Value | Status / Note |
|---|---|---|---|---|
| **Semantic Relevance** | Runtime Query-Item Cosine Sim | `UNAVAILABLE` | `UNAVAILABLE` | Invalid due to 768d vs 384d mismatch |
| **Ranking Quality** | HitRate@K | `0.8400` | `0.8400` | Ground truth hit rate |
| **Ranking Quality** | NDCG@K | `0.4450` | `0.4450` | Graded relevance DCG/IDCG |
| **Quality Floor** | Avg Vote Average@K | `8.3594` | `8.2820` | Mean IMDb vote average |
| **Quality Floor** | Avg Vote Count@K | `25389.1` | `25126.8` | Mean vote volume |
| **Diversity** | Intra-List Distance ILD@K | `0.5972` | `0.5972` | Pairwise cosine distance |
| **History Coherence** | Personalization Score@K | `0.0000` | `0.0000` | History content similarity |

---

## 3. Offline Semantic Embedding Diagnostic

> [!NOTE]
> **Diagnostic Scope & Methodology**
> This is an **isolated artifact diagnostic** measuring query-to-item cosine similarity using `all-MiniLM-L6-v2` (384d) against stored movie embeddings in `feature_matrix.pkl`.
> **This diagnostic does NOT represent runtime recommender performance**, because runtime `MovieRanker` uses 768-dim query embeddings and defaults semantic scores to 0.0.

| Diagnostic Metric | @5 Value | @10 Value | Scope |
|---|---|---|---|
| **MiniLM (384d) Cosine Similarity** | `0.3079` | `0.3079` | Stored `feature_matrix.pkl` artifact check |

---

## 4. Per-Test-Case Detailed Breakdown

| ID | Category | Query | Source | Top-5 Recommended Titles | Reco Cov @5 | HitRate@5 | NDCG@5 | MiniLM Diag @5 |
|---|---|---|---|---|---|---|---|---|
| `TC-001` | `semantic_conceptual` | mind-bending simulated reality, virtual ... | `live_tmdb` | The Shawshank Redemption, The Godfather, The  | `40%` | `1.00` | `0.0968` | `0.1599` |
| `TC-002` | `semantic_conceptual` | space travel, wormholes, relativity, and... | `live_tmdb` | Avatar, Star Wars, Interstellar, Spider-Man:  | `40%` | `1.00` | `0.4306` | `0.2340` |
| `TC-003` | `semantic_conceptual` | insomnia, consumerism, underground rebel... | `live_tmdb` | The Shawshank Redemption, The Godfather, The  | `40%` | `1.00` | `0.1443` | `0.1474` |
| `TC-004` | `semantic_conceptual` | non-linear crime stories, hitmen, boxing... | `live_tmdb` | Pulp Fiction, The Dark Knight, The Shawshank  | `60%` | `1.00` | `0.8828` | `0.3676` |
| `TC-005` | `semantic_conceptual` | vigilante justice, corrupt Gotham city, ... | `live_tmdb` | The Shawshank Redemption, The Godfather, The  | `40%` | `1.00` | `0.3726` | `0.5417` |
| `TC-006` | `genre_standard` | science fiction space exploration and fu... | `live_tmdb` | Avatar, Star Wars, Interstellar, Spider-Man:  | `40%` | `1.00` | `0.3608` | `0.4067` |
| `TC-007` | `genre_standard` | crime drama mafia family power struggle ... | `live_tmdb` | Benji, Oliver Twist, Millions, The River Rat, | `0%` | `0.00` | `0.0000` | `N/A` |
| `TC-008` | `genre_standard` | inspiring prison drama friendship hope a... | `live_tmdb` | The Shawshank Redemption, Fight Club, Forrest | `60%` | `1.00` | `0.8179` | `0.4244` |
| `TC-009` | `genre_standard` | historical World War II drama compassion... | `live_tmdb` | Schindler's List, The Pianist, Saving Private | `0%` | `1.00` | `0.6735` | `N/A` |
| `TC-010` | `genre_standard` | heartwarming comedy drama through decade... | `live_tmdb` | Green Book, BlacKkKlansman, The Favourite, Ne | `0%` | `0.00` | `0.0000` | `N/A` |
| `TC-011` | `mixed_constraint` | 1990s action science fiction cyber thril... | `live_tmdb` | The Dark Knight, Terminator 2: Judgment Day,  | `20%` | `0.00` | `0.0000` | `0.3064` |
| `TC-012` | `mixed_constraint` | 2000s psychological thriller heist with ... | `live_tmdb` | The Dark Knight, The Departed, No Country for | `20%` | `1.00` | `0.2887` | `0.4169` |
| `TC-013` | `mixed_constraint` | 1970s crime mafia epic masterpiece... | `live_tmdb` | Pulp Fiction, The Dark Knight, The Shawshank  | `60%` | `1.00` | `0.4307` | `0.3484` |
| `TC-014` | `mixed_constraint` | 1990s Quentin Tarantino neo-noir indie c... | `live_tmdb` | Pulp Fiction, The Dark Knight, The Shawshank  | `60%` | `1.00` | `1.0000` | `0.3456` |
| `TC-015` | `mixed_constraint` | 2010s epic sci-fi adventure in space... | `live_tmdb` | Interstellar, Spider-Man: Into the Spider-Ver | `0%` | `1.00` | `0.9558` | `N/A` |
| `TC-016` | `mood_tone` | dark dystopian psychological dread and e... | `live_tmdb` | The Shawshank Redemption, The Godfather, The  | `40%` | `1.00` | `0.1161` | `0.2669` |
| `TC-017` | `mood_tone` | emotional uplifting hope perseverance th... | `live_tmdb` | The Shawshank Redemption, The Godfather, The  | `40%` | `1.00` | `0.5419` | `0.2050` |
| `TC-018` | `mood_tone` | gritty tense moral dilemma crime in urba... | `live_tmdb` | Pulp Fiction, The Dark Knight, The Shawshank  | `60%` | `1.00` | `0.9839` | `0.3563` |
| `TC-019` | `mood_tone` | awe-inspiring grand cosmic scale lonelin... | `live_tmdb` | The Shawshank Redemption, The Godfather, The  | `40%` | `1.00` | `0.3390` | `0.2026` |
| `TC-020` | `mood_tone` | stylish cool witty fast-paced dialogue a... | `live_tmdb` | The Shawshank Redemption, The Godfather, The  | `40%` | `0.00` | `0.0000` | `0.2966` |
| `TC-021` | `history_sensitive` | futuristic thrillers with high stakes... | `live_tmdb` | The Shawshank Redemption, The Godfather, The  | `40%` | `1.00` | `0.2021` | `0.3100` |
| `TC-022` | `history_sensitive` | intense crime dramas about criminal synd... | `live_tmdb` | Pulp Fiction, The Dark Knight, The Shawshank  | `60%` | `1.00` | `0.9219` | `0.3611` |
| `TC-023` | `history_sensitive` | profound historical and human struggle s... | `live_tmdb` | The Shawshank Redemption, The Dark Knight, Th | `40%` | `1.00` | `0.5419` | `0.2096` |
| `TC-024` | `history_sensitive` | mind-bending dark action thrillers... | `live_tmdb` | The Dark Knight, Avatar, The Lord of the Ring | `40%` | `1.00` | `0.5419` | `0.3486` |
| `TC-025` | `history_sensitive` | sci-fi epics with philosophical themes... | `live_tmdb` | Star Wars, Avatar, Interstellar, Spider-Man:  | `40%` | `1.00` | `0.4806` | `0.2110` |

---

## 5. Failure Cases & Root Cause Analysis

### Test Case `TC-001`: "mind-bending simulated reality, virtual worlds and corporate espionage inside dreams"
- **Category**: `semantic_conceptual`
- **NDCG@5**: `0.0968` | **HitRate@5**: `1.0000`
- **Recommended Titles**: The Shawshank Redemption, The Godfather, The Dark Knight, Interstellar, Project Hail Mary
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-002`: "space travel, wormholes, relativity, and saving humanity from environmental collapse"
- **Category**: `semantic_conceptual`
- **NDCG@5**: `0.4306` | **HitRate@5**: `1.0000`
- **Recommended Titles**: Avatar, Star Wars, Interstellar, Spider-Man: Into the Spider-Verse, Inception
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-003`: "insomnia, consumerism, underground rebellion, split personality disorder, and anarchism"
- **Category**: `semantic_conceptual`
- **NDCG@5**: `0.1443` | **HitRate@5**: `1.0000`
- **Recommended Titles**: The Shawshank Redemption, The Godfather, The Dark Knight, Interstellar, Project Hail Mary
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-005`: "vigilante justice, corrupt Gotham city, psychotic villain Joker, and moral decay"
- **Category**: `semantic_conceptual`
- **NDCG@5**: `0.3726` | **HitRate@5**: `1.0000`
- **Recommended Titles**: The Shawshank Redemption, The Godfather, The Dark Knight, Interstellar, Project Hail Mary
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-006`: "science fiction space exploration and futuristic technology"
- **Category**: `genre_standard`
- **NDCG@5**: `0.3608` | **HitRate@5**: `1.0000`
- **Recommended Titles**: Avatar, Star Wars, Interstellar, Spider-Man: Into the Spider-Verse, Inception
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-007`: "crime drama mafia family power struggle and loyalty"
- **Category**: `genre_standard`
- **NDCG@5**: `0.0000` | **HitRate@5**: `0.0000`
- **Recommended Titles**: Benji, Oliver Twist, Millions, The River Rat, Picture Perfect Lies
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-010`: "heartwarming comedy drama through decades of American history"
- **Category**: `genre_standard`
- **NDCG@5**: `0.0000` | **HitRate@5**: `0.0000`
- **Recommended Titles**: Green Book, BlacKkKlansman, The Favourite, Newsies, Saving Mr. Banks
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-011`: "1990s action science fiction cyber thriller"
- **Category**: `mixed_constraint`
- **NDCG@5**: `0.0000` | **HitRate@5**: `0.0000`
- **Recommended Titles**: The Dark Knight, Terminator 2: Judgment Day, Aliens, V for Vendetta, The Dark Knight Rises
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-012`: "2000s psychological thriller heist with Christopher Nolan direction"
- **Category**: `mixed_constraint`
- **NDCG@5**: `0.2887` | **HitRate@5**: `1.0000`
- **Recommended Titles**: The Dark Knight, The Departed, No Country for Old Men, Kill Bill: Vol. 2, Zodiac
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-013`: "1970s crime mafia epic masterpiece"
- **Category**: `mixed_constraint`
- **NDCG@5**: `0.4307` | **HitRate@5**: `1.0000`
- **Recommended Titles**: Pulp Fiction, The Dark Knight, The Shawshank Redemption, The Godfather, The Godfather Part II
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-016`: "dark dystopian psychological dread and existential angst"
- **Category**: `mood_tone`
- **NDCG@5**: `0.1161` | **HitRate@5**: `1.0000`
- **Recommended Titles**: The Shawshank Redemption, The Godfather, The Dark Knight, Interstellar, Project Hail Mary
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-019`: "awe-inspiring grand cosmic scale loneliness and love across time"
- **Category**: `mood_tone`
- **NDCG@5**: `0.3390` | **HitRate@5**: `1.0000`
- **Recommended Titles**: The Shawshank Redemption, The Godfather, The Dark Knight, Interstellar, Project Hail Mary
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-020`: "stylish cool witty fast-paced dialogue and sudden violence"
- **Category**: `mood_tone`
- **NDCG@5**: `0.0000` | **HitRate@5**: `0.0000`
- **Recommended Titles**: The Shawshank Redemption, The Godfather, The Dark Knight, Interstellar, Project Hail Mary
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-021`: "futuristic thrillers with high stakes"
- **Category**: `history_sensitive`
- **NDCG@5**: `0.2021` | **HitRate@5**: `1.0000`
- **Recommended Titles**: The Shawshank Redemption, The Godfather, The Dark Knight, Interstellar, Project Hail Mary
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

### Test Case `TC-025`: "sci-fi epics with philosophical themes"
- **Category**: `history_sensitive`
- **NDCG@5**: `0.4806` | **HitRate@5**: `1.0000`
- **Recommended Titles**: Star Wars, Avatar, Interstellar, Spider-Man: Into the Spider-Verse, Inception
- **Diagnostic Root Cause**: Semantic scoring path is broken (`0.0`), forcing candidate selection to rely strictly on IMDb popularity, TF-IDF, or genre match. When candidate discovery yields movies lacking keyword overlap, ranking falls back to global popularity.

---

## 6. System & Dataset Limitations

1. **10-Movie FeatureMatrix Pool**: `feature_matrix.pkl` contains only 10 movies. When TMDB API is offline or uninitialized, local fallback recommendations are severely constrained.
2. **Embedding Model Incompatibility**: `config.yaml` specifies 768-dim `multi-qa-mpnet-base-dot-v1`, but stored `feature_matrix.pkl` contains 384-dim `all-MiniLM-L6-v2` embeddings, rendering semantic scoring completely inactive at runtime.
3. **Candidate Coverage Gap**: Live TMDB recommendations not present in `feature_matrix.pkl` cannot be evaluated for embedding-dependent ILD or Personalization metrics.

---

## 7. Future Semantic-Weight Experiment Protocol

> [!IMPORTANT]
> **Blocked Experiment Status**
> `semantic_weight` experiments (`0.1`, `0.4`, `0.7`, `0.9`) are **BLOCKED** on the current system baseline because `semantic_score` is always `0.0`. Tuning weights on a zero score yields zero output change.

### Protocol for Post-Repair Execution
Once the embedding pipeline is repaired (so query embeddings and movie embeddings use identical models and vector dimensions), run the following protocol:
1. **Configurations to Evaluate**: `semantic_weight = [0.1, 0.4, 0.7, 0.9]`
2. **Fixed Environment**: Identical 25 benchmark test cases, identical ground truth, identical candidate policies.
3. **Target Measurement**: Quantify trade-offs between Semantic Relevance, Ranking Quality (NDCG), and Diversity (ILD).

---

## 8. Executive Summary & Final Assessment

- **System Status**: The recommender's semantic search path is currently broken due to a 768d vs 384d embedding dimension mismatch.
- **Ranking & Quality Baseline**: Recommender performance currently relies primarily on IMDb ratings, popularity, and TF-IDF keyword match.
- **Evaluation Integrity**: Runtime Semantic Relevance is accurately marked `UNAVAILABLE`, avoiding manufactured numbers. Offline artifact quality is documented via isolated MiniLM diagnostic.
