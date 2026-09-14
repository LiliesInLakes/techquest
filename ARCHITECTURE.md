# System Architecture & Flowchart Documentation

This document describes the complete architecture, pipeline flowcharts, memory subsystems, data flow, and module responsibilities of the **Movie Recommender Agent**.

---

## 1. High-Level System Architecture

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
                │  2. NLFeatureExtractor: Structured TMDB discovery parameters (DeepSeek / regex)  │
                │  3. TMDBClient (or local pool): Live Discovery with Provider ID resolution       │
                │  4. Vectorization: multi-qa-mpnet-base-dot-v1 (768-dim) + TF-IDF (5,000-dim)    │
                │  5. MovieRanker.rank_movies_stage1: Query similarity + Quality + Popularity      │
                │  6. MovieRanker.rank_movies_stage2: History boost + Multi-feature Neg Penalties  │
                │  7. DiversityFilter: Genre and director frequency limits                         │
                │  8. Single Final Recommendation (top_k=1): #1 Pick + AI Verdict & Rationale      │
                │  9. Rejection Replacement Engine: Skip → Penalize Dislikes → Instant Fresh Pick │
                └──────────────┬─────────────────────────┬─────────────────────────┬───────────────┘
                               │                         │                         │
                               ▼                         ▼                         ▼
                        ┌────────────┐            ┌──────────────┐          ┌────────────────────┐
                        │ TMDB API   │            │ Offline pool │          │ LLM (DeepSeek /    │
                        │ (Live)     │            │ data/        │          │ OpenRouter)        │
                        │ Discover / │            │ processed/   │          │ Top Pick Verdict & │
                        │ Providers  │            │ feature_mat. │          │ Batch Reasoning    │
                        └────────────┘            └──────────────┘          └────────────────────┘
```

---

## 2. End-to-End Recommendation Pipeline Flowchart

```mermaid
flowchart TD
    subgraph ClientInterfaces ["1. Client Interfaces"]
        WebUI["Web UI (src/ui/web_app.py :8080)\nCinematic Single Showcase & Real-Time Skip Loop"]
        CLI["CLI (main.py)\nrecommend (top_k=1) | reject | profile | rate"]
        MCP["MCP Server (mcp_server.py)\nFastAPI :8000 (4 Tools)"]
        A2A["A2A Protocol (a2a_protocol.py)\nAgent-to-Agent (3 Capabilities)"]
    end

    subgraph InputProcessing ["2. Input & Memory Extraction"]
        UserReq["User Request\n• User ID\n• Natural Language Query\n• Session Excluded Movie IDs\n• Subscriptions / Region"]
        NLParser{"NL Query Present?"}
        LLMExtract["NLFeatureExtractor\n(DeepSeek / OpenRouter JSON parse)"]
        RegexFallback["Regex Extraction Fallback\n(Genre / Decade / Keyword scan)"]
        VocabMap["map_to_vocabulary()\nSnap keywords to model vocabulary"]
        MemoryLoad["UserMemoryService\nLoad Multi-Dimensional Profile\n(Genres, MovieLens Tags, Themes, Tones, Dislikes)"]
    end

    subgraph CandidateRetrieval ["3. Candidate Discovery"]
        TMDBCheck{"TMDB Available?"}
        LiveTMDB["TMDBClient.discover_movies()\n• Resolve numeric watch provider IDs\n• Filter: Genre, Release Dates, Runtime\n• Sort: popularity.desc / vote_average.desc\n• Exclude unwanted genres/keywords"]
        LocalPool["Local Candidate Pool\n(4,800+ Movies in data/processed/)"]
    end

    subgraph RankingEngine ["4. Two-Stage Ranking & Diversity"]
        Stage1["Stage 1: Fast Query Relevance (768-dim MPNet)\nScore = 0.30·IMDb + 0.20·TFIDF + 0.30·Semantic + 0.10·Popularity"]
        Top100["Select Top 100 Stage-1 Candidates"]
        Stage2["Stage 2: Personalization & Negative Penalties\n• Content Similarity + Keyword Overlap Boost\n• Blended Multi-Feature Negative Penalty\n(Genres + Tags + Themes + Tones)"]
        DivFilter["DiversityFilter\nCap max same genre (3) & director (2)"]
        ExcludeFilter["Exclude Session-Seen / Rejected Movie IDs"]
        Top1Select["Select Single Final #1 Candidate (top_k=1)"]
    end

    subgraph ExplanationGen ["5. Single Verdict & Decision Loop"]
        LLMVerdict{"LLM Reasoner Available?"}
        DeepSeekVerdict["DeepSeekReasoner.generate_top_pick_verdict()\nStrict Target Movie Constraint + Decisive Rationale"]
        TemplateVerdict["ReasoningTemplateEngine.generate_top_pick()\nRule-based prioritized verdict"]
        SingleShowcase["Single Recommendation Hero Card\n• Title, Poster, Match %, Rating, Year\n• MovieLens Tags, Themes, Tones, Stream\n• AI Verdict & Personalized Why-You'll-Like-It"]
        UserDecision{"User Action?"}
        Accept["Watched / Watchlist / Rate\n(Positive Interaction Event)"]
        Skip["Not For Me / Suggest Another\n(Rejection Event)"]
        LearnNeg["Learn Negative Preferences\nPenalize Disliked Movie's Tags/Themes/Tones"]
    end

    %% Client Routing
    WebUI --> UserReq
    CLI --> UserReq
    MCP --> UserReq
    A2A --> UserReq

    %% Processing
    UserReq --> NLParser
    UserReq --> MemoryLoad

    NLParser -- Yes --> LLMExtract
    LLMExtract -- Fails/No Key --> RegexFallback
    LLMExtract --> VocabMap
    RegexFallback --> VocabMap

    NLParser -- No --> MemoryLoad
    VocabMap --> TMDBCheck
    MemoryLoad --> TMDBCheck

    TMDBCheck -- Yes --> LiveTMDB
    TMDBCheck -- No / Offline --> LocalPool

    LiveTMDB --> Stage1
    LocalPool --> Stage1

    Stage1 --> Top100
    Top100 --> Stage2
    MemoryLoad -.-> Stage2
    Stage2 --> DivFilter
    DivFilter --> ExcludeFilter
    ExcludeFilter --> Top1Select

    Top1Select --> LLMVerdict
    LLMVerdict -- Yes --> DeepSeekVerdict
    LLMVerdict -- No / Error --> TemplateVerdict

    DeepSeekVerdict --> SingleShowcase
    TemplateVerdict --> SingleShowcase

    SingleShowcase --> UserDecision
    UserDecision -- Accept / Rate --> Accept
    UserDecision -- Skip / Dislike --> Skip
    Skip --> LearnNeg
    LearnNeg -.-> ExcludeFilter
```

---

## 3. Adaptive Multi-Dimensional Memory Subsystem

```mermaid
flowchart LR
    subgraph UserAction ["User Interaction Spectrum"]
        PosAct["Positive Signals\n• Click (w=0.1)\n• Watchlist (w=0.2)\n• Watched (w=0.3)\n• Rate 4.0-5.0 stars (w=0.5..1.0)"]
        RejAct["Negative Signals\n• Skip / 'Not For Me' (w=-0.4)\n• Rate 0.0-2.0 stars (w=-0.5..-1.0)"]
    end

    subgraph FeatureExtractor ["Metadata Feature Extraction"]
        Ext["Feature Extractor (engine.py)\n• Genres (TMDB / Local Metadata)\n• MovieLens Genome Tags (#space-opera, #dystopia)\n• Story Themes (survival, revenge, corruption)\n• Moods / Tones (dark, cerebral, suspenseful)"]
    end

    subgraph DatabaseLedger ["SQLite Storage (data/memory.db)"]
        Ledger["Append Event Record\n• EventType: REJECTED | WATCHED | RATED\n• Extracted Multi-Feature Payload\n• Signal Weight & Timestamp"]
    end

    subgraph DualMemoryUpdate ["Dual-Branch Memory Update"]
        direction TB
        ShortTerm["Short-Term Memory (P_short)\n• Exponential Time-Decay (7-Day Half-Life)\n• Immediate context & current mood"]
        LongTerm["Long-Term Memory (P_long)\n• Asymptotic Addition: P += (1-P)·w\n• Bayesian Confidence: N / (N + K)\n• Permanent affinity profile"]
        Negative["Negative Preferences (P_neg)\n• Triggered on Skips & Low Ratings\n• Tracks penalized genres, tags, themes, tones"]
    end

    subgraph BlendingEngine ["Smooth Beta Dynamic Balance"]
        BetaCalc["Dynamic Balance Calculation:\nβ(N) = 0.40 + 0.60 · max(0, 1 - N/10)"]
        Combined["Combined Taste Profile:\nP_combined = β · P_short + (1 - β) · P_long"]
    end

    PosAct --> Ext
    RejAct --> Ext
    Ext --> Ledger
    Ledger --> ShortTerm
    Ledger --> LongTerm
    Ledger --> Negative

    ShortTerm --> BetaCalc
    LongTerm --> BetaCalc
    BetaCalc --> Combined
```

### Mathematical Formulations

1. **Short-Term Memory Exponential Decay:**
   $$\Delta t = t_{\text{current}} - t_{\text{event}}$$
   $$\lambda = \frac{\ln(2)}{7 \times 86400} \approx 1.1458 \times 10^{-6} \text{ s}^{-1}$$
   $$P_{\text{short}}^{(t)} = \sum_{i} w_i \cdot e^{-\lambda \cdot (t - t_i)}$$

2. **Long-Term Asymptotic Memory Addition:**
   $$P_{\text{long}}^{(t)} = P_{\text{long}}^{(t-1)} + \left(1.0 - P_{\text{long}}^{(t-1)}\right) \cdot w_{\text{new}}$$

3. **Bayesian Confidence Shrinkage:**
   $$C(N) = \frac{N}{N + K}, \quad \text{where } K = 10$$

4. **Smooth Beta Dynamic Balance:**
   $$\beta(N) = 0.40 + 0.60 \cdot \max\left(0, 1 - \frac{N}{10}\right)$$
   $$P_{\text{combined}} = \beta(N) \cdot P_{\text{short}} + (1 - \beta(N)) \cdot P_{\text{long}}$$

5. **Blended Negative Preference Penalty (Multi-Feature):**
   $$\text{PenaltyFactor} = 1.0 - \min\left(0.90, 0.75 \cdot \max_{f \in M}(\text{Neg}(f)) + 0.25 \cdot \frac{1}{|F|}\sum_{f \in M}\text{Neg}(f)\right)$$
   *Applied across all extracted candidate features: genres, MovieLens tags, themes, and tones.*

---

## 4. Two-Stage Ranking Mathematics Flowchart

```mermaid
flowchart TD
    Candidates["Input Candidates (Live TMDB or Local Matrix Pool)"]

    subgraph Stage1Calc ["Stage 1: Content Relevance & Semantic Matching"]
        IMDb["IMDb Score\n(vote_avg / 10, halved if votes < 100)"]
        TFIDF["TF-IDF Cosine Similarity\n(Query text ↔ Movie combined features)"]
        MPNet["multi-qa-mpnet-base-dot-v1 Cosine Similarity\n(768-dim semantic dense embedding)"]
        Pop["Popularity Score\n((pop/1000 + votes/10000) / 2)"]
        
        S1Score["Stage 1 Score =\n0.30·IMDb + 0.20·TFIDF + 0.30·Semantic(MPNet) + 0.10·Popularity"]
    end

    subgraph Stage2Calc ["Stage 2: User Personalization & Dislike Penalties"]
        HistBoost["History Match Boost:\n+ 0.30·ContentSim + 0.15·KeywordOverlap"]
        NegPenalty["Blended Multi-Feature Negative Penalty:\n1.0 - min(0.9, 0.75·max(Neg) + 0.25·avg(Neg))"]
        GenreGate{"Matches Target Query Genre?"}
        
        S2Score["Final Score =\nStage1Score · (1 + HistBoost) · NegPenalty"]
        DiscountScore["Discounted Score =\nStage1Score · 0.20 · NegPenalty"]
    end

    subgraph SelectionStage ["Stage 3: Diversity & Single Pick Selection"]
        DiversityCheck["DiversityFilter\nCap max same genre (3) & director (2)"]
        SessionExclude["Filter Out Excluded Movie IDs\n(Already shown or rejected in session)"]
        SingleTopPick["top_k = 1 Selection\nYields the single definitive #1 movie recommendation"]
    end

    Candidates --> IMDb & TFIDF & MPNet & Pop
    IMDb & TFIDF & MPNet & Pop --> S1Score
    S1Score --> Top100Candidates["Filter Top 100 Stage-1"]
    
    Top100Candidates --> GenreGate
    GenreGate -- Yes --> HistBoost & NegPenalty --> S2Score
    GenreGate -- No --> DiscountScore
    
    S2Score & DiscountScore --> DiversityCheck
    DiversityCheck --> SessionExclude
    SessionExclude --> SingleTopPick
```

---

## 5. Offline Precomputation Pipeline Flowchart

```mermaid
flowchart TD
    subgraph RawData ["Raw Datasets (data/)"]
        TMDBMovies["tmdb_5000_movies.csv\n(title, overview, genres, keywords)"]
        TMDBCredits["tmdb_5000_credits.csv\n(cast, crew, directors)"]
        MovieLens["genome-tags.csv & genome-scores.csv\nlinks.csv (ID mapping)"]
    end

    subgraph FeatureEngineering ["Feature Engineering (FeatureMatrixBuilder)"]
        Merge["Rename 'movie_id' to 'id' & Merge on ID → Extract Cast & Director"]
        TagJoin["Map MovieLens Tags (Top 20 relevant tags per film)"]
        CombinedText["Build Combined Text Document per Movie:\noverview + keywords + genres + director + cast + movielens_tags"]
    end

    subgraph Vectorization ["Matrix Generation & 768-Dim Dense Embeddings"]
        CountVec["CountVectorizer (1-2 grams, 5000 features)"]
        TFIDFVec["TfidfVectorizer (min_df=2, max_df=0.95)"]
        MPNetModel["SentenceTransformer\n(multi-qa-mpnet-base-dot-v1, 768-dim)"]
        SemIndex["Top-50 Cosine Nearest Neighbors Index"]
    end

    subgraph PickledArtifacts ["Serialized Artifacts (data/processed/)"]
        P1["feature_matrix.pkl (768-dim embeddings)"]
        P2["semantic_index.pkl"]
        P3["count_vectorizer.pkl"]
        P4["tfidf_vectorizer.pkl"]
    end

    TMDBMovies & TMDBCredits --> Merge
    MovieLens & Merge --> TagJoin
    TagJoin --> CombinedText

    CombinedText --> CountVec --> P3
    CombinedText --> TFIDFVec --> P4
    CombinedText --> MPNetModel --> P1
    MPNetModel --> SemIndex --> P2
```

---

## 6. Architectural Component Summary

| Component | Source File | Key Responsibilities |
| :--- | :--- | :--- |
| **Interactive Web Application** | `src/ui/web_app.py`<br>`src/ui/static/index.html` | Serves FastAPI UI on `:8080` with single-movie hero showcase, real-time "Not For Me / Suggest Another" skip loop, session history strip, and multi-dimensional profile drawer. |
| **Pipeline Orchestrator** | `src/pipeline/recommender.py` | Coordinates candidate discovery, feature extraction, 2-stage ranking, diversity filtering, exclusion filtering, and single-movie recommendation delivery. |
| **Memory Subsystem** | `src/memory/service.py`<br>`src/memory/engine.py`<br>`src/memory/storage.py` | Manages SQLite event ledger (`memory.db`), 7-day half-life decay, asymptotic long-term preference updates, Smooth Beta balancing, and multi-dimensional tag/theme/tone dislike learning. |
| **Semantic Search & Vectorization** | `src/features/semantic_search.py`<br>`src/features/matrix_builder.py` | Encodes queries and items with `multi-qa-mpnet-base-dot-v1` (768-dim dense representations), builds WordNet synonym expansions, and manages TF-IDF matrices. |
| **Ranking Engine** | `src/ranking/ranker.py` | Calculates Stage 1 query relevance, Stage 2 personalization boosts, blended multi-feature negative preference penalties, and applies `DiversityFilter`. |
| **NL & Preference Extraction** | `src/llm/nl_feature_extractor.py`<br>`src/features/preference_extractor.py` | Parses natural language requests into structured TMDB parameters; extracts genre, tag, theme, and tone affinity vectors from user history. |
| **Live TMDB Client** | `src/tmdb/client.py` | Asynchronously queries TMDB REST endpoints for live movies, details, watch providers (with name-to-numeric-ID resolution), credits, and keywords. |
| **LLM Reasoning Engine** | `src/llm/reasoning.py` | Generates decisive, target-constrained #1 top pick verdicts and batch explanations (*"Because you enjoyed..."*) using DeepSeek/OpenRouter LLMs or template fallback. |
| **Interfaces (CLI, MCP, A2A)** | `main.py`<br>`mcp_server.py`<br>`a2a_protocol.py` | Exposes the recommender as a single-recommendation CLI (`main.py recommend` & `main.py reject`), an MCP server on `:8000`, or an A2A protocol endpoint. |
