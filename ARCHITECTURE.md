# System Architecture & Flowchart Documentation

This document describes the complete architecture, pipeline flowcharts, memory subsystems, data flow, and module responsibilities of the **Movie Recommender Agent**.

---

## 1. High-Level System Architecture

```
                    ┌──────────────┐      ┌──────────────┐
                    │  CLI (main.py)│      │ MCP server   │      ┌─────────────┐
  User / Agent ─────▶│  Interactive │      │ (FastAPI     │────▶│ A2A agent   │
   or External App  │  & Discovery │      │  :8000)      │      │ (3 caps)    │
                    └──────┬───────┘      └──────┬───────┘      └──────┬──────┘
                           │                     │                     │
                           ▼                     ▼                     ▼
                ┌──────────────────────────────────────────────────────────────┐
                │              Pipeline: MovieRecommender                       │
                │  1. UserMemoryService / PreferenceExtractor → User Profile   │
                │  2. NLFeatureExtractor  → Structured features from NL query   │
                │  3. TMDBClient (or local pool) → Stage-1 Candidates          │
                │  4. MovieRanker.rank_movies_stage1 → Query-similarity rank   │
                │  5. MovieRanker.rank_movies_stage2 → Watch-history boost     │
                │  6. DiversityFilter → De-duplicate genres / directors        │
                │  7. DeepSeekReasoner / TemplateEngine → Explanations         │
                └──────┬───────────────┬───────────────────┬───────────────────┘
                       │               │                   │
                       ▼               ▼                   ▼
                ┌────────────┐ ┌──────────────┐ ┌────────────────────┐
                │ TMDB API   │ │ Offline pool │ │ LLM (DeepSeek /    │
                │ (Live)     │ │ data/        │ │ OpenRouter)        │
                │ Discover / │ │ processed/   │ │ Reasoning & NL     │
                │ Providers  │ │ feature_mat. │ │ Feature Parsing    │
                └────────────┘ └──────────────┘ └────────────────────┘
```

---

## 2. End-to-End Recommendation Pipeline Flowchart

```mermaid
flowchart TD
    subgraph ClientInterfaces ["1. Client Interfaces"]
        CLI["CLI (main.py)\nrecommend | profile | rate | interactive"]
        MCP["MCP Server (mcp_server.py)\nFastAPI :8000 (4 Tools)"]
        A2A["A2A Protocol (a2a_protocol.py)\nAgent-to-Agent (3 Capabilities)"]
    end

    subgraph InputProcessing ["2. Input & Feature Extraction"]
        UserReq["User Input\n• User ID\n• Natural Language Query\n• Watch History / Subscriptions"]
        NLParser{"NL Query Present?"}
        LLMExtract["NLFeatureExtractor\n(DeepSeek / OpenRouter JSON parse)"]
        RegexFallback["Regex Extraction Fallback\n(Genre / Decade / Keyword scan)"]
        VocabMap["map_to_vocabulary()\nSnap keywords to model vocabulary"]
        MemoryLoad["UserMemoryService\nLoad Combined Profile\n(Short-Term + Long-Term + Negatives)"]
    end

    subgraph CandidateRetrieval ["3. Candidate Discovery"]
        TMDBCheck{"TMDB Available?"}
        LiveTMDB["TMDBClient.discover_movies()\n• Filter: Genre, Release Dates, Runtime\n• Sort: popularity.desc / vote_average.desc\n• Exclude unwanted genres/keywords"]
        LocalPool["Local Candidate Pool\n(4,800+ Movies in data/processed/)"]
    end

    subgraph RankingEngine ["4. Two-Stage Ranking & Diversity"]
        Stage1["Stage 1: Fast Query Ranking\nScore = 0.30·IMDb + 0.20·TFIDF + 0.30·Semantic + 0.10·Popularity"]
        Top100["Select Top 100 Stage-1 Candidates"]
        Stage2["Stage 2: Personalization & Negative Penalties\n• Content Similarity + Keyword Overlap Boost\n• Blended Negative Penalty Reduction\n• Target Genre Retention"]
        DivFilter["DiversityFilter\nCap max same genre & director"]
        TopK["Select Top-K Ranked Movies"]
    end

    subgraph ExplanationGen ["5. Explanation & Output"]
        LLMReason{"LLM Available?"}
        DeepSeekExplain["DeepSeekReasoner\nBatch contextual 'Because you...' explanations"]
        TemplateExplain["ReasoningTemplateEngine\nRule-based prioritized templates"]
        FinalJSON["Return Final JSON Result\n• Ranked Movies & Scores\n• Personalized Reasonings\n• User Profile Stats"]
    end

    %% Connections
    CLI --> UserReq
    MCP --> UserReq
    A2A --> UserReq

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
    DivFilter --> TopK

    TopK --> LLMReason
    LLMReason -- Yes --> DeepSeekExplain
    LLMReason -- No / Error --> TemplateExplain

    DeepSeekExplain --> FinalJSON
    TemplateExplain --> FinalJSON
```

---

## 3. Adaptive Memory Subsystem Flowchart & Math

```mermaid
flowchart LR
    subgraph UserAction ["User Interaction"]
        Act["Interaction Occurs\n• Click (w=0.1)\n• Watch (w=0.3)\n• Watchlist (w=0.2)\n• Rate 1-10 stars"]
    end

    subgraph DatabaseLedger ["SQLite Storage (data/memory.db)"]
        Ledger["Append Event Record\n• Event Type / Signal Weight\n• Timestamp & Movie Metadata"]
    end

    subgraph DualMemoryUpdate ["Dual-Branch Memory Update"]
        direction TB
        ShortTerm["Short-Term Memory (P_short)\n• Exponential Time-Decay (7-Day Half-Life)\n• Immediate context / current mood"]
        LongTerm["Long-Term Memory (P_long)\n• Asymptotic Addition: P += (1-P)·w\n• Bayesian Confidence: N / (N + K)\n• Permanent affinity profile"]
        Negative["Negative Preferences (P_neg)\n• Triggered on ratings <= 2.0 / 5.0\n• Penalizes disliked tropes, genres, cast"]
    end

    subgraph BlendingEngine ["Smooth Beta Dynamic Balance"]
        BetaCalc["Dynamic Balance Calculation:\nβ(N) = 0.40 + 0.60 · max(0, 1 - N/10)"]
        Combined["Final Combined Preference:\nP_final = β · P_short + (1 - β) · P_long"]
    end

    Act --> Ledger
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

5. **Blended Negative Preference Penalty:**
   $$\text{PenaltyFactor} = 1.0 - \min\left(0.90, 0.75 \cdot \max_{f \in M}(\text{Neg}(f)) + 0.25 \cdot \frac{1}{|F|}\sum_{f \in M}\text{Neg}(f)\right)$$

---

## 4. Two-Stage Ranking Mathematics Flowchart

```mermaid
flowchart TD
    Candidates["Input Candidates (Live TMDB or Local Matrix Pool)"]

    subgraph Stage1Calc ["Stage 1: Content Relevance & Quality"]
        IMDb["IMDb Score\n(vote_avg / 10, halved if votes < 100)"]
        TFIDF["TF-IDF Cosine Similarity\n(Query text ↔ Movie combined features)"]
        Semantic["MiniLM Embedding Cosine Similarity\n(384-dim semantic representation)"]
        Pop["Popularity Score\n((pop/1000 + votes/10000) / 2)"]
        
        S1Score["Stage 1 Score =\n0.30·IMDb + 0.20·TFIDF + 0.30·Semantic + 0.10·Popularity"]
    end

    subgraph Stage2Calc ["Stage 2: User Personalization & Penalties"]
        HistBoost["History Match Boost:\n+ 0.30·ContentSim + 0.15·KeywordOverlap"]
        NegPenalty["Blended Negative Penalty:\n1.0 - min(0.9, 0.75·max(Neg) + 0.25·avg(Neg))"]
        GenreGate{"Matches Target Query Genre?"}
        
        S2Score["Final Score =\nStage1Score · (1 + HistBoost) · NegPenalty"]
        DiscountScore["Discounted Score =\nStage1Score · 0.20 · NegPenalty"]
    end

    subgraph Diversity ["Stage 3: Diversity Filter"]
        CapCheck{"Genre count < max_genre (5) AND\nDirector count < max_dir (2)?"}
        Accept["Keep Candidate"]
        Reject["Prune Candidate"]
    end

    Candidates --> IMDb & TFIDF & Semantic & Pop
    IMDb & TFIDF & Semantic & Pop --> S1Score
    S1Score --> Top100Candidates["Filter Top 100"]
    
    Top100Candidates --> GenreGate
    GenreGate -- Yes --> HistBoost & NegPenalty --> S2Score
    GenreGate -- No --> DiscountScore
    
    S2Score & DiscountScore --> CapCheck
    CapCheck -- Yes --> Accept
    CapCheck -- No --> Reject
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
        Merge["Merge on Movie ID → Extract Top Cast & Director"]
        TagJoin["Map MovieLens Tags (Top 20 relevant tags per film)"]
        CombinedText["Build Combined Text Document per Movie:\noverview + keywords + genres + director + cast + movielens_tags"]
    end

    subgraph Vectorization ["Matrix Generation & Embeddings"]
        CountVec["CountVectorizer (1-2 grams, 5000 features)"]
        TFIDFVec["TfidfVectorizer (min_df=2, max_df=0.95)"]
        MiniLM["SentenceTransformer (all-MiniLM-L6-v2, 384-dim)"]
        SemIndex["Top-50 Cosine Nearest Neighbors Index"]
    end

    subgraph PickledArtifacts ["Serialized Artifacts (data/processed/)"]
        P1["feature_matrix.pkl"]
        P2["semantic_index.pkl"]
        P3["count_vectorizer.pkl"]
        P4["tfidf_vectorizer.pkl"]
    end

    TMDBMovies & TMDBCredits --> Merge
    MovieLens & Merge --> TagJoin
    TagJoin --> CombinedText

    CombinedText --> CountVec --> P3
    CombinedText --> TFIDFVec --> P4
    CombinedText --> MiniLM --> P1
    MiniLM --> SemIndex --> P2
```

---

## 6. Architectural Component Summary

| Component | Source File | Key Responsibilities |
| :--- | :--- | :--- |
| **Pipeline Orchestrator** | `src/pipeline/recommender.py` | Coordinates candidate discovery, feature extraction, 2-stage ranking, diversity filtering, and explanation synthesis. |
| **Memory Subsystem** | `src/memory/service.py`<br>`src/memory/engine.py`<br>`src/memory/storage.py` | Manages SQLite event ledger (`memory.db`), 7-day half-life decay, asymptotic long-term preference updates, and Smooth Beta balancing. |
| **Ranking Engine** | `src/ranking/ranker.py` | Calculates Stage 1 query relevance, Stage 2 personalization boosts, negative preference penalties, and applies `DiversityFilter`. |
| **NL & Preference Extraction** | `src/llm/nl_feature_extractor.py`<br>`src/features/preference_extractor.py` | Parses natural language requests into structured TMDB parameters; extracts genre/keyword affinity vectors from user history. |
| **Live TMDB Client** | `src/tmdb/client.py` | Asynchronously queries TMDB REST endpoints for live movies, details, watch providers, credits, and keywords with fallback resilience. |
| **LLM Reasoning Engine** | `src/llm/reasoning.py` | Synthesizes natural-language explanations (*"Because you enjoyed..."*) using DeepSeek/OpenRouter LLMs or rule-based fallback templates. |
| **Interfaces (CLI, MCP, A2A)** | `main.py`<br>`mcp_server.py`<br>`a2a_protocol.py` | Exposes the recommender as an interactive CLI, an HTTP Model Context Protocol (MCP) server, or an Agent-to-Agent (A2A) protocol endpoint. |

