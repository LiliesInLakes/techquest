# Movie Recommender Agent

An agentic movie recommendation system for hospitality and airline businesses that provides personalized recommendations based on user subscriptions, watch history, and preferences.

## Architecture

- **MCP Compatible**: Exposes tools via Model Context Protocol
- **A2A Compatible**: Implements Agent-to-Agent protocol for inter-agent communication
- **TMDB Integration**: Fetches movie data, genres, keywords, watch providers
- **Feature Engineering**: CountVectorizer + TF-IDF + Semantic Embeddings (Sentence Transformers)
- **Semantic Search**: Synonym-aware keyword matching using WordNet + Embeddings
- **Ranking**: Multi-factor scoring (IMDb, TF-IDF, Watch History, Semantic, Content, Keywords)
- **LLM Reasoning**: DeepSeek API for natural language recommendation explanations

## Project Structure

```
TechQuest/
├── main.py                    # Entry point (MCP server, A2A agent, CLI)
├── mcp_server.py              # MCP server implementation
├── a2a_protocol.py            # A2A protocol implementation
├── config.yaml                # Configuration
├── requirements.txt           # Dependencies
├── .env.example               # Environment variables template
├── src/
│   ├── tmdb/client.py         # TMDB API client
│   ├── features/
│   │   ├── matrix_builder.py  # Feature matrix (CountVectorizer, TF-IDF, Embeddings)
│   │   ├── semantic_search.py # Semantic keyword search with synonyms
│   │   └── preference_extractor.py # User preference profiling
│   ├── ranking/ranker.py      # Multi-factor ranking algorithm
│   ├── llm/reasoning.py       # DeepSeek LLM reasoning
│   └── pipeline/recommender.py # Main recommendation pipeline
├── tests/
│   └── test_recommender.py    # Unit tests
└── data/
    └── processed/             # Cached feature matrices
```

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure API keys**:
```bash
cp .env.example .env
# Edit .env with your TMDB_API_KEY and DEEPSEEK_API_KEY
```

3. **Download datasets** (place in `data/`):
- TMDB 5000: `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`
- MovieLens: `genome-tags.csv`, `genome-scores.csv`, `links.csv`

4. **Precompute feature matrix**:
```bash
python main.py precompute
```

## Running

### MCP Server (port 8000)
```bash
python main.py mcp
```

### A2A Agent
```bash
python main.py a2a
```

### CLI Recommendations
```bash
# Search movies
python main.py recommend --user-id user123 --genre Action --region US --subscriptions netflix disney

# With watch history
python main.py recommend --user-id user123 --watch-history watch_history.json --top-k 10
```

Watch history format (JSON):
```json
[
  {"movie_id": 550, "rating": 9.0},
  {"movie_id": 680, "rating": 8.5}
]
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_movies` | Search by genre, region, language, subscriptions |
| `get_recommendations` | Personalized recommendations from watch history |
| `get_movie_details` | Full movie details with credits/keywords |
| `analyze_user_preferences` | Extract preference profile from history |

## A2A Capabilities

- `search_movies` - Discover movies matching criteria
- `get_recommendations` - Personalized recommendations with LLM reasoning
- `analyze_preferences` - User taste profiling

## Configuration (config.yaml)

```yaml
recommendation:
  top_k_movies: 100
  final_recommendations: 10
  imdb_weight: 0.3
  tfidf_weight: 0.2
  watch_history_weight: 0.5
  semantic_similarity_threshold: 0.7
```

## Ranking Algorithm

Final score combines:
- **IMDb Rating** (30%): Normalized vote average with vote count adjustment
- **TF-IDF Similarity** (20%): Query-movie textual similarity
- **Watch History** (50%): Content similarity + keyword overlap + semantic match
- **Popularity Boost**: Minor boost for popular/well-voted movies

## LLM Reasoning

Generates natural explanations like:
> "Because you love sci-fi action movies, The Matrix is a perfect match — it's an 8.7/10 rated film that combines the futuristic themes and intense action sequences you've consistently rated highly."

Falls back to template engine if DeepSeek unavailable.

## Testing

```bash
pytest tests/ -v
```