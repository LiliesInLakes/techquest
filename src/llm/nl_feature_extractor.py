import sys
import os
import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ExtractedFeatures:
    # --- HARD FILTERS (Mapped to TMDB) ---
    genres: List[str] = field(default_factory=list)               # Primary genres (backward-compatible)
    primary_genres: List[str] = field(default_factory=list)       # Explicit primary genres
    loose_genres: List[str] = field(default_factory=list)         # Expanded genres for fallback
    exclude_genres: List[str] = field(default_factory=list)       # Excluded genres (backward-compatible)
    excluded_genres: List[str] = field(default_factory=list)      # Alias for excluded genres
    actors: List[str] = field(default_factory=list)
    directors: List[str] = field(default_factory=list)
    decades: List[str] = field(default_factory=list)
    release_year_min: Optional[int] = None
    release_year_max: Optional[int] = None
    min_runtime_minutes: Optional[int] = None
    max_runtime_minutes: Optional[int] = None
    original_language: Optional[str] = None
    keywords: List[str] = field(default_factory=list)             # Keywords (backward-compatible)
    tmdb_keywords: List[str] = field(default_factory=list)        # Explicit TMDB searchable keywords
    exclude_keywords: List[str] = field(default_factory=list)
    min_rating: float = 0.0

    # --- SOFT FEATURES (For downstream Semantic Search, Vector DB & Final LLM) ---
    mood: Optional[str] = None
    theme_or_plot: str = ""
    reference_movies: List[str] = field(default_factory=list)
    semantic_search_query: str = ""

    # --- METADATA ---
    query_text: str = ""
    confidence: float = 0.0
    original_text: str = ""

    def __post_init__(self):
        # Synchronize primary_genres and genres
        if not self.genres and self.primary_genres:
            self.genres = list(self.primary_genres)
        elif not self.primary_genres and self.genres:
            self.primary_genres = list(self.genres)

        # Synchronize excluded_genres and exclude_genres
        if not self.exclude_genres and self.excluded_genres:
            self.exclude_genres = list(self.excluded_genres)
        elif not self.excluded_genres and self.exclude_genres:
            self.excluded_genres = list(self.exclude_genres)

        # Synchronize tmdb_keywords and keywords
        if not self.keywords and self.tmdb_keywords:
            self.keywords = list(self.tmdb_keywords)
        elif not self.tmdb_keywords and self.keywords:
            self.tmdb_keywords = list(self.keywords)

        # Synchronize semantic_search_query and query_text
        if not self.semantic_search_query and self.query_text:
            self.semantic_search_query = self.query_text
        elif not self.query_text and self.semantic_search_query:
            self.query_text = self.semantic_search_query

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genres": self.genres,
            "primary_genres": self.primary_genres,
            "loose_genres": self.loose_genres,
            "exclude_genres": self.exclude_genres,
            "excluded_genres": self.excluded_genres,
            "actors": self.actors,
            "directors": self.directors,
            "decades": self.decades,
            "release_year_min": self.release_year_min,
            "release_year_max": self.release_year_max,
            "min_runtime_minutes": self.min_runtime_minutes,
            "max_runtime_minutes": self.max_runtime_minutes,
            "original_language": self.original_language,
            "keywords": self.keywords,
            "tmdb_keywords": self.tmdb_keywords,
            "exclude_keywords": self.exclude_keywords,
            "min_rating": self.min_rating,
            "mood": self.mood,
            "theme_or_plot": self.theme_or_plot,
            "reference_movies": self.reference_movies,
            "semantic_search_query": self.semantic_search_query,
            "query_text": self.query_text,
            "confidence": self.confidence,
            "original_text": self.original_text
        }


class NLFeatureExtractor:
    TMDB_GENRES = {
        "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35,
        "Crime": 80, "Documentary": 99, "Drama": 18, "Family": 10751,
        "Fantasy": 14, "History": 36, "Horror": 27, "Music": 10402,
        "Mystery": 9648, "Romance": 10749, "Science Fiction": 878,
        "TV Movie": 10770, "Thriller": 53, "War": 10752, "Western": 37
    }

    # Query expansion fallback graph ("Loose genres = add more genres")
    GENRE_AFFINITIES = {
        "Action": ["Adventure", "Thriller", "Crime"],
        "Adventure": ["Action", "Fantasy", "Science Fiction"],
        "Animation": ["Family", "Adventure", "Comedy"],
        "Comedy": ["Romance", "Drama", "Family"],
        "Crime": ["Thriller", "Mystery", "Action"],
        "Documentary": ["History", "Drama"],
        "Drama": ["Romance", "History", "Thriller"],
        "Family": ["Animation", "Comedy", "Adventure"],
        "Fantasy": ["Adventure", "Science Fiction", "Action"],
        "History": ["War", "Drama"],
        "Horror": ["Thriller", "Mystery"],
        "Music": ["Drama", "Romance", "Comedy"],
        "Mystery": ["Thriller", "Crime", "Science Fiction"],
        "Romance": ["Comedy", "Drama"],
        "Science Fiction": ["Action", "Mystery", "Adventure"],
        "TV Movie": ["Drama", "Comedy"],
        "Thriller": ["Mystery", "Crime", "Action"],
        "War": ["History", "Action", "Drama"],
        "Western": ["Action", "Adventure", "Drama"]
    }

    def __init__(self, config_path: str = "config.yaml", semantic_search=None):
        self.config = self._load_config(config_path)
        deepseek_config = self.config.get("deepseek", {})

        self.api_key = os.getenv("DEEPSEEK_API_KEY") or deepseek_config.get("api_key")
        # Don't raise: without a key or library the extractor degrades to regex fallback
        # so the pipeline still runs end-to-end (lower extraction quality).
        self._llm_available = bool(
            OPENAI_AVAILABLE and self.api_key and self.api_key != "${DEEPSEEK_API_KEY}"
            and "your_deepseek_api_key" not in self.api_key
        )
        if not self._llm_available:
            print("DeepSeek/OpenAI not configured, NL extraction uses regex fallback", file=sys.stderr)

        self.base_url = deepseek_config.get("base_url", "https://api.deepseek.com/v1")
        self.model = deepseek_config.get("model", "deepseek-chat")

        self.client = AsyncOpenAI(
            api_key=self.api_key or "no-key",
            base_url=self.base_url
        ) if self._llm_available else None

        self.semantic_search = semantic_search

    def _load_config(self, config_path: str) -> Dict:
        if yaml is None:
            return {}
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception:
            return {}

    async def extract(self, text: str, context: Dict[str, Any] = None) -> ExtractedFeatures:
        context = context or {}

        if not self._llm_available or self.client is None:
            return self._fallback_extraction(text, context)

        prompt = self._build_extraction_prompt(text, context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            parsed = json.loads(content)
            return self._parse_and_validate(parsed, text)
            
        except Exception as e:
            print(f"LLM extraction failed: {e}, using fallback", file=sys.stderr)
            return self._fallback_extraction(text, context)

    def _get_system_prompt(self) -> str:
        genres_str = ", ".join(self.TMDB_GENRES.keys())
        return f"""You are an expert movie preference parser for a recommendation system.
Extract structured features from the user's natural language movie request.

VALID TMDB GENRES (19 official genres only):
{genres_str}

OUTPUT FORMAT (JSON only):
{{
  "primary_genres": ["OfficialGenre1", "OfficialGenre2"],
  "loose_genres": ["RelatedGenreA", "RelatedGenreB"],
  "excluded_genres": ["GenreToAvoid"],
  "actors": ["Actor Name"],
  "directors": ["Director Name"],
  "decades": ["1990s", "2000s"],
  "release_year_min": 1990,
  "release_year_max": 1999,
  "min_runtime_minutes": null,
  "max_runtime_minutes": 120,
  "original_language": "en",
  "tmdb_keywords": ["heist", "time travel", "space mission"],
  "exclude_keywords": ["gore", "zombie"],
  "min_rating": 0.0,
  "mood": "lighthearted|dark|thrilling|emotional|mind-bending|nostalgic|feel-good|gritty|tense",
  "theme_or_plot": "concise description of key plot tropes, themes or dilemmas",
  "reference_movies": ["Inception", "Interstellar"],
  "semantic_search_query": "synthetic query summarizing the plot vibe and style for vector embedding search",
  "confidence": 0.0-1.0
}}

RULES:
1. GENRE NORMALIZATION: Map all informal genre names to official TMDB genres:
   - "sci-fi" / "scifi" -> "Science Fiction"
   - "rom-com" / "romcom" -> ["Comedy", "Romance"]
   - "spooky" -> "Horror"
   - "superhero" -> "Action"
2. QUERY EXPANSION (LOOSE GENRES):
   - Always populate "loose_genres" with 1-3 related genres to fall back on if results are too narrow.
   - Example: For "mind-bending thriller", primary_genres: ["Thriller"], loose_genres: ["Science Fiction", "Mystery"].
3. TEMPORAL PARSING:
   - "90s" / "nineties" -> decades: ["1990s"], release_year_min: 1990, release_year_max: 1999
   - "80s" -> decades: ["1980s"], release_year_min: 1980, release_year_max: 1989
   - "recent" / "new" -> release_year_min: 2021, release_year_max: 2026
   - "classic" -> release_year_max: 1979
4. RUNTIME PARSING:
   - "under 2 hours" / "less than 2h" -> max_runtime_minutes: 120
   - "quick watch" / "short" -> max_runtime_minutes: 95
   - "epic" / "long movie" -> min_runtime_minutes: 150
5. EXCLUSIONS:
   - "not horror" -> excluded_genres: ["Horror"]
   - "no animation" -> excluded_genres: ["Animation"]
   - "no gore" -> exclude_keywords: ["gore"]
6. REFERENCE MOVIES:
   - "like Inception or Sicario" -> reference_movies: ["Inception", "Sicario"]
7. SEMANTIC SEARCH QUERY:
   - Dense, high-signal summary of style, vibe, and tropes (max 30 words) for dense embedding search.
8. MIN RATING:
   - "highly rated" / "good reviews" -> min_rating: 7.0; "masterpiece" -> 8.0; otherwise 0.0.
"""

    def _build_extraction_prompt(self, text: str, context: Dict[str, Any]) -> str:
        context_str = ""
        if context.get("subscriptions"):
            context_str += f"\nAvailable subscriptions: {', '.join(context['subscriptions'])}"
        if context.get("region"):
            context_str += f"\nRegion: {context['region']}"
        if context.get("language"):
            context_str += f"\nLanguage: {context['language']}"
        if context.get("watch_history_summary"):
            context_str += f"\nWatch history: {context['watch_history_summary']}"
        
        return f"User request: \"{text}\"{context_str}\n\nExtract features as JSON:"

    def _parse_and_validate(self, parsed: Dict, original_text: str) -> ExtractedFeatures:
        primary_genres = self._validate_genres(parsed.get("primary_genres") or parsed.get("genres", []))
        loose_genres = self._validate_genres(parsed.get("loose_genres", []))
        if not loose_genres:
            loose_genres = self._derive_loose_genres(primary_genres)

        excluded_genres = self._validate_genres(parsed.get("excluded_genres") or parsed.get("exclude_genres", []))
        keywords = parsed.get("tmdb_keywords") or parsed.get("keywords", [])
        
        # Validate year bounds
        year_min = parsed.get("release_year_min")
        year_max = parsed.get("release_year_max")
        if isinstance(year_min, (int, float)):
            year_min = int(year_min)
        else:
            year_min = None
        if isinstance(year_max, (int, float)):
            year_max = int(year_max)
        else:
            year_max = None

        # Validate runtimes
        runtime_max = parsed.get("max_runtime_minutes")
        runtime_min = parsed.get("min_runtime_minutes")
        runtime_max = int(runtime_max) if isinstance(runtime_max, (int, float)) else None
        runtime_min = int(runtime_min) if isinstance(runtime_min, (int, float)) else None

        features = ExtractedFeatures(
            primary_genres=primary_genres,
            loose_genres=loose_genres,
            excluded_genres=excluded_genres,
            actors=parsed.get("actors", [])[:10],
            directors=parsed.get("directors", [])[:5],
            decades=self._validate_decades(parsed.get("decades", [])),
            release_year_min=year_min,
            release_year_max=year_max,
            min_runtime_minutes=runtime_min,
            max_runtime_minutes=runtime_max,
            original_language=parsed.get("original_language"),
            tmdb_keywords=keywords[:20],
            exclude_keywords=parsed.get("exclude_keywords", [])[:10],
            min_rating=max(0.0, min(10.0, float(parsed.get("min_rating", 0.0)))),
            mood=parsed.get("mood"),
            theme_or_plot=parsed.get("theme_or_plot", ""),
            reference_movies=parsed.get("reference_movies", [])[:5],
            semantic_search_query=parsed.get("semantic_search_query") or parsed.get("query_text", original_text)[:500],
            confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.7)))),
            original_text=original_text
        )
        
        if not features.semantic_search_query:
            features.semantic_search_query = self._build_query_text(features)
        
        return features

    def _validate_genres(self, genres: List[str]) -> List[str]:
        valid = []
        genre_map = {g.lower(): g for g in self.TMDB_GENRES}
        genre_map.update({
            "sci-fi": "Science Fiction", "scifi": "Science Fiction",
            "science-fiction": "Science Fiction",
            "romcom": "Comedy", "rom-com": "Comedy",
            "action": "Action", "thriller": "Thriller",
            "horror": "Horror", "comedy": "Comedy",
            "drama": "Drama", "romance": "Romance",
            "fantasy": "Fantasy", "animation": "Animation",
            "animated": "Animation", "anime": "Animation",
            "documentary": "Documentary", "mystery": "Mystery",
            "crime": "Crime", "adventure": "Adventure",
            "family": "Family", "war": "War", "western": "Western",
            "history": "History", "historical": "History",
            "music": "Music", "musical": "Music",
            "superhero": "Action", "spooky": "Horror"
        })
        
        for g in genres:
            g_lower = g.lower().strip()
            if g_lower in genre_map:
                mapped = genre_map[g_lower]
                if mapped not in valid:
                    valid.append(mapped)
            elif g in self.TMDB_GENRES and g not in valid:
                valid.append(g)
        
        return valid[:5]

    def _derive_loose_genres(self, primary_genres: List[str]) -> List[str]:
        loose = []
        for g in primary_genres:
            affinities = self.GENRE_AFFINITIES.get(g, [])
            for aff in affinities:
                if aff not in primary_genres and aff not in loose:
                    loose.append(aff)
        return loose[:4]

    def _validate_decades(self, decades: List[str]) -> List[str]:
        valid = []
        for d in decades:
            d_clean = d.strip().lower()
            if d_clean.endswith('s') and d_clean[:-1].isdigit():
                year = int(d_clean[:-1])
                if 1900 <= year <= 2030:
                    valid.append(f"{year}s")
            elif d_clean.isdigit():
                year = int(d_clean)
                if 1900 <= year <= 2030:
                    valid.append(f"{(year // 10) * 10}s")
        return valid[:3]

    def _build_query_text(self, features: ExtractedFeatures) -> str:
        parts = []
        parts.extend(features.primary_genres)
        parts.extend(features.tmdb_keywords)
        parts.extend(features.actors)
        parts.extend(features.directors)
        parts.extend(features.decades)
        if features.mood:
            parts.append(features.mood)
        if features.theme_or_plot:
            parts.append(features.theme_or_plot)
        return " ".join(parts)

    def _fallback_extraction(self, text: str, context: Dict[str, Any]) -> ExtractedFeatures:
        text_lower = text.lower()
        primary_genres = []
        excluded_genres = []
        
        genre_keywords = {
            "sci-fi": "Science Fiction", "scifi": "Science Fiction", "space": "Science Fiction",
            "action": "Action", "comedy": "Comedy", "funny": "Comedy", "humor": "Comedy",
            "drama": "Drama", "horror": "Horror", "scary": "Horror",
            "thriller": "Thriller", "suspense": "Thriller",
            "romance": "Romance", "romantic": "Romance", "romcom": "Comedy",
            "fantasy": "Fantasy", "magic": "Fantasy",
            "animation": "Animation", "animated": "Animation", "anime": "Animation",
            "documentary": "Documentary", "mystery": "Mystery", "detective": "Mystery",
            "crime": "Crime", "heist": "Crime", "gangster": "Crime",
            "adventure": "Adventure", "family": "Family", "kids": "Family",
            "war": "War", "western": "Western", "history": "History"
        }
        
        # Check negative patterns (e.g. "not horror", "no horror", "without gore")
        neg_patterns = re.findall(r'(?:no|not|without|avoid)\s+([a-z\-]+)', text_lower)
        for neg in neg_patterns:
            if neg in genre_keywords:
                matched_genre = genre_keywords[neg]
                if matched_genre not in excluded_genres:
                    excluded_genres.append(matched_genre)

        for kw, genre in genre_keywords.items():
            if re.search(rf'\b{re.escape(kw)}\b', text_lower):
                if genre not in excluded_genres and genre not in primary_genres:
                    primary_genres.append(genre)
        
        primary_genres = primary_genres[:3]
        loose_genres = self._derive_loose_genres(primary_genres)
        
        # Parse decades & years
        decades = []
        year_min = None
        year_max = None
        
        # Check "90s", "80s", "2000s"
        decade_matches = re.findall(r'\b(19\d{1}|20[0-2]\d)s\b', text_lower)
        if not decade_matches:
            decade_matches = re.findall(r'\b([2-9]0)s\b', text_lower)
            decade_matches = [f"19{m}" if int(m) >= 30 else f"20{m}" for m in decade_matches]
            
        for d in decade_matches:
            year = int(d)
            base_dec = (year // 10) * 10
            decades.append(f"{base_dec}s")
            if year_min is None or base_dec < year_min:
                year_min = base_dec
            if year_max is None or (base_dec + 9) > year_max:
                year_max = base_dec + 9
        
        # Runtime extraction
        max_runtime = None
        min_runtime = None
        runtime_hr_match = re.search(r'(?:under|less than|within)\s+(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)', text_lower)
        if runtime_hr_match:
            max_runtime = int(float(runtime_hr_match.group(1)) * 60)
        elif "quick watch" in text_lower or "short movie" in text_lower:
            max_runtime = 95
            
        runtime_min_match = re.search(r'(?:under|less than)\s+(\d+)\s*(?:mins?|minutes?)', text_lower)
        if runtime_min_match:
            max_runtime = int(runtime_min_match.group(1))

        # Language extraction
        lang_map = {
            "english": "en", "french": "fr", "korean": "ko",
            "spanish": "es", "japanese": "ja", "hindi": "hi",
            "german": "de", "italian": "it"
        }
        original_language = None
        for lang_name, code in lang_map.items():
            if lang_name in text_lower:
                original_language = code
                break

        # Reference movies (e.g., "like Inception", "similar to Sicario")
        reference_movies = []
        ref_match = re.findall(r'(?:like|similar to)\s+([A-Z][A-Za-z0-9\s]+?)(?:$|,|\.|\s+but|\s+and|\s+with)', text)
        for r in ref_match:
            cleaned_ref = r.strip()
            if cleaned_ref and len(cleaned_ref) > 2:
                reference_movies.append(cleaned_ref)

        # Mood extraction
        mood = None
        mood_keywords = {
            "funny": "lighthearted", "dark": "dark", "scary": "thrilling",
            "emotional": "emotional", "mind-bending": "mind-bending",
            "nostalgic": "nostalgic", "feel-good": "feel-good", "gritty": "gritty",
            "tense": "tense", "exciting": "thrilling", "cozy": "feel-good"
        }
        for kw, m in mood_keywords.items():
            if kw in text_lower:
                mood = m
                break
        
        # Keywords extraction
        stopwords = {
            "i", "want", "to", "watch", "a", "an", "the", "movie", "film", 
            "with", "about", "like", "similar", "something", "good", "great",
            "recommend", "give", "me", "show", "movies", "films", "please"
        }
        words = [w for w in re.findall(r'\b[a-z]{3,}\b', text_lower) if w not in stopwords]
        keywords = words[:15]
        
        # Actors / Directors simple capital word heuristic
        potential_names = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text)
        actors = [name for name in potential_names if name not in reference_movies][:3]

        theme_or_plot = " ".join([w for w in keywords if w not in genre_keywords])

        features = ExtractedFeatures(
            primary_genres=primary_genres,
            loose_genres=loose_genres,
            excluded_genres=excluded_genres,
            actors=actors,
            decades=list(set(decades))[:2],
            release_year_min=year_min,
            release_year_max=year_max,
            min_runtime_minutes=min_runtime,
            max_runtime_minutes=max_runtime,
            original_language=original_language,
            tmdb_keywords=keywords,
            mood=mood,
            theme_or_plot=theme_or_plot,
            reference_movies=reference_movies[:3],
            min_rating=6.0 if any(w in text_lower for w in ["good", "great", "best", "highly rated"]) else 0.0,
            semantic_search_query=" ".join(keywords + primary_genres),
            confidence=0.5,
            original_text=text
        )
        return features

    def get_tmdb_genre_ids(self) -> Dict[str, int]:
        return dict(self.TMDB_GENRES)

    def build_tmdb_discover_params(
        self,
        features: "ExtractedFeatures",
        subscriptions: List[str] = None,
        region: str = "US",
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Build strict / primary parameters for TMDB /discover/movie.
        """
        subscriptions = subscriptions or []
        primary = features.primary_genres or features.genres
        first_genre = primary[0] if primary else None

        params = {
            "genre": first_genre,
            "region": region,
            "language": language,
            "min_rating": features.min_rating or 0.0,
            "sort_by": "popularity.desc",
            "page": 1,
            "with_watch_providers": subscriptions or None,
            "watch_region": region,
        }

        # Multi-genre IDs (AND query)
        genre_ids = [str(self.TMDB_GENRES[g]) for g in primary if g in self.TMDB_GENRES]
        if genre_ids:
            params["with_genres"] = ",".join(genre_ids)

        # Exclusions (without_genres)
        excl_genres = features.excluded_genres or features.exclude_genres
        excl_ids = [str(self.TMDB_GENRES[g]) for g in excl_genres if g in self.TMDB_GENRES]
        if excl_ids:
            params["without_genres"] = ",".join(excl_ids)

        # Temporal dates
        if features.release_year_min:
            params["primary_release_date_gte"] = f"{features.release_year_min}-01-01"
        if features.release_year_max:
            params["primary_release_date_lte"] = f"{features.release_year_max}-12-31"

        # Runtime constraints
        if features.max_runtime_minutes:
            params["with_runtime_lte"] = features.max_runtime_minutes
        if features.min_runtime_minutes:
            params["with_runtime_gte"] = features.min_runtime_minutes

        # Language
        if features.original_language:
            params["language"] = features.original_language

        return params

    def build_fallback_tmdb_params(
        self,
        features: "ExtractedFeatures",
        subscriptions: List[str] = None,
        region: str = "US",
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Loosened query builder: switches genre matching from AND (comma) to OR (pipe |)
        and merges in loose_genres to guarantee a rich candidate pool.
        """
        subscriptions = subscriptions or []
        primary = features.primary_genres or features.genres
        combined_genres = list(primary)
        for lg in features.loose_genres:
            if lg not in combined_genres:
                combined_genres.append(lg)

        genre_ids = [str(self.TMDB_GENRES[g]) for g in combined_genres if g in self.TMDB_GENRES]

        params = {
            "genre": primary[0] if primary else None,
            "region": region,
            "language": language,
            "min_rating": 0.0,
            "sort_by": "popularity.desc",
            "page": 1,
            "with_watch_providers": subscriptions or None,
            "watch_region": region,
        }

        # Pipe separated = OR logic
        if genre_ids:
            params["with_genres"] = "|".join(genre_ids)

        # Keep hard exclusions
        excl_genres = features.excluded_genres or features.exclude_genres
        excl_ids = [str(self.TMDB_GENRES[g]) for g in excl_genres if g in self.TMDB_GENRES]
        if excl_ids:
            params["without_genres"] = ",".join(excl_ids)

        return params

    def map_to_vocabulary(self, features: ExtractedFeatures, vocab: Set[str]) -> ExtractedFeatures:
        if not self.semantic_search:
            return features
        
        mapped_keywords = []
        for kw in features.keywords:
            if kw.lower() in vocab:
                mapped_keywords.append(kw.lower())
            else:
                similar = self.semantic_search.find_similar_keywords(kw, top_k=3)
                for match, score in similar:
                    if score >= 0.7 and match in vocab:
                        mapped_keywords.append(match)
                        break
                else:
                    mapped_keywords.append(kw.lower())
        
        features.keywords = list(set(mapped_keywords))[:20]
        features.tmdb_keywords = list(features.keywords)
        return features


async def main():
    try:
        extractor = NLFeatureExtractor()
        
        test_queries = [
            "I want a funny 90s space comedy with Tom Hanks under 2 hours",
            "Dark gritty thriller like Sicario but not too violent and no horror",
            "Feel-good romantic comedy from the 2000s in French",
            "Mind-bending sci-fi with time travel, highly rated",
            "Action movie with Tom Cruise, not Mission Impossible"
        ]
        
        for query in test_queries:
            print(f"\n==========================================", file=sys.stderr)
            print(f"Query: {query}", file=sys.stderr)
            features = await extractor.extract(query)
            print(f"  Primary Genres: {features.primary_genres}", file=sys.stderr)
            print(f"  Loose Genres:   {features.loose_genres}", file=sys.stderr)
            print(f"  Excluded:       {features.excluded_genres}", file=sys.stderr)
            print(f"  Years:          {features.release_year_min} - {features.release_year_max}", file=sys.stderr)
            print(f"  Max Runtime:    {features.max_runtime_minutes} mins", file=sys.stderr)
            print(f"  Language:       {features.original_language}", file=sys.stderr)
            print(f"  Mood:           {features.mood}", file=sys.stderr)
            print(f"  Theme:          {features.theme_or_plot}", file=sys.stderr)
            print(f"  References:     {features.reference_movies}", file=sys.stderr)
            print(f"  Semantic Query: {features.semantic_search_query}", file=sys.stderr)
            
            discover_params = extractor.build_tmdb_discover_params(features)
            print(f"  TMDB Params:    {discover_params}", file=sys.stderr)
            
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
