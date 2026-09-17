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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class ExtractedFeatures:
    # --- HARD FILTERS (Mapped to TMDB) ---
    genres: List[str] = field(default_factory=list)               # Primary genres (backward-compatible)
    primary_genres: List[str] = field(default_factory=list)       # Primary genres
    explicit_genres: List[str] = field(default_factory=list)      # Explicitly requested genre constraints
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
    include_adult: Optional[bool] = None

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

        if not self.explicit_genres and self.primary_genres:
            self.explicit_genres = list(self.primary_genres)

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
            "explicit_genres": self.explicit_genres,
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
            "include_adult": self.include_adult,
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

    # General colloquial, alternate, abbreviated, and hyphenated genre expressions
    GENRE_ALIASES: Dict[str, Any] = {
        # Science Fiction
        "sci-fi": "Science Fiction", "scifi": "Science Fiction", "science-fiction": "Science Fiction",
        "sciencefiction": "Science Fiction", "space": "Science Fiction", "cyberpunk": "Science Fiction",
        "dystopian": "Science Fiction", "dystopia": "Science Fiction", "post-apocalyptic": "Science Fiction",
        "time travel": "Science Fiction", "space opera": "Science Fiction",
        # Romance / Comedy / Romcom
        "romcom": ["Comedy", "Romance"], "rom-com": ["Comedy", "Romance"], "romantic comedy": ["Comedy", "Romance"],
        "romantic": "Romance",
        # Horror
        "spooky": "Horror", "scary": "Horror", "slasher": "Horror", "creepy": "Horror", "supernatural": "Horror",
        # Action
        "superhero": "Action", "martial arts": "Action", "action-packed": "Action",
        # Animation
        "animated": "Animation", "anime": "Animation", "cartoon": "Animation",
        # Family
        "kids": "Family", "children": "Family",
        # Thriller / Mystery
        "suspense": "Thriller", "psychological thriller": "Thriller",
        "detective": "Mystery", "whodunit": "Mystery", "whodunnit": "Mystery",
        # Crime
        "heist": "Crime", "gangster": "Crime", "mafia": "Crime", "mob": "Crime",
        # History
        "historical": "History", "period": "History", "period piece": "History", "biopic": "History", "biography": "History",
        # Music
        "musical": "Music",
        # Documentary
        "doc": "Documentary", "docuseries": "Documentary",
        # Comedy
        "funny": "Comedy", "hilarious": "Comedy", "sitcom": "Comedy",
        # Western
        "cowboy": "Western"
    }

    def __init__(self, config_path: str = "config.yaml", semantic_search=None):
        self.config = self._load_config(config_path)
        kado_config = self.config.get("kado", {})
        deepseek_config = self.config.get("deepseek", {})

        kado_key = os.getenv("KADO_API_KEY") or kado_config.get("api_key")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY") or deepseek_config.get("api_key")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        # Determine primary API key (prefer KADO_API_KEY, fallback to legacy DEEPSEEK_API_KEY)
        api_key = None
        if kado_key and kado_key != "${KADO_API_KEY}" and "your_kado_api_key" not in kado_key:
            api_key = kado_key
        elif deepseek_key and deepseek_key != "${DEEPSEEK_API_KEY}" and "your_deepseek_api_key" not in deepseek_key:
            api_key = deepseek_key

        default_base_url = "https://awesome.kado.so/openai/v1"
        default_model = "kado"

        if OPENAI_AVAILABLE and api_key:
            self.api_key = api_key
            self.base_url = os.getenv("KADO_BASE_URL") or kado_config.get("base_url") or os.getenv("DEEPSEEK_BASE_URL") or deepseek_config.get("base_url") or default_base_url
            self.model = os.getenv("KADO_MODEL") or kado_config.get("model") or os.getenv("DEEPSEEK_MODEL") or deepseek_config.get("model") or default_model
            self._llm_available = True
        elif OPENAI_AVAILABLE and openrouter_key and openrouter_key != "${OPENROUTER_API_KEY}" and "your_openrouter_api_key" not in openrouter_key:
            self.api_key = openrouter_key
            self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
            self._llm_available = True
        else:
            self.api_key = None
            self.base_url = default_base_url
            self.model = default_model
            self._llm_available = False
            if not OPENAI_AVAILABLE:
                print("OpenAI client not installed, NL extraction uses regex fallback", file=sys.stderr)
            else:
                print("LLM API key not configured, NL extraction uses regex fallback", file=sys.stderr)

        self.client = AsyncOpenAI(
            api_key=self.api_key or "no-key",
            base_url=self.base_url
        ) if self._llm_available else None

        self.semantic_search = semantic_search
        rec_config = self.config.get("recommendation", {}) if self.config else {}
        self.latest_window_years = int(rec_config.get("recency_latest_window_years", 2))
        self.recent_window_years = int(rec_config.get("recency_recent_window_years", 4))

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
                max_tokens=800
            )

            content = response.choices[0].message.content

            # Reasoning models (e.g. DeepSeek-V4-Flash via Kado) may return an
            # empty content field and put the JSON inside the `reasoning` field.
            if not content or not content.strip():
                raw_reasoning = getattr(response.choices[0].message, "reasoning", None)
                if raw_reasoning:
                    content = raw_reasoning

            parsed = self._try_parse_json(content)
            if parsed is not None:
                return self._parse_and_validate(parsed, text)

            print(f"LLM returned unparseable response, using fallback", file=sys.stderr)
            return self._fallback_extraction(text, context)

        except Exception as e:
            print(f"LLM extraction failed: {e}, using fallback", file=sys.stderr)
            return self._fallback_extraction(text, context)

    @staticmethod
    def _clean_json_response(text: str) -> str:
        if not text:
            return ""
        # Remove markdown code block fences (```json ... ``` or ``` ... ```)
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```', '', text)
        text = text.strip()
        # Find the outermost JSON object if surrounded by text
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            text = text[brace_start:brace_end + 1]
        # Replace single quotes with double quotes (only where valid in JSON)
        text = re.sub(r"(?<!\\)'", '"', text)
        # Remove trailing commas before closing braces/brackets
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        return text.strip()

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict]:
        cleaned = NLFeatureExtractor._clean_json_response(text)
        # Attempt 1: standard json.loads
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Attempt 2: ast.literal_eval for Python-style dicts
        try:
            import ast
            result = ast.literal_eval(cleaned)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return None

    def _extract_temporal_constraints(self, text: str) -> tuple[Optional[int], Optional[int], List[str]]:
        import datetime
        text_lower = text.lower()
        decades = []
        year_min = None
        year_max = None
        current_year = datetime.date.today().year

        # 1. Explicit year ranges (e.g., "from 1995 to 2005", "between 1990 and 2000", "1995-2005")
        range_match = re.search(r'\b(?:from|between)\s+(19\d\d|20[0-2]\d)\s+(?:to|and|-)\s+(19\d\d|20[0-2]\d)\b', text_lower)
        if not range_match:
            range_match = re.search(r'\b(19\d\d|20[0-2]\d)\s*-\s*(19\d\d|20[0-2]\d)\b', text_lower)

        if range_match:
            y1, y2 = int(range_match.group(1)), int(range_match.group(2))
            year_min = min(y1, y2)
            year_max = max(y1, y2)
        else:
            # 2. Strict / Inclusive operators
            after_match = re.search(r'\b(?:after|post)\s+(19\d\d|20[0-2]\d)\b', text_lower)
            if after_match:
                year_min = int(after_match.group(1)) + 1

            since_match = re.search(r'\b(?:since|from)\s+(19\d\d|20[0-2]\d)\b', text_lower)
            if not after_match and since_match:
                year_min = int(since_match.group(1))

            plus_match = re.search(r'\b(19\d\d|20[0-2]\d)\+\b', text_lower)
            if not after_match and not since_match and plus_match:
                year_min = int(plus_match.group(1))

            before_match = re.search(r'\b(?:before|prior to|pre)\s+(19\d\d|20[0-2]\d)\b', text_lower)
            if before_match:
                year_max = int(before_match.group(1)) - 1

            until_match = re.search(r'\b(?:until|through|up to)\s+(19\d\d|20[0-2]\d)\b', text_lower)
            if not before_match and until_match:
                year_max = int(until_match.group(1))

            # Decades check: "90s", "80s", "1990s", "2000s"
            decade_matches = re.findall(r'\b(19\d{2}|20[0-2]\d)s\b', text_lower)
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

            # Single explicit year (e.g. "2024 horror movie", "in 2024", "from 2024")
            if year_min is None and year_max is None and not decades:
                single_year_match = re.search(r'\b(19\d\d|20[0-2]\d)\b', text_lower)
                if single_year_match:
                    y = int(single_year_match.group(1))
                    year_min = y
                    year_max = y

            # 3. Qualitative Recency terms (only if explicit years were not extracted)
            if year_min is None and year_max is None and not decades:
                if re.search(r'\bthis year\b', text_lower):
                    year_min = current_year
                    year_max = current_year
                elif re.search(r'\blast year\b', text_lower):
                    year_min = current_year - 1
                    year_max = current_year - 1
                elif re.search(r'\b(latest|newest)\b', text_lower):
                    year_min = current_year - self.latest_window_years + 1
                    year_max = current_year
                elif re.search(r'\b(recent|recently released|recent releases|recently)\b', text_lower):
                    year_min = current_year - self.recent_window_years + 1
                    year_max = current_year
                else:
                    new_movie_match = re.search(
                        r'\bnew\s+(?:horror|action|comedy|drama|sci-fi|scifi|science fiction|thriller|romance|fantasy|animation|mystery|crime|adventure|western|movie|film|cinema|release)s?\b',
                        text_lower
                    ) or re.search(r'\b(?:brand\s+new|brand-new)\s+(?:movie|film|release)s?\b', text_lower)
                    if new_movie_match:
                        year_min = current_year - self.latest_window_years + 1
                        year_max = current_year

        return year_min, year_max, list(dict.fromkeys(decades))

    def _get_system_prompt(self) -> str:
        import datetime
        current_year = datetime.date.today().year
        latest_min = current_year - self.latest_window_years + 1
        recent_min = current_year - self.recent_window_years + 1
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
  "original_language": null,
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
   - "latest" / "newest" -> release_year_min: {latest_min}, release_year_max: {current_year}
   - "recent" / "recently released" -> release_year_min: {recent_min}, release_year_max: {current_year}
   - "this year" -> release_year_min: {current_year}, release_year_max: {current_year}
   - "last year" -> release_year_min: {current_year - 1}, release_year_max: {current_year - 1}
   - "new" (when modifying movie/genre) -> release_year_min: {latest_min}, release_year_max: {current_year}
   - "after YYYY" -> release_year_min: YYYY + 1 (strictly after)
   - "before YYYY" -> release_year_max: YYYY - 1 (strictly before)
   - "from YYYY to YYYY" -> release_year_min: YYYY, release_year_max: YYYY (inclusive)
   - "90s" / "nineties" -> decades: ["1990s"], release_year_min: 1990, release_year_max: 1999
   - "80s" -> decades: ["1980s"], release_year_min: 1980, release_year_max: 1989
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
9. LANGUAGE:
   - Set "original_language" (ISO 639-1 code e.g. "en", "fr", "ja", "ko", "es") ONLY if the user explicitly specifies a language or foreign origin in their query (e.g. "French movie", "Korean cinema", "in Spanish"). If no language is explicitly requested, always set "original_language": null.
10. CONCRETE KEYWORDS VS MOOD/THEMES:
   - "tmdb_keywords" must contain only concrete, literal plot subjects, settings, or entities (e.g. "space", "time travel", "heist", "submarine", "mars", "artificial intelligence").
   - Do NOT convert soft stylistic moods, tones, or adjectives (e.g. "mind-bending", "cozy", "dark", "atmospheric", "philosophical", "epic") into fabricated literal keywords (e.g. never map "mind-bending" to "mind control"). Soft concepts belong in "mood", "theme_or_plot", and "semantic_search_query".
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
        raw_keywords = parsed.get("tmdb_keywords") or parsed.get("keywords", [])

        # Filter out accidental fabricated keywords matching mood concepts
        soft_mood_terms = {"mind control", "mind-bending", "mind bending", "cozy", "dark", "atmospheric", "feel good", "feel-good"}
        keywords = [k for k in raw_keywords if str(k).lower().strip() not in soft_mood_terms]

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

        # Post-validate / refine temporal bounds using deterministic extractor
        det_min, det_max, det_decades = self._extract_temporal_constraints(original_text)
        if det_min is not None:
            year_min = det_min
        if det_max is not None:
            year_max = det_max
        decades = self._validate_decades(parsed.get("decades", []))
        if det_decades and not decades:
            decades = det_decades

        runtime_max = parsed.get("max_runtime_minutes")
        runtime_min = parsed.get("min_runtime_minutes")
        runtime_max = int(runtime_max) if isinstance(runtime_max, (int, float)) else None
        runtime_min = int(runtime_min) if isinstance(runtime_min, (int, float)) else None

        lang_raw = parsed.get("original_language")
        if isinstance(lang_raw, str):
            lang_raw = lang_raw.strip().lower()
            if lang_raw in ["null", "none", "", "unspecified", "any"]:
                lang_raw = None

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
            original_language=lang_raw,
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
        canonical_map = {g.lower(): g for g in self.TMDB_GENRES}

        for g in genres:
            if not isinstance(g, str):
                continue
            cleaned = g.strip()
            if not cleaned:
                continue

            # 1. Exact canonical name match
            if cleaned in self.TMDB_GENRES:
                valid.append(cleaned)
                continue

            # 2. Case-insensitive canonical match
            norm = cleaned.lower()
            if norm in canonical_map:
                valid.append(canonical_map[norm])
                continue

            # 3. Normalized alias lookup (handling hyphens, spaces, underscores)
            alias_key = re.sub(r'[\s\-_]+', '-', norm).strip('-')
            alt_key = re.sub(r'[\s\-_]+', '', norm)

            matched = self.GENRE_ALIASES.get(norm) or self.GENRE_ALIASES.get(alias_key) or self.GENRE_ALIASES.get(alt_key)
            if matched:
                if isinstance(matched, list):
                    valid.extend(matched)
                else:
                    valid.append(matched)
                continue

            # 4. Fallback substring matching for compound phrases
            for c_norm, c_name in canonical_map.items():
                if re.search(rf'\b{re.escape(c_norm)}\b', norm):
                    valid.append(c_name)

        return list(dict.fromkeys(valid))

    def _validate_decades(self, decades: List[str]) -> List[str]:
        valid = []
        for d in decades:
            if not isinstance(d, str):
                continue
            m = re.match(r'^(19\d0|20[0-2]0)s$', d.strip())
            if m:
                valid.append(m.group(0))
        return valid

    def _derive_loose_genres(self, primary_genres: List[str]) -> List[str]:
        loose = []
        for genre in primary_genres:
            if genre in self.GENRE_AFFINITIES:
                for related in self.GENRE_AFFINITIES[genre]:
                    if related not in primary_genres and related not in loose:
                        loose.append(related)
        return loose[:3]

    def _build_query_text(self, features: ExtractedFeatures) -> str:
        parts = []
        if features.mood:
            parts.append(features.mood)
        if features.primary_genres:
            parts.extend(features.primary_genres)
        if features.theme_or_plot:
            parts.append(features.theme_or_plot)
        if features.keywords:
            parts.extend(features.keywords[:5])
        elif features.tmdb_keywords:
            parts.extend(features.tmdb_keywords[:5])
        return " ".join(parts).strip() if parts else features.original_text

    def _fallback_extraction(self, text: str, context: Dict[str, Any]) -> ExtractedFeatures:
        import datetime
        text_lower = text.lower()
        primary_genres = []
        excluded_genres = []

        genre_keywords = {
            "sci-fi": "Science Fiction", "scifi": "Science Fiction", "science fiction": "Science Fiction",
            "space": "Science Fiction", "action": "Action", "comedy": "Comedy", "funny": "Comedy",
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

        # Reference movies (e.g., "like Inception", "similar to Sicario", "movies similar to Harry Potter")
        reference_movies = []
        ref_patterns = [
            r'(?:movies?|films?|something)?\s*(?:similar to|like)\s+([A-Za-z0-9\s:\'-]+?)(?:$|,|\.|\s+after\s+\d{4}|\s+before\s+\d{4}|\s+from\s+\d{4}|\s+in\s+[a-z]+|\s+with\s+|\s+but\s+|\s+and\s+)',
            r'(?:similar movies? to)\s+([A-Za-z0-9\s:\'-]+?)(?:$|,|\.|\s+after\s+\d{4}|\s+before\s+\d{4}|\s+from\s+\d{4}|\s+in\s+[a-z]+|\s+with\s+|\s+but\s+|\s+and\s+)'
        ]
        for pat in ref_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for r in matches:
                cleaned_ref = r.strip().strip('"\'')
                # Filter out stopwords or general non-title words
                if cleaned_ref and len(cleaned_ref) > 2 and cleaned_ref.lower() not in {"that", "this", "it", "movie", "film", "movies", "films"}:
                    if cleaned_ref not in reference_movies:
                        reference_movies.append(cleaned_ref)

        # Parse decades & years using shared single-source-of-truth extractor
        year_min, year_max, decades = self._extract_temporal_constraints(text)

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

        # Mood extraction
        mood = None
        mood_patterns = [
            (r'\bmind[\s\-]bending\b', 'mind-bending'),
            (r'\bfunny\b', 'lighthearted'),
            (r'\bdark\b', 'dark'),
            (r'\bscary\b', 'thrilling'),
            (r'\bmysterious\b', 'thrilling'),
            (r'\bemotional\b', 'emotional'),
            (r'\bnostalgic\b', 'nostalgic'),
            (r'\bfeel[\s\-]good\b', 'feel-good'),
            (r'\bgritty\b', 'gritty'),
            (r'\btense\b', 'tense'),
            (r'\bexciting\b', 'thrilling'),
            (r'\bcozy\b', 'feel-good'),
            (r'\batmospheric\b', 'dark'),
            (r'\bphilosophical\b', 'mind-bending')
        ]
        for pat, m in mood_patterns:
            if re.search(pat, text_lower):
                mood = m
                break

        # Keywords extraction
        stopwords = {
            "i", "want", "to", "watch", "a", "an", "the", "movie", "film",
            "with", "about", "like", "similar", "something", "good", "great",
            "recommend", "give", "me", "show", "movies", "films", "please",
            "from", "under", "mins", "minutes", "hour", "hours", "but", "not", "without", "than",
            "latest", "newest", "recent", "recently", "released", "releases", "new", "this", "last", "year"
        }
        mood_and_genre_tokens = {
            "sci", "fi", "scifi", "mind", "bending", "dark", "funny", "scary",
            "emotional", "nostalgic", "feel", "good", "gritty", "tense", "cozy",
            "action", "comedy", "drama", "horror", "thriller", "romance", "fantasy",
            "animation", "documentary", "mystery", "crime", "adventure", "family", "war", "western"
        }

        # Collect tokens from reference movie titles so they don't pollute keywords
        ref_tokens = set()
        for ref in reference_movies:
            for token in re.findall(r'\b[a-z]{3,}\b', ref.lower()):
                ref_tokens.add(token)

        words = [
            w for w in re.findall(r'\b[a-z]{3,}\b', text_lower)
            if w not in stopwords and w not in mood_and_genre_tokens and w not in ref_tokens
        ]
        keywords = words[:15]

        # Actors / Directors capital word heuristic (strictly excluding reference movie title tokens)
        potential_names = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text)
        ref_joined_lower = " ".join(reference_movies).lower()
        actors = []
        for name in potential_names:
            name_lower = name.lower()
            if name_lower in ref_joined_lower or any(part in ref_joined_lower.split() for part in name_lower.split()):
                continue
            if name not in reference_movies and name not in actors:
                actors.append(name)
        actors = actors[:3]

        theme_or_plot = " ".join([w for w in words if w not in genre_keywords])
        if mood:
            theme_or_plot = f"{mood} {theme_or_plot}".strip()

        # Adult content check (only set True if explicitly requested as adult/sexually explicit content)
        include_adult = None
        if re.search(r'\b(include adult|nsfw|sexually explicit|xxx|erotic)\b', text_lower):
            include_adult = True
        elif re.search(r'\b(no adult|family friendly|sfw|safe for work)\b', text_lower):
            include_adult = False

        explicit_genres = []
        explicit_genre_words = {
            "sci-fi": "Science Fiction", "scifi": "Science Fiction", "science fiction": "Science Fiction",
            "action": "Action", "comedy": "Comedy", "drama": "Drama", "horror": "Horror",
            "thriller": "Thriller", "romance": "Romance", "romcom": "Comedy",
            "fantasy": "Fantasy", "animation": "Animation", "documentary": "Documentary",
            "mystery": "Mystery", "crime": "Crime", "adventure": "Adventure",
            "family": "Family", "war": "War", "western": "Western", "history": "History", "music": "Music"
        }
        for kw, g in explicit_genre_words.items():
            if re.search(rf'\b{re.escape(kw)}\b', text_lower):
                if g not in excluded_genres and g not in explicit_genres:
                    explicit_genres.append(g)

        features = ExtractedFeatures(
            primary_genres=primary_genres,
            explicit_genres=explicit_genres,
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
            include_adult=include_adult,
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
        subscriptions: Optional[List[str]] = None,
        region: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build strict parameters for TMDB /discover/movie.
        Only explicit hard constraints present in `features` or parameters are sent.
        """
        params: Dict[str, Any] = {
            "sort_by": "popularity.desc",
            "page": 1,
        }

        # Language: only filter if explicitly extracted or passed
        target_lang = features.original_language or language
        if target_lang:
            params["language"] = target_lang

        # Region & Watch Providers
        if region:
            params["region"] = region
            params["watch_region"] = region
        if subscriptions:
            params["with_watch_providers"] = subscriptions
            if "watch_region" not in params and region:
                params["watch_region"] = region

        # Explicit genre constraints
        explicit = features.explicit_genres
        if explicit:
            genre_ids = [str(self.TMDB_GENRES[g]) for g in explicit if g in self.TMDB_GENRES]
            if genre_ids:
                params["with_genres"] = ",".join(genre_ids)

        # Genre Exclusions (without_genres)
        excl_genres = features.excluded_genres or features.exclude_genres
        if excl_genres:
            excl_ids = [str(self.TMDB_GENRES[g]) for g in excl_genres if g in self.TMDB_GENRES]
            if excl_ids:
                params["without_genres"] = ",".join(excl_ids)

        # Cast & Crew
        if features.actors:
            params["with_cast"] = features.actors
        if features.directors:
            params["with_crew"] = features.directors

        # Adult content
        if features.include_adult is not None:
            params["include_adult"] = features.include_adult

        # Temporal dates
        if features.release_year_min and features.release_year_max and features.release_year_min == features.release_year_max:
            params["primary_release_year"] = features.release_year_min
        else:
            if features.release_year_min:
                params["primary_release_date_gte"] = f"{features.release_year_min}-01-01"
            if features.release_year_max:
                params["primary_release_date_lte"] = f"{features.release_year_max}-12-31"

        # Runtime constraints
        if features.max_runtime_minutes:
            params["with_runtime_lte"] = features.max_runtime_minutes
        if features.min_runtime_minutes:
            params["with_runtime_gte"] = features.min_runtime_minutes

        # Rating threshold
        if features.min_rating and features.min_rating > 0.0:
            params["min_rating"] = features.min_rating

        # Text-based date fallbacks if not set
        text = (features.original_text or "").lower()
        if "primary_release_date_gte" not in params and "primary_release_year" not in params:
            if "after 2024" in text or "since 2024" in text or "2024 or later" in text or "from 2024" in text or "post 2024" in text or "2024+" in text:
                params["primary_release_date_gte"] = "2024-01-01"
            elif "after 2023" in text or "since 2023" in text or "2023+" in text:
                params["primary_release_date_gte"] = "2023-01-01"
            elif features.decades:
                dec = features.decades[0]
                if dec.endswith('s') and dec[:-1].isdigit():
                    start_yr = int(dec[:-1])
                    params["primary_release_date_gte"] = f"{start_yr}-01-01"
                    params["primary_release_date_lte"] = f"{start_yr + 9}-12-31"

        return params

    def build_fallback_tmdb_params(
        self,
        features: "ExtractedFeatures",
        subscriptions: Optional[List[str]] = None,
        region: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Loosened query builder: uses explicit and loose genres with OR logic.
        """
        params: Dict[str, Any] = {
            "sort_by": "popularity.desc",
            "page": 1,
        }

        target_lang = features.original_language or language
        if target_lang:
            params["language"] = target_lang

        if region:
            params["region"] = region
            params["watch_region"] = region
        if subscriptions:
            params["with_watch_providers"] = subscriptions
            if "watch_region" not in params and region:
                params["watch_region"] = region

        combined_genres = list(features.explicit_genres or features.primary_genres)
        for lg in features.loose_genres:
            if lg not in combined_genres:
                combined_genres.append(lg)

        genre_ids = [str(self.TMDB_GENRES[g]) for g in combined_genres if g in self.TMDB_GENRES]
        if genre_ids:
            params["with_genres"] = "|".join(genre_ids)

        excl_genres = features.excluded_genres or features.exclude_genres
        if excl_genres:
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
