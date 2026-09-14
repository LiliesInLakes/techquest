import sys
import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from openai import AsyncOpenAI
import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ExtractedFeatures:
    genres: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    actors: List[str] = field(default_factory=list)
    directors: List[str] = field(default_factory=list)
    decades: List[str] = field(default_factory=list)
    mood: Optional[str] = None
    min_rating: float = 0.0
    exclude_genres: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)
    query_text: str = ""
    confidence: float = 0.0
    original_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genres": self.genres,
            "keywords": self.keywords,
            "actors": self.actors,
            "directors": self.directors,
            "decades": self.decades,
            "mood": self.mood,
            "min_rating": self.min_rating,
            "exclude_genres": self.exclude_genres,
            "exclude_keywords": self.exclude_keywords,
            "query_text": self.query_text,
            "confidence": self.confidence,
            "original_text": self.original_text
        }


class NLFeatureExtractor:
    TMDB_GENRES = [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
        "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
        "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
    ]

    def __init__(self, config_path: str = "config.yaml", semantic_search=None):
        self.config = self._load_config(config_path)
        deepseek_config = self.config.get("deepseek", {})

        deepseek_key = os.getenv("DEEPSEEK_API_KEY") or deepseek_config.get("api_key")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        if deepseek_key and deepseek_key != "${DEEPSEEK_API_KEY}" and "your_deepseek_api_key" not in deepseek_key:
            self.api_key = deepseek_key
            self.base_url = deepseek_config.get("base_url", "https://api.deepseek.com/v1")
            self.model = deepseek_config.get("model", "deepseek-chat")
            self._llm_available = True
        elif openrouter_key and openrouter_key != "${OPENROUTER_API_KEY}" and "your_openrouter_api_key" not in openrouter_key:
            self.api_key = openrouter_key
            self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
            self._llm_available = True
        else:
            self.api_key = None
            self.base_url = deepseek_config.get("base_url", "https://api.deepseek.com/v1")
            self.model = deepseek_config.get("model", "deepseek-chat")
            self._llm_available = False
            print("Neither DeepSeek nor OpenRouter configured, NL extraction uses regex fallback", file=sys.stderr)

        self.client = AsyncOpenAI(
            api_key=self.api_key or "no-key",
            base_url=self.base_url
        ) if self._llm_available else None

        self.semantic_search = semantic_search

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
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
        genres_str = ", ".join(self.TMDB_GENRES)
        return f"""You are a movie preference parser for a recommendation system. 
Extract structured features from natural language movie requests.

VALID GENRES (TMDB): {genres_str}

OUTPUT FORMAT (JSON only):
{{
  "genres": ["Genre1", "Genre2"],
  "keywords": ["keyword1", "keyword2"],
  "actors": ["Actor Name"],
  "directors": ["Director Name"],
  "decades": ["1990s", "2000s"],
  "mood": "lighthearted|dark|thrilling|emotional|mind-bending|nostalgic|feel-good|gritty",
  "min_rating": 0.0-10.0,
  "exclude_genres": ["Horror"],
  "exclude_keywords": ["gore"],
  "query_text": "concise search terms for vectorization",
  "confidence": 0.0-1.0
}}

RULES:
- Map synonyms to TMDB genres (e.g., "sci-fi" → "Science Fiction", "romcom" → ["Comedy", "Romance"])
- Extract specific entities: actor/director names, decades, franchises
- Mood: single word describing emotional tone
- min_rating: if user says "highly rated", "good reviews", use 7.0; "masterpiece" → 8.0
- Exclusions: "not horror", "no gore", "avoid sequels"
- query_text: compact terms for TF-IDF/semantic search (max 50 words)
- confidence: how well you understood the request (0.0-1.0)"""

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
        features = ExtractedFeatures(
            genres=self._validate_genres(parsed.get("genres", [])),
            keywords=parsed.get("keywords", [])[:20],
            actors=parsed.get("actors", [])[:10],
            directors=parsed.get("directors", [])[:5],
            decades=self._validate_decades(parsed.get("decades", [])),
            mood=parsed.get("mood"),
            min_rating=max(0.0, min(10.0, float(parsed.get("min_rating", 0.0)))),
            exclude_genres=self._validate_genres(parsed.get("exclude_genres", [])),
            exclude_keywords=parsed.get("exclude_keywords", [])[:10],
            query_text=parsed.get("query_text", original_text)[:500],
            confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.7)))),
            original_text=original_text
        )
        
        if not features.query_text:
            features.query_text = self._build_query_text(features)
        
        return features

    def _validate_genres(self, genres: List[str]) -> List[str]:
        valid = []
        genre_map = {g.lower(): g for g in self.TMDB_GENRES}
        genre_map.update({
            "sci-fi": "Science Fiction", "scifi": "Science Fiction",
            "romcom": "Comedy", "rom-com": "Comedy",
            "action": "Action", "thriller": "Thriller",
            "horror": "Horror", "comedy": "Comedy",
            "drama": "Drama", "romance": "Romance",
            "fantasy": "Fantasy", "animation": "Animation",
            "documentary": "Documentary", "mystery": "Mystery",
            "crime": "Crime", "adventure": "Adventure",
            "family": "Family", "war": "War", "western": "Western",
            "history": "History", "music": "Music"
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
        parts.extend(features.genres)
        parts.extend(features.keywords)
        parts.extend(features.actors)
        parts.extend(features.directors)
        parts.extend(features.decades)
        if features.mood:
            parts.append(features.mood)
        return " ".join(parts)

    def _fallback_extraction(self, text: str, context: Dict[str, Any]) -> ExtractedFeatures:
        text_lower = text.lower()
        genres = []
        keywords = []
        
        genre_keywords = {
            "action": "Action", "comedy": "Comedy", "drama": "Drama",
            "horror": "Horror", "sci-fi": "Science Fiction", "scifi": "Science Fiction",
            "thriller": "Thriller", "romance": "Romance", "fantasy": "Fantasy",
            "animation": "Animation", "documentary": "Documentary", "mystery": "Mystery",
            "crime": "Crime", "adventure": "Adventure", "romcom": "Comedy"
        }
        
        for kw, genre in genre_keywords.items():
            if kw in text_lower:
                genres.append(genre)
        
        decades = []
        import re
        decade_matches = re.findall(r'\b(19\d{1}|20[0-2]\d)\b', text)
        for d in decade_matches:
            year = int(d)
            decades.append(f"{(year // 10) * 10}s")
        
        stopwords = {"i", "want", "to", "watch", "a", "an", "the", "movie", "film", 
                     "with", "about", "like", "similar", "something", "good", "great"}
        words = [w for w in text_lower.split() if w not in stopwords and len(w) > 2]
        keywords = words[:15]
        
        mood = None
        mood_keywords = {
            "funny": "lighthearted", "dark": "dark", "scary": "thrilling",
            "emotional": "emotional", "mind-bending": "mind-bending",
            "nostalgic": "nostalgic", "feel-good": "feel-good", "gritty": "gritty"
        }
        for kw, m in mood_keywords.items():
            if kw in text_lower:
                mood = m
                break
        
        return ExtractedFeatures(
            genres=list(set(genres))[:3],
            keywords=keywords,
            decades=decades[:2],
            mood=mood,
            min_rating=6.0 if any(w in text_lower for w in ["good", "great", "best", "highly rated"]) else 0.0,
            query_text=" ".join(keywords + genres),
            confidence=0.5,
            original_text=text
        )

    def get_tmdb_genre_ids(self) -> Dict[str, int]:
        genre_map = {
            "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35, "Crime": 80,
            "Documentary": 99, "Drama": 18, "Family": 10751, "Fantasy": 14, "History": 36,
            "Horror": 27, "Music": 10402, "Mystery": 9648, "Romance": 10749,
            "Science Fiction": 878, "TV Movie": 10770, "Thriller": 53, "War": 10752, "Western": 37
        }
        return genre_map

    def build_tmdb_discover_params(
        self,
        features: "ExtractedFeatures",
        subscriptions: List[str],
        region: str = "US",
        language: str = "en"
    ) -> Dict[str, Any]:
        params = {
            "genre": features.genres[0] if features.genres else None,
            "region": region,
            "language": language,
            "min_rating": features.min_rating or 0.0,
            "sort_by": "popularity.desc",
            "page": 1,
            "with_watch_providers": subscriptions or None,
            "watch_region": region,
        }

        text = (features.original_text or "").lower()
        import re

        # Parse year / decade / date constraints
        year_match = re.search(r'\b(19\d\d|20\d\d)\b', text)
        if "after 2024" in text or "since 2024" in text or "2024 or later" in text or "from 2024" in text or "post 2024" in text or "2024+" in text:
            params["primary_release_date_gte"] = "2024-01-01"
        elif "after 2023" in text or "since 2023" in text or "2023+" in text:
            params["primary_release_date_gte"] = "2023-01-01"
        elif year_match:
            year = int(year_match.group(1))
            if "after" in text or "since" in text or "post" in text:
                params["primary_release_date_gte"] = f"{year}-01-01"
            elif "before" in text or "prior" in text:
                params["primary_release_date_lte"] = f"{year}-12-31"
            elif "in " + str(year) in text or str(year) in text:
                params["primary_release_year"] = year
        elif features.decades:
            dec = features.decades[0]
            if dec.endswith('s') and dec[:-1].isdigit():
                start_yr = int(dec[:-1])
                params["primary_release_date_gte"] = f"{start_yr}-01-01"
                params["primary_release_date_lte"] = f"{start_yr + 9}-12-31"

        # Parse runtime constraints (e.g., 2 hours, 2hrs, 90 mins, 1.5 hours)
        if re.search(r'\b(2\s*(?:hours|hrs|hr|h)|120\s*min(?:ute)?s?)\b', text):
            params["with_runtime_gte"] = 95
            params["with_runtime_lte"] = 145
        elif re.search(r'\b(1\.5\s*(?:hours|hrs|hr)|90\s*min(?:ute)?s?)\b', text):
            params["with_runtime_gte"] = 75
            params["with_runtime_lte"] = 110
        elif re.search(r'\b(3\s*(?:hours|hrs|hr)|180\s*min(?:ute)?s?)\b', text):
            params["with_runtime_gte"] = 150

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
        return features


async def main():
    try:
        extractor = NLFeatureExtractor()
        
        test_queries = [
            "I want a funny 90s space comedy with Tom Hanks",
            "Dark gritty thriller like Sicario but not too violent",
            "Feel-good romantic comedy from the 2000s",
            "Mind-bending sci-fi with time travel, highly rated",
            "Action movie with Tom Cruise, not Mission Impossible"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}", file=sys.stderr)
            features = await extractor.extract(query)
            print(f"  Genres: {features.genres}", file=sys.stderr)
            print(f"  Keywords: {features.keywords[:10]}", file=sys.stderr)
            print(f"  Actors: {features.actors}", file=sys.stderr)
            print(f"  Decades: {features.decades}", file=sys.stderr)
            print(f"  Mood: {features.mood}", file=sys.stderr)
            print(f"  Min rating: {features.min_rating}", file=sys.stderr)
            print(f"  Query text: {features.query_text}", file=sys.stderr)
            print(f"  Confidence: {features.confidence}", file=sys.stderr)
            
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
