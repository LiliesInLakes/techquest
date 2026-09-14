import os
import re
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RecommendationReasoning:
    movie_title: str
    reasoning: str
    key_factors: List[str]
    user_preferences_matched: List[str]


class DeepSeekReasoner:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        deepseek_config = self.config.get("deepseek", {})
        
        deepseek_key = os.getenv("DEEPSEEK_API_KEY") or deepseek_config.get("api_key")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        if deepseek_key and deepseek_key != "${DEEPSEEK_API_KEY}" and "your_deepseek_api_key" not in deepseek_key:
            self.api_key = deepseek_key
            self.base_url = deepseek_config.get("base_url", "https://api.deepseek.com/v1")
            self.model = deepseek_config.get("model", "deepseek-chat")
        elif openrouter_key and openrouter_key != "${OPENROUTER_API_KEY}" and "your_openrouter_api_key" not in openrouter_key:
            self.api_key = openrouter_key
            self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
        else:
            raise ValueError("Neither DEEPSEEK_API_KEY nor OPENROUTER_API_KEY is configured")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    async def generate_recommendation_reasoning(
        self,
        movie: Dict[str, Any],
        user_profile,
        ranking_factors: Dict[str, float],
        user_query: str = "",
        watch_history_summary: str = ""
    ) -> RecommendationReasoning:
        prompt = self._build_reasoning_prompt(
            movie, user_profile, ranking_factors, user_query, watch_history_summary
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            return self._parse_reasoning_response(content, movie.get('title', ''))
            
        except Exception as e:
            return self._fallback_reasoning(movie, user_profile, ranking_factors, str(e))

    def _get_system_prompt(self) -> str:
        return """You are a movie recommendation expert for hospitality and airline businesses. 
Your job is to explain WHY a specific movie was recommended to a user in a natural, conversational way.

Guidelines:
- Write in first person as if you're the recommender talking to the user
- Reference specific user preferences (genres, keywords, ratings, watch history)
- Mention the movie's qualities that match their taste
- Keep it concise (2-4 sentences)
- Be specific, not generic
- Don't mention technical details like "TF-IDF" or "cosine similarity"
- Focus on WHAT the user likes and HOW this movie delivers that"""

    def _build_reasoning_prompt(
        self,
        movie: Dict[str, Any],
        user_profile,
        ranking_factors: Dict[str, float],
        user_query: str,
        watch_history_summary: str
    ) -> str:
        movie_title = movie.get('title', 'Unknown')
        movie_genres = movie.get('genres', [])
        movie_overview = movie.get('overview', '')
        movie_rating = movie.get('vote_average', 0)
        
        top_genres = [g for g, _ in user_profile.top_genres[:5]] if user_profile.top_genres else []
        top_keywords = [k for k, _ in user_profile.top_keywords[:10]] if user_profile.top_keywords else []
        top_directors = [d for d, _ in user_profile.top_directors[:3]] if user_profile.top_directors else []
        top_cast = [c for c, _ in user_profile.top_cast[:3]] if user_profile.top_cast else []
        
        matched_genres = [g for g in movie_genres if g in top_genres]
        matched_keywords = [k for k in top_keywords if k.lower() in movie_overview.lower()]
        
        factors = []
        if ranking_factors.get('imdb', 0) > 0.7:
            factors.append(f"high IMDb rating ({movie_rating}/10)")
        if ranking_factors.get('watch_history', 0) > 0.5:
            factors.append("strong match with your watch history")
        if ranking_factors.get('keyword_overlap', 0) > 0.3:
            factors.append("keywords you've enjoyed before")
        if ranking_factors.get('semantic', 0) > 0.6:
            factors.append("semantically similar to movies you like")
        
        prompt = f"""Movie: {movie_title}
Genres: {', '.join(movie_genres)}
IMDb Rating: {movie_rating}/10
Overview: {movie_overview[:300]}

User Preferences:
- Favorite Genres: {', '.join(top_genres) if top_genres else 'Not available'}
- Favorite Keywords: {', '.join(top_keywords) if top_keywords else 'Not available'}
- Favorite Directors: {', '.join(top_directors) if top_directors else 'Not available'}
- Favorite Actors: {', '.join(top_cast) if top_cast else 'Not available'}
- Average Rating Given: {user_profile.avg_rating:.1f}/10
- Total Movies Watched: {user_profile.watch_count}

User Query: {user_query if user_query else 'General recommendation'}

Watch History Context: {watch_history_summary if watch_history_summary else 'Based on your viewing patterns'}

Key Matches:
- Matching Genres: {', '.join(matched_genres) if matched_genres else 'None'}
- Matching Keywords in Overview: {', '.join(matched_keywords) if matched_keywords else 'None'}
- Ranking Factors: {', '.join(factors) if factors else 'General compatibility'}

Write a natural, conversational recommendation reason (2-4 sentences) explaining why this movie was chosen for this user. Start with 'Because you...' or 'Since you...'"""
        
        return prompt

    def _parse_reasoning_response(self, content: str, movie_title: str) -> RecommendationReasoning:
        lines = content.strip().split('\n')
        reasoning = ' '.join(lines).strip()
        
        key_factors = []
        user_prefs = []
        
        content_lower = content.lower()
        if 'genre' in content_lower:
            key_factors.append('genre preference')
        if 'director' in content_lower:
            key_factors.append('director preference')
        if 'actor' in content_lower or 'cast' in content_lower:
            key_factors.append('actor preference')
        if 'rating' in content_lower or 'imdb' in content_lower:
            key_factors.append('high rating')
        if 'keyword' in content_lower or 'theme' in content_lower:
            key_factors.append('thematic match')
        if 'history' in content_lower or 'watched' in content_lower:
            key_factors.append('watch history')
        
        return RecommendationReasoning(
            movie_title=movie_title,
            reasoning=reasoning,
            key_factors=key_factors,
            user_preferences_matched=user_prefs
        )

    def _fallback_reasoning(
        self,
        movie: Dict[str, Any],
        user_profile,
        ranking_factors: Dict[str, float],
        error: str
    ) -> RecommendationReasoning:
        movie_title = movie.get('title', 'Unknown')
        genres = movie.get('genres', [])
        rating = movie.get('vote_average', 0)
        
        top_genres = [g for g, _ in user_profile.top_genres[:3]] if user_profile.top_genres else []
        matched = [g for g in genres if g in top_genres]
        
        if matched:
            reason = f"Because you enjoy {', '.join(matched)} movies, I recommend {movie_title} — it's a {rating}/10 rated {', '.join(genres)} film that matches your taste."
        else:
            reason = f"Since you've rated movies {user_profile.avg_rating:.1f}/10 on average, {movie_title} ({rating}/10) fits your preference for well-rated {', '.join(genres)} films."
        
        return RecommendationReasoning(
            movie_title=movie_title,
            reasoning=reason,
            key_factors=['genre preference', 'rating'] if matched else ['rating'],
            user_preferences_matched=matched
        )

    async def generate_top_pick_verdict(
        self,
        top_movie: Dict[str, Any],
        user_profile,
        ranking_factors: Dict[str, float],
        user_query: str = "",
        watch_history_summary: str = ""
    ) -> str:
        title = top_movie.get('title', 'Unknown')
        genres = ', '.join([g if isinstance(g, str) else g.get('name', '') for g in top_movie.get('genres', [])])
        prompt = f"""You are a movie recommendation expert. Write a decisive, compelling 2-3 sentence verdict explaining specifically why "{title}" is the definitive film for the user to watch tonight.

Target Film: {title}
Genres: {genres}
Rating: {top_movie.get('vote_average', 0)}/10
Overview: {top_movie.get('overview', '')[:300]}
User Search Query: {user_query if user_query else 'Personalized taste match'}
Watch History Summary: {watch_history_summary}

STRICT CONSTRAINTS:
1. You must ONLY discuss and evaluate "{title}". Do NOT substitute, mention, or recommend any other movie.
2. Do NOT output metadata headers like "**#1 Top Pick:**", "Genres:", "Rating:", or "Runtime:".
3. Write directly in engaging, conversational prose explaining why "{title}" fulfills their taste profile and search intent tonight."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a movie curation expert. You write decisive verdicts solely for the specified film ({title}) without mentioning any other movie or outputting boilerplate template headers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=250
            )
            verdict = response.choices[0].message.content.strip()
            # Clean up any lingering markdown boilerplate headers
            verdict = re.sub(r'^\*{0,2}#?1?\s*Top Pick:?.*?\*{0,2}\s*', '', verdict, flags=re.IGNORECASE)
            verdict = re.sub(r'^\*{0,2}Final Verdict:?\*{0,2}\s*', '', verdict, flags=re.IGNORECASE)
            verdict = re.sub(r'^\*{0,2}Why this is your ultimate watch:?\*{0,2}\s*', '', verdict, flags=re.IGNORECASE)
            return verdict.strip()
        except Exception:
            return self._fallback_top_pick(top_movie, user_profile)

    def _fallback_top_pick(self, top_movie: Dict[str, Any], user_profile) -> str:
        title = top_movie.get('title', 'This movie')
        rating = top_movie.get('vote_average', 0)
        genres = ', '.join(top_movie.get('genres', []))
        return f"If you're picking just one movie tonight, make it {title} ({rating}/10) — it stands out as your definitive top match across {genres} and your viewing preferences."

    async def generate_batch_reasoning(
        self,
        movies: List[Dict[str, Any]],
        user_profile,
        ranking_factors_list: List[Dict[str, float]],
        user_query: str = "",
        watch_history_summary: str = ""
    ) -> List[RecommendationReasoning]:
        tasks = [
            self.generate_recommendation_reasoning(movie, user_profile, factors, user_query, watch_history_summary)
            for movie, factors in zip(movies, ranking_factors_list)
        ]
        return await asyncio.gather(*tasks)


class ReasoningTemplateEngine:
    def __init__(self):
        self.templates = {
            "genre_match": "Because you love {genre} movies, {title} is a great pick — it's a {rating}/10 {genres} film that delivers exactly what you enjoy.",
            "director_match": "Since you've enjoyed films by {director}, you'll like {title} — it carries their signature style with a {rating}/10 rating.",
            "actor_match": "Given your preference for {actor}'s performances, {title} showcases them in a {rating}/10 {genres} role.",
            "keyword_match": "Because you've liked movies with themes like {keywords}, {title} ({rating}/10) explores similar territory.",
            "high_rating": "With your average rating of {avg_rating}/10, {title} ({rating}/10) meets your quality bar for {genres} films.",
            "watch_history": "Based on your watch history, {title} ({rating}/10) aligns with the {genres} patterns you've consistently enjoyed.",
            "semantic": "Since you enjoyed similar movies, {title} ({rating}/10) captures that same {genres} feel you're looking for."
        }

    def generate(self, movie: Dict, profile, factors: Dict) -> str:
        title = movie.get('title', 'this movie')
        genres = movie.get('genres', [])
        rating = movie.get('vote_average', 0)
        
        top_genres = [g for g, _ in profile.top_genres[:3]] if profile.top_genres else []
        matched_genres = [g for g in genres if g in top_genres]
        
        if matched_genres:
            return self.templates["genre_match"].format(
                genre=matched_genres[0], title=title, rating=rating, genres=', '.join(genres)
            )
        
        if factors.get('watch_history', 0) > 0.5:
            return self.templates["watch_history"].format(
                title=title, rating=rating, genres=', '.join(genres)
            )
        
        if factors.get('imdb', 0) > 0.7:
            return self.templates["high_rating"].format(
                title=title, rating=rating, genres=', '.join(genres), avg_rating=profile.avg_rating
            )
        
        return f"{title} ({rating}/10) is a {', '.join(genres)} film that matches your viewing preferences."

    def generate_top_pick(self, top_movie: Dict, profile, factors: Dict = None, user_query: str = "") -> str:
        title = top_movie.get('title', 'This movie')
        rating = top_movie.get('vote_average', 0)
        genres = ', '.join(top_movie.get('genres', []))
        return f"If you're picking just one movie tonight, make it {title} ({rating}/10) — it is your definitive #1 recommendation based on your taste profile in {genres}."


async def main():
    try:
        reasoner = DeepSeekReasoner()
        print("DeepSeekReasoner initialized successfully")
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Set DEEPSEEK_API_KEY environment variable")


if __name__ == "__main__":
    asyncio.run(main())