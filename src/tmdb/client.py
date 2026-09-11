import os
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import yaml


@dataclass
class Movie:
    id: int
    title: str
    overview: str
    genres: List[str]
    release_date: str
    vote_average: float
    vote_count: int
    popularity: float
    original_language: str
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    adult: bool = False
    video: bool = False
    keywords: List[str] = None
    credits: Dict[str, Any] = None
    watch_providers: Dict[str, Any] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.credits is None:
            self.credits = {}
        if self.watch_providers is None:
            self.watch_providers = {}


class TMDBClient:
    def __init__(self, api_key: Optional[str] = None, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.api_key = api_key or os.getenv("TMDB_API_KEY") or self.config.get("tmdb", {}).get("api_key")
        if not self.api_key or self.api_key == "${TMDB_API_KEY}":
            raise ValueError("TMDB_API_KEY not configured")
        
        self.base_url = self.config.get("tmdb", {}).get("base_url", "https://api.themoviedb.org/3")
        self.image_base_url = self.config.get("tmdb", {}).get("image_base_url", "https://image.tmdb.org/t/p/w500")
        self.session: Optional[aiohttp.ClientSession] = None

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _request(self, endpoint: str, params: Dict = None) -> Dict:
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}{endpoint}"
        request_params = {"api_key": self.api_key, "language": "en-US"}
        if params:
            request_params.update(params)
        
        async with self.session.get(url, params=request_params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"TMDB API error {response.status}: {error_text}")
            return await response.json()

    async def discover_movies(
        self,
        genre: Optional[str] = None,
        region: str = "US",
        language: str = "en",
        min_rating: float = 0.0,
        min_votes: int = 100,
        sort_by: str = "popularity.desc",
        page: int = 1,
        with_watch_providers: Optional[List[str]] = None,
        watch_region: str = "US"
    ) -> List[Movie]:
        genre_ids = await self._get_genre_ids() if genre else None
        genre_id = genre_ids.get(genre.lower()) if genre_ids and genre else None
        
        params = {
            "sort_by": sort_by,
            "page": page,
            "vote_average.gte": min_rating,
            "vote_count.gte": min_votes,
            "with_original_language": language,
            "region": region,
            "watch_region": watch_region
        }
        
        if genre_id:
            params["with_genres"] = genre_id
        
        if with_watch_providers:
            params["with_watch_providers"] = "|".join(with_watch_providers)
            params["watch_region"] = watch_region
        
        data = await self._request("/discover/movie", params)
        return [self._parse_movie(m) for m in data.get("results", [])]

    async def search_movies(
        self,
        query: str,
        region: str = "US",
        language: str = "en",
        page: int = 1
    ) -> List[Movie]:
        params = {
            "query": query,
            "region": region,
            "language": language,
            "page": page
        }
        data = await self._request("/search/movie", params)
        return [self._parse_movie(m) for m in data.get("results", [])]

    async def get_movie_details(
        self,
        movie_id: int,
        include_credits: bool = True,
        include_keywords: bool = True,
        region: str = "US"
    ) -> Movie:
        append_to_response = []
        if include_credits:
            append_to_response.append("credits")
        if include_keywords:
            append_to_response.append("keywords")
        append_to_response.append("watch/providers")
        
        params = {
            "append_to_response": ",".join(append_to_response),
            "region": region
        }
        
        data = await self._request(f"/movie/{movie_id}", params)
        movie = self._parse_movie(data)
        
        if include_credits and "credits" in data:
            movie.credits = {
                "cast": [c["name"] for c in data["credits"].get("cast", [])[:10]],
                "crew": [c["name"] for c in data["credits"].get("crew", []) if c["job"] == "Director"][:5]
            }
        
        if include_keywords and "keywords" in data:
            movie.keywords = [k["name"] for k in data["keywords"].get("keywords", [])]
        
        if "watch/providers" in data:
            movie.watch_providers = data["watch/providers"].get("results", {}).get(region, {})
        
        return movie

    async def get_movie_keywords(self, movie_id: int) -> List[str]:
        data = await self._request(f"/movie/{movie_id}/keywords")
        return [k["name"] for k in data.get("keywords", [])]

    async def get_movie_credits(self, movie_id: int) -> Dict:
        data = await self._request(f"/movie/{movie_id}/credits")
        return {
            "cast": [c["name"] for c in data.get("cast", [])[:15]],
            "crew": [c["name"] for c in data.get("crew", []) if c["job"] in ["Director", "Writer", "Producer"]][:10]
        }

    async def get_watch_providers(self, movie_id: int, region: str = "US") -> Dict:
        data = await self._request(f"/movie/{movie_id}/watch/providers")
        return data.get("results", {}).get(region, {})

    async def get_genres(self) -> Dict[str, int]:
        data = await self._request("/genre/movie/list")
        return {g["name"].lower(): g["id"] for g in data.get("genres", [])}

    async def _get_genre_ids(self) -> Dict[str, int]:
        if not hasattr(self, "_genre_cache"):
            self._genre_cache = await self.get_genres()
        return self._genre_cache

    def _parse_movie(self, data: Dict) -> Movie:
        genre_names = []
        if "genre_ids" in data:
            genre_map = {v: k for k, v in (asyncio.run(self.get_genres()) if not hasattr(self, "_genre_cache") else self._genre_cache).items()}
            genre_names = [genre_map.get(gid, "") for gid in data.get("genre_ids", [])]
        elif "genres" in data:
            genre_names = [g["name"] for g in data.get("genres", [])]
        
        return Movie(
            id=data["id"],
            title=data.get("title") or data.get("original_title", ""),
            overview=data.get("overview", ""),
            genres=genre_names,
            release_date=data.get("release_date", ""),
            vote_average=data.get("vote_average", 0.0),
            vote_count=data.get("vote_count", 0),
            popularity=data.get("popularity", 0.0),
            original_language=data.get("original_language", "en"),
            poster_path=data.get("poster_path"),
            backdrop_path=data.get("backdrop_path"),
            adult=data.get("adult", False),
            video=data.get("video", False)
        )

    async def get_top_rated_movies(
        self,
        region: str = "US",
        language: str = "en",
        page: int = 1,
        min_votes: int = 500
    ) -> List[Movie]:
        params = {
            "region": region,
            "language": language,
            "page": page,
            "vote_count.gte": min_votes,
            "sort_by": "vote_average.desc"
        }
        data = await self._request("/discover/movie", params)
        return [self._parse_movie(m) for m in data.get("results", [])]

    async def get_popular_movies(
        self,
        region: str = "US",
        language: str = "en",
        page: int = 1
    ) -> List[Movie]:
        params = {"region": region, "language": language, "page": page}
        data = await self._request("/movie/popular", params)
        return [self._parse_movie(m) for m in data.get("results", [])]


async def main():
    async with TMDBClient() as client:
        movies = await client.discover_movies(genre="Action", region="US", min_rating=7.0, max_results=10)
        for m in movies[:5]:
            print(f"{m.title} ({m.release_date[:4]}) - {m.vote_average}/10 - {m.genres}")


if __name__ == "__main__":
    asyncio.run(main())