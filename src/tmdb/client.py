import os
import sys
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
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
        url = f"{self.base_url}{endpoint}"
        request_params = {"api_key": self.api_key, "language": "en-US"}
        if params:
            request_params.update(params)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        
        if not self.session or self.session.closed:
            connector = aiohttp.TCPConnector(ssl=False)
            self.session = aiohttp.ClientSession(connector=connector, headers=headers)
        
        try:
            async with self.session.get(url, params=request_params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"TMDB API error {response.status}: {error_text}")
                return await response.json()
        except Exception as e:
            import requests
            from urllib3.util import Retry
            from requests.adapters import HTTPAdapter
            def _sync_req():
                s = requests.Session()
                retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
                s.mount("https://", HTTPAdapter(max_retries=retries))
                resp = s.get(url, params=request_params, headers=headers, timeout=12)
                if resp.status_code != 200:
                    raise Exception(f"TMDB API error {resp.status_code}: {resp.text}")
                return resp.json()
            return await asyncio.to_thread(_sync_req)

    async def search_person(self, name: str) -> Optional[int]:
        """Search for an actor or director by name and return their TMDB Person ID."""
        if not hasattr(self, "_person_cache"):
            self._person_cache: Dict[str, Optional[int]] = {}
        
        name_clean = name.strip()
        if name_clean in self._person_cache:
            return self._person_cache[name_clean]
            
        try:
            data = await self._request("/search/person", {"query": name_clean})
            results = data.get("results", [])
            if results:
                # Prefer strong/exact match if present, else first result
                exact_matches = [r for r in results if r.get("name", "").lower() == name_clean.lower()]
                best = exact_matches[0] if exact_matches else results[0]
                pid = best.get("id")
                print(f"[TMDBClient] Resolved person '{name_clean}' -> Person ID {pid}", file=sys.stderr)
                self._person_cache[name_clean] = pid
                return pid
            else:
                print(f"[TMDBClient] Could not resolve person '{name_clean}' (0 results found)", file=sys.stderr)
                self._person_cache[name_clean] = None
        except Exception as e:
            print(f"[TMDBClient] Error resolving person '{name_clean}': {e}", file=sys.stderr)
            self._person_cache[name_clean] = None
            
        return None

    async def search_keyword(self, keyword: str) -> Optional[int]:
        """Search for a keyword and return its TMDB Keyword ID."""
        try:
            data = await self._request("/search/keyword", {"query": keyword})
            results = data.get("results", [])
            if results:
                return results[0].get("id")
        except Exception:
            pass
        return None

    async def discover_movies(
        self,
        genre: Optional[str] = None,
        region: Optional[str] = None,
        language: Optional[str] = None,
        min_rating: float = 0.0,
        min_votes: int = 0,
        sort_by: str = "popularity.desc",
        page: int = 1,
        with_watch_providers: Optional[List[str]] = None,
        watch_region: Optional[str] = None,
        with_genres: Optional[str] = None,
        without_genres: Optional[str] = None,
        with_cast: Optional[Any] = None,
        with_crew: Optional[Any] = None,
        primary_release_year: Optional[int] = None,
        primary_release_date_gte: Optional[str] = None,
        primary_release_date_lte: Optional[str] = None,
        with_runtime_gte: Optional[int] = None,
        with_runtime_lte: Optional[int] = None,
        with_keywords: Optional[str] = None,
        include_adult: Optional[bool] = None,
        **kwargs
    ) -> List[Movie]:
        params: Dict[str, Any] = {
            "sort_by": sort_by,
            "page": page,
        }
        
        if min_rating and min_rating > 0.0:
            params["vote_average.gte"] = min_rating
            
        if min_votes and min_votes > 0:
            params["vote_count.gte"] = min_votes
            
        if language:
            params["with_original_language"] = language
            
        if region:
            params["region"] = region
            params["watch_region"] = watch_region or region
        elif watch_region:
            params["watch_region"] = watch_region

        if include_adult is not None:
            params["include_adult"] = include_adult
        
        if with_genres:
            params["with_genres"] = with_genres
        elif genre:
            genre_ids = await self._get_genre_ids()
            genre_id = genre_ids.get(genre.lower()) if genre_ids else None
            if genre_id:
                params["with_genres"] = genre_id

        if without_genres:
            params["without_genres"] = without_genres
        
        if with_cast:
            resolved_cast = []
            items = with_cast if isinstance(with_cast, list) else [c.strip() for c in str(with_cast).split(",") if c.strip()]
            for item in items:
                if str(item).isdigit():
                    resolved_cast.append(str(item))
                else:
                    pid = await self.search_person(str(item))
                    if pid is not None:
                        resolved_cast.append(str(pid))
            if resolved_cast:
                params["with_cast"] = ",".join(resolved_cast)

        if with_crew:
            resolved_crew = []
            items = with_crew if isinstance(with_crew, list) else [c.strip() for c in str(with_crew).split(",") if c.strip()]
            for item in items:
                if str(item).isdigit():
                    resolved_crew.append(str(item))
                else:
                    pid = await self.search_person(str(item))
                    if pid is not None:
                        resolved_crew.append(str(pid))
            if resolved_crew:
                params["with_crew"] = ",".join(resolved_crew)

        if primary_release_year:
            params["primary_release_year"] = primary_release_year

        if primary_release_date_gte:
            params["primary_release_date.gte"] = primary_release_date_gte

        if primary_release_date_lte:
            params["primary_release_date.lte"] = primary_release_date_lte

        if with_runtime_gte is not None:
            params["with_runtime.gte"] = with_runtime_gte

        if with_runtime_lte is not None:
            params["with_runtime.lte"] = with_runtime_lte

        if with_keywords:
            params["with_keywords"] = with_keywords
            
        for k, v in kwargs.items():
            if v is not None:
                params[k] = v
        
        if with_watch_providers:
            numeric_pids = [p for p in with_watch_providers if str(p).isdigit()]
            named_p = [p for p in with_watch_providers if not str(p).isdigit()]
            resolved_pids = [int(p) for p in numeric_pids]
            w_reg = watch_region or region or "US"
            if named_p:
                resolved_pids.extend(await self._resolve_watch_provider_ids(named_p, w_reg))
            if resolved_pids:
                params["with_watch_providers"] = "|".join(str(i) for i in resolved_pids)
                params["watch_region"] = w_reg
        
        data = await self._request("/discover/movie", params)
        return [self._parse_movie(m) for m in data.get("results", [])]

    async def _resolve_watch_provider_ids(
        self, providers: List[str], watch_region: str = "US"
    ) -> List[int]:
        # TMDB with_watch_providers only accepts numeric provider IDs, but callers
        # pass names like "netflix". Resolve via /watch/providers/movie.
        if not providers:
            return []

        numeric = [p for p in providers if str(p).isdigit()]
        names = [p for p in providers if not str(p).isdigit()]

        resolved = set(numeric)
        if names:
            try:
                data = await self._request(
                    "/watch/providers/movie", {"watch_region": watch_region}
                )
                name_to_id = {
                    p.get("provider_name", "").lower(): p.get("provider_id")
                    for p in data.get("results", [])
                }
                for name in names:
                    pid = name_to_id.get(str(name).lower())
                    if pid:
                        resolved.add(pid)
            except Exception as e:
                print(f"Watch provider resolution failed ({e}); ignoring provider filter", file=sys.stderr)

        return sorted(resolved)

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

    async def _resolve_watch_provider_ids(self, providers: List[str], watch_region: str = "US") -> List[int]:
        try:
            data = await self._request("/watch/providers/movie", {"watch_region": watch_region})
            provider_list = data.get("results", [])
            mapping = {p["provider_name"].lower(): p["provider_id"] for p in provider_list}
            ids = []
            for name in providers:
                n = name.strip().lower()
                if n in mapping:
                    ids.append(mapping[n])
                else:
                    for pname, pid in mapping.items():
                        if n in pname or pname in n:
                            ids.append(pid)
                            break
            return ids
        except Exception as e:
            print(f"Warning: could not resolve provider IDs: {e}", file=sys.stderr)
            return []

    def _parse_movie(self, data: Dict) -> Movie:
        genre_names = []
        if "genre_ids" in data:
            # _parse_movie is sync but called from async context; never use
            # asyncio.run() here (crashes inside a running loop). If the genre
            # cache isn't populated yet, leave names empty — discover_movies()
            # populates it via _get_genre_ids() before parsing.
            cache = getattr(self, "_genre_cache", None) or {}
            genre_map = {v: k for k, v in cache.items()}
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