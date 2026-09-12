import json
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from mcp.types import Tool, TextContent, CallToolResult


class MCPToolType(str, Enum):
    SEARCH_MOVIES = "search_movies"
    GET_RECOMMENDATIONS = "get_recommendations"
    GET_MOVIE_DETAILS = "get_movie_details"
    ANALYZE_USER_PREFERENCES = "analyze_user_preferences"


class SearchMoviesParams(BaseModel):
    genre: Optional[str] = None
    region: Optional[str] = None
    language: Optional[str] = None
    subscription_services: List[str] = Field(default_factory=list)
    min_rating: float = 0.0
    max_results: int = 100


class GetRecommendationsParams(BaseModel):
    user_id: str
    watch_history: List[Dict[str, Any]] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    subscriptions: List[str] = Field(default_factory=list)
    region: str = "US"
    language: str = "en"
    top_k: int = 10
    natural_language: Optional[str] = None


class MovieDetailsParams(BaseModel):
    movie_id: int
    include_credits: bool = True
    include_keywords: bool = True


class AnalyzePreferencesParams(BaseModel):
    watch_history: List[Dict[str, Any]]
    ratings: Dict[int, float] = Field(default_factory=dict)


@dataclass
class MCPServer:
    name: str = "movie-recommender"
    version: str = "1.0.0"
    description: str = "Movie recommendation agent"

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name=MCPToolType.SEARCH_MOVIES.value,
                description="Search movies by genre, region, language, and subscription availability",
                inputSchema=SearchMoviesParams.model_json_schema()
            ),
            Tool(
                name=MCPToolType.GET_RECOMMENDATIONS.value,
                description="Get personalized movie recommendations from watch history and an optional natural-language request",
                inputSchema=GetRecommendationsParams.model_json_schema()
            ),
            Tool(
                name=MCPToolType.GET_MOVIE_DETAILS.value,
                description="Get detailed movie information including credits and keywords",
                inputSchema=MovieDetailsParams.model_json_schema()
            ),
            Tool(
                name=MCPToolType.ANALYZE_USER_PREFERENCES.value,
                description="Analyze user watch history to extract preference profile",
                inputSchema=AnalyzePreferencesParams.model_json_schema()
            )
        ]


app = FastAPI(title="Movie Recommender MCP Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

mcp_server = MCPServer()


@app.get("/mcp/tools")
async def list_tools():
    return {"tools": [asdict(tool) for tool in mcp_server.get_tools()]}


@app.post("/mcp/tools/{tool_name}")
async def call_tool(tool_name: str, request: Request):
    body = await request.json()

    try:
        if tool_name == MCPToolType.SEARCH_MOVIES.value:
            return await search_movies(SearchMoviesParams(**body))
        elif tool_name == MCPToolType.GET_RECOMMENDATIONS.value:
            return await get_recommendations(GetRecommendationsParams(**body))
        elif tool_name == MCPToolType.GET_MOVIE_DETAILS.value:
            return await get_movie_details(MovieDetailsParams(**body))
        elif tool_name == MCPToolType.ANALYZE_USER_PREFERENCES.value:
            return await analyze_preferences(AnalyzePreferencesParams(**body))
        else:
            raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def search_movies(params: SearchMoviesParams) -> CallToolResult:
    from src.pipeline.recommender import MovieRecommender

    recommender = MovieRecommender()
    movies = await recommender.search_movies(
        genre=params.genre,
        region=params.region,
        language=params.language,
        subscriptions=params.subscription_services,
        min_rating=params.min_rating,
        max_results=params.max_results
    )
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(movies, indent=2))])


async def get_recommendations(params: GetRecommendationsParams) -> CallToolResult:
    from src.pipeline.recommender import MovieRecommender

    recommender = MovieRecommender()
    recommendations = await recommender.get_recommendations(
        user_id=params.user_id,
        watch_history=params.watch_history,
        preferences=params.preferences,
        subscriptions=params.subscriptions,
        region=params.region,
        language=params.language,
        top_k=params.top_k,
        natural_language=params.natural_language
    )
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(recommendations, indent=2))])


async def get_movie_details(params: MovieDetailsParams) -> CallToolResult:
    from src.tmdb.client import TMDBClient

    client = TMDBClient()
    details = await client.get_movie_details(
        params.movie_id,
        include_credits=params.include_credits,
        include_keywords=params.include_keywords
    )
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(details, indent=2))])


async def analyze_preferences(params: AnalyzePreferencesParams) -> CallToolResult:
    from src.pipeline.recommender import MovieRecommender

    recommender = MovieRecommender()
    profile = recommender.analyze_user_preferences(
        watch_history=params.watch_history,
        ratings=params.ratings
    )
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(profile, indent=2))])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
