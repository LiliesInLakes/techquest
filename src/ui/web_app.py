import os
import sys
import json
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

from src.pipeline.recommender import MovieRecommender
from src.memory.service import UserMemoryService
from src.memory.models import EventType

app = FastAPI(title="Movie Recommender Agent & Memory UI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender: Optional[MovieRecommender] = None
memory_service: Optional[UserMemoryService] = None

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


async def get_recommender() -> MovieRecommender:
    global recommender
    if recommender is None:
        recommender = MovieRecommender()
        await recommender.initialize()
    return recommender


def get_memory_service() -> UserMemoryService:
    global memory_service
    if memory_service is None:
        memory_service = UserMemoryService()
    return memory_service


class RecommendRequest(BaseModel):
    user_id: str = "demo"
    natural_language: Optional[str] = None
    genre: Optional[str] = None
    region: str = "US"
    language: str = "en"
    subscriptions: List[str] = []
    top_k: int = 1
    excluded_movie_ids: List[int] = []


class InteractionRequest(BaseModel):
    user_id: str = "demo"
    movie_id: int
    event_type: str  # movie_clicked, movie_viewed, added_to_watchlist, marked_watched
    metadata: Optional[Dict[str, Any]] = None


class RateRequest(BaseModel):
    user_id: str = "demo"
    movie_id: int
    rating: float  # 1.0 to 10.0
    review_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SubscriptionsRequest(BaseModel):
    user_id: str = "demo"
    subscriptions: List[str]


class RejectRequest(BaseModel):
    user_id: str = "demo"
    movie_id: int
    reason: Optional[str] = None
    current_movie_ids: List[int] = []
    natural_language: Optional[str] = None
    subscriptions: List[str] = []
    metadata: Optional[Dict[str, Any]] = None


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/subscriptions")
async def update_subscriptions(req: SubscriptionsRequest):
    try:
        mem = get_memory_service()
        prof = mem.update_user_subscriptions(req.user_id, req.subscriptions)
        return {
            "status": "success",
            "subscriptions": prof.subscriptions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend")
async def recommend_movies(req: RecommendRequest):
    try:
        rec = await get_recommender()
        mem = get_memory_service()
        prof = mem.get_user_profile(req.user_id)
        
        # Pull watch history for user
        events = mem.storage.get_events(req.user_id)
        ratings = mem.storage.get_ratings(req.user_id)
        
        watch_history = []
        for r in ratings:
            watch_history.append({"movie_id": r.movie_id, "rating": r.rating})
        
        subs = req.subscriptions if req.subscriptions else getattr(prof, "subscriptions", ["Netflix", "Prime Video", "Disney+"])
        
        result = await rec.get_recommendations(
            user_id=req.user_id,
            watch_history=watch_history,
            subscriptions=subs,
            region=req.region,
            language=req.language,
            top_k=req.top_k,
            natural_language=req.natural_language,
            excluded_movie_ids=req.excluded_movie_ids
        )
        
        # Record search exploration keywords/tropes in user memory
        if req.natural_language:
            try:
                mem.record_search_query(
                    user_id=req.user_id,
                    query_text=req.natural_language,
                    extracted_features=result.get("extracted_features")
                )
            except Exception as search_err:
                print(f"Error recording search memory: {search_err}", file=sys.stderr)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reject")
async def reject_movie(req: RejectRequest):
    try:
        mem = get_memory_service()
        rec = await get_recommender()

        # 1. Record rejection event and apply negative penalty updates across all features
        event, updated_prof = mem.record_rejection(
            user_id=req.user_id,
            movie_id=req.movie_id,
            reason=req.reason,
            metadata=req.metadata
        )

        # 2. Exclude all current movies + rejected movie from replacement
        excluded = list(set(req.current_movie_ids + [req.movie_id]))
        subs = req.subscriptions if req.subscriptions else getattr(updated_prof, "subscriptions", ["Netflix", "Prime Video", "Disney+"])

        # 3. Pull fresh replacement in background
        replacement = await rec.get_replacement_recommendation(
            user_id=req.user_id,
            excluded_movie_ids=excluded,
            natural_language=req.natural_language,
            subscriptions=subs
        )

        # 4. Get updated profile summary
        summary = mem.get_user_profile_summary(req.user_id)

        return {
            "status": "success",
            "rejected_movie_id": req.movie_id,
            "replacement": replacement,
            "profile_summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profile/{user_id}")
async def get_user_profile(user_id: str):
    try:
        mem = get_memory_service()
        summary = mem.get_user_profile_summary(user_id)
        profile = mem.get_user_profile(user_id)
        serialized = mem._serialize_profile(profile)

        return {
            "profile": json.loads(serialized),
            "summary": summary,
            "combined_genres": summary.get("genres", []),
            "combined_tags": summary.get("tags", []),
            "combined_themes": summary.get("themes", []),
            "combined_tones": summary.get("tones", []),
            "combined_affinities": summary.get("keywords", []),
            "negative_penalties": summary.get("negative_penalties", []),
            "stats": summary.get("stats", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history/{user_id}")
async def get_user_history(user_id: str):
    try:
        mem = get_memory_service()
        events = mem.storage.get_events(user_id)
        ratings = mem.storage.get_ratings(user_id)
        
        event_list = [
            {
                "id": ev.id,
                "movie_id": ev.movie_id,
                "event_type": ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type),
                "signal_weight": ev.signal_weight,
                "timestamp": ev.timestamp,
                "metadata": ev.metadata
            }
            for ev in reversed(events)
        ]
        
        rating_list = [
            {
                "id": r.id,
                "movie_id": r.movie_id,
                "rating": r.rating,
                "signal_target": r.signal_target,
                "timestamp": r.timestamp,
                "review_text": r.review_text
            }
            for r in reversed(ratings)
        ]
        
        return {
            "events": event_list,
            "ratings": rating_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interact")
async def record_interaction(req: InteractionRequest):
    try:
        mem = get_memory_service()
        event_type = EventType(req.event_type) if req.event_type in [e.value for e in EventType] else EventType.MOVIE_CLICKED
        ev = mem.record_interaction(
            user_id=req.user_id,
            movie_id=req.movie_id,
            event_type=event_type,
            metadata=req.metadata
        )
        return {
            "status": "success",
            "event_id": ev.id,
            "event_type": ev.event_type.value,
            "signal_weight": ev.signal_weight
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rate")
async def record_rating(req: RateRequest):
    try:
        mem = get_memory_service()
        r = mem.record_rating(
            user_id=req.user_id,
            movie_id=req.movie_id,
            rating=req.rating,
            review_text=req.review_text,
            metadata=req.metadata
        )
        return {
            "status": "success",
            "rating_id": r.id,
            "rating": r.rating,
            "signal_target": r.signal_target
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/details/{movie_id}")
async def get_movie_details(movie_id: int):
    try:
        rec = await get_recommender()
        details = await rec.get_movie_details(movie_id)
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)

