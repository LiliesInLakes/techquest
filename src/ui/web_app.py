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
    top_k: int = 6


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
            natural_language=req.natural_language
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


@app.get("/api/profile/{user_id}")
async def get_user_profile(user_id: str):
    try:
        mem = get_memory_service()
        profile = mem.get_user_profile(user_id)
        serialized = mem._serialize_profile(profile)
        
        # Calculate dynamic beta & learning curve metrics
        n = profile.total_interactions
        beta = 0.40 + 0.60 * max(0.0, 1.0 - n / 10.0)
        conf_pct = round((n / (n + 10.0)) * 100, 1) if (n + 10.0) > 0 else 0.0

        if n < 4:
            stage_text = "🌱 Initial Discovery"
            stability_text = "Exploration"
        elif n < 10:
            stage_text = "⚡ Active Calibration"
            stability_text = "Balancing"
        else:
            stage_text = "🎯 Tuned Profile"
            stability_text = "Calibrated"

        # Calculate combined affinities & genres
        combined = mem.get_combined_preference_profile(user_id)
        combined_keywords = [
            {"name": k, "score": round(score, 3)} 
            for k, score in getattr(combined, "top_keywords", []) if score > 0
        ]
        combined_genres = [
            {"name": g, "score": round(score, 3)} 
            for g, score in getattr(combined, "top_genres", []) if score > 0
        ]
        
        return {
            "profile": json.loads(serialized),
            "combined_affinities": combined_keywords,
            "combined_genres": combined_genres,
            "stats": {
                "total_interactions": n,
                "beta": round(beta, 3),
                "short_term_pct": round(beta * 100, 1),
                "long_term_pct": round((1 - beta) * 100, 1),
                "confidence_pct": conf_pct,
                "stage": stage_text,
                "stability": stability_text
            }
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

