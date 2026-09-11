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
    CREATE_SESSION = "create_session"
    END_SESSION = "end_session"
    GET_CHURN_RISK = "get_churn_risk"
    TRIGGER_BUSINESS_ACTION = "trigger_business_action"
    INGEST_EVENTS = "ingest_events"
    APPLY_BUSINESS_RULES = "apply_business_rules"
    GET_PARTNER_CONFIG = "get_partner_config"


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
    partner_id: Optional[str] = None
    session_id: Optional[str] = None


class MovieDetailsParams(BaseModel):
    movie_id: int
    include_credits: bool = True
    include_keywords: bool = True


class AnalyzePreferencesParams(BaseModel):
    watch_history: List[Dict[str, Any]]
    ratings: Dict[int, float] = Field(default_factory=dict)


class CreateSessionParams(BaseModel):
    user_id: str
    partner_id: str
    device_type: str = "mobile"
    party_type: str = "solo"
    party_size: int = 1
    region: str = "US"
    language: str = "en"
    subscriptions: List[str] = Field(default_factory=list)
    flight_duration_minutes: Optional[int] = None
    flight_number: Optional[str] = None
    hotel_room_type: Optional[str] = None
    check_in_date: Optional[str] = None
    check_out_date: Optional[str] = None
    subscription_tier: str = "basic"
    offline_mode: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EndSessionParams(BaseModel):
    session_id: str
    watch_events: List[Dict[str, Any]] = Field(default_factory=list)


class GetChurnRiskParams(BaseModel):
    user_id: str
    partner_id: str
    user_data: Dict[str, Any] = Field(default_factory=dict)


class TriggerBusinessActionParams(BaseModel):
    action_type: str
    partner_id: str
    user_id: str
    trigger: str
    payload: Dict[str, Any]
    reasoning: str
    priority: str = "medium"
    expires_in_hours: Optional[int] = None


class IngestEventsParams(BaseModel):
    events: List[Dict[str, Any]]


class ApplyBusinessRulesParams(BaseModel):
    partner_id: str
    context: Dict[str, Any]
    movies: List[Dict[str, Any]]


class GetPartnerConfigParams(BaseModel):
    partner_id: str


@dataclass
class MCPServer:
    name: str = "movie-recommender"
    version: str = "1.0.0"
    description: str = "Movie recommendation agent for hospitality/airline businesses with business intelligence"

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name=MCPToolType.SEARCH_MOVIES.value,
                description="Search movies by genre, region, language, and subscription availability",
                inputSchema=SearchMoviesParams.model_json_schema()
            ),
            Tool(
                name=MCPToolType.GET_RECOMMENDATIONS.value,
                description="Get personalized movie recommendations based on user watch history and preferences",
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
            ),
            Tool(
                name=MCPToolType.CREATE_SESSION.value,
                description="Create a new user session with context for personalized recommendations",
                inputSchema=CreateSessionParams.model_json_schema()
            ),
            Tool(
                name=MCPToolType.END_SESSION.value,
                description="End a user session and record watch events",
                inputSchema=EndSessionParams.model_json_schema()
            ),
            Tool(
                name=MCPToolType.GET_CHURN_RISK.value,
                description="Assess user churn risk and get recommended retention actions",
                inputSchema=GetChurnRiskParams.model_json_schema()
            ),
            Tool(
                name=MCPToolType.TRIGGER_BUSINESS_ACTION.value,
                description="Trigger a business action (upgrade offer, reengage notification, etc.)",
                inputSchema=TriggerBusinessActionParams.model_json_schema()
            ),
            Tool(
                name=MCPToolType.INGEST_EVENTS.value,
                description="Ingest analytics events for tracking and analysis",
                inputSchema=IngestEventsParams.model_json_schema()
            ),
            Tool(
                name=MCPToolType.APPLY_BUSINESS_RULES.value,
                description="Apply partner-specific business rules to filter and rank movies",
                inputSchema=ApplyBusinessRulesParams.model_json_schema()
            ),
            Tool(
                name=MCPToolType.GET_PARTNER_CONFIG.value,
                description="Get partner configuration including business rules and content filters",
                inputSchema=GetPartnerConfigParams.model_json_schema()
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
        elif tool_name == MCPToolType.CREATE_SESSION.value:
            return await create_session(CreateSessionParams(**body))
        elif tool_name == MCPToolType.END_SESSION.value:
            return await end_session(EndSessionParams(**body))
        elif tool_name == MCPToolType.GET_CHURN_RISK.value:
            return await get_churn_risk(GetChurnRiskParams(**body))
        elif tool_name == MCPToolType.TRIGGER_BUSINESS_ACTION.value:
            return await trigger_business_action(TriggerBusinessActionParams(**body))
        elif tool_name == MCPToolType.INGEST_EVENTS.value:
            return await ingest_events(IngestEventsParams(**body))
        elif tool_name == MCPToolType.APPLY_BUSINESS_RULES.value:
            return await apply_business_rules(ApplyBusinessRulesParams(**body))
        elif tool_name == MCPToolType.GET_PARTNER_CONFIG.value:
            return await get_partner_config(GetPartnerConfigParams(**body))
        else:
            raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def search_movies(params: SearchMoviesParams) -> CallToolResult:
    from src.tmdb.client import TMDBClient
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
        natural_language=params.natural_language,
        partner_id=params.partner_id,
        session_id=params.session_id
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


async def create_session(params: CreateSessionParams) -> CallToolResult:
    from src.business.session_context import SessionManager

    manager = SessionManager()
    session = manager.create_session(
        user_id=params.user_id,
        partner_id=params.partner_id,
        context_overrides={
            "device_type": params.device_type,
            "party_type": params.party_type,
            "party_size": params.party_size,
            "region": params.region,
            "language": params.language,
            "subscriptions": params.subscriptions,
            "flight_duration_minutes": params.flight_duration_minutes,
            "flight_number": params.flight_number,
            "hotel_room_type": params.hotel_room_type,
            "check_in_date": params.check_in_date,
            "check_out_date": params.check_out_date,
            "subscription_tier": params.subscription_tier,
            "offline_mode": params.offline_mode,
            "metadata": params.metadata
        }
    )
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(session.to_dict(), indent=2))])


async def end_session(params: EndSessionParams) -> CallToolResult:
    from src.business.session_context import SessionManager

    manager = SessionManager()
    session = manager.end_session(params.session_id, params.watch_events)
    if session:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(session.to_dict(), indent=2))])
    else:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": "Session not found"}))])


async def get_churn_risk(params: GetChurnRiskParams) -> CallToolResult:
    from src.business.churn_risk import ChurnRiskService

    service = ChurnRiskService()
    user_data = params.user_data.copy()
    user_data["user_id"] = params.user_id
    user_data["partner_id"] = params.partner_id
    prediction = service.assess_user(user_data)
    return CallToolResult(content=[TextContent(type="text", text=json.dumps({
        "user_id": prediction.user_id,
        "partner_id": prediction.partner_id,
        "risk_score": prediction.risk_score,
        "risk_level": prediction.risk_level.value,
        "top_risk_factors": prediction.top_risk_factors,
        "recommended_actions": [a.value for a in prediction.recommended_actions],
        "confidence": prediction.confidence,
        "predicted_churn_date": prediction.predicted_churn_date,
        "estimated_ltv_loss": prediction.estimated_ltv_loss
    }, indent=2))])


async def trigger_business_action(params: TriggerBusinessActionParams) -> CallToolResult:
    from src.business.actions import BusinessActionEngine, ActionType, TriggerType, ActionPriority, create_default_action_engine

    engine = create_default_action_engine()
    action = engine.create_action(
        action_type=ActionType(params.action_type),
        partner_id=params.partner_id,
        user_id=params.user_id,
        trigger=TriggerType(params.trigger),
        payload=params.payload,
        reasoning=params.reasoning,
        priority=ActionPriority(params.priority),
        expires_in_hours=params.expires_in_hours
    )
    result = engine.execute_action(action)
    return CallToolResult(content=[TextContent(type="text", text=json.dumps({
        "action_id": action.action_id,
        "result": result
    }, indent=2))])


async def ingest_events(params: IngestEventsParams) -> CallToolResult:
    from src.business.events import EventIngestion, build_event, EventType, create_ingestion_pipeline

    ingestion = create_ingestion_pipeline()
    ingestion.start()

    events = []
    for e in params.events:
        event = build_event(
            event_type=EventType(e["event_type"]),
            partner_id=e["partner_id"],
            user_id=e["user_id"],
            session_id=e.get("session_id"),
            payload=e["payload"],
            metadata=e.get("metadata", {})
        )
        events.append(event)

    count = ingestion.ingest_batch(events)
    ingestion.stop()

    return CallToolResult(content=[TextContent(type="text", text=json.dumps({
        "ingested": count,
        "total": len(params.events)
    }, indent=2))])


async def apply_business_rules(params: ApplyBusinessRulesParams) -> CallToolResult:
    from src.business.rules_engine import create_default_rules_engine

    engine = create_default_rules_engine()
    filtered = engine.apply_content_filters(params.movies, params.partner_id, params.context)
    eval_result = engine.evaluate(params.partner_id, params.context)

    return CallToolResult(content=[TextContent(type="text", text=json.dumps({
        "filtered_movies": filtered,
        "filters_applied": eval_result["filters"],
        "weight_adjustments": eval_result["weight_adjustments"],
        "triggered_actions": eval_result["actions"],
        "warnings": eval_result["warnings"]
    }, indent=2))])


async def get_partner_config(params: GetPartnerConfigParams) -> CallToolResult:
    from src.business.rules_engine import create_default_rules_engine

    engine = create_default_rules_engine()
    config = engine.get_partner_config(params.partner_id)
    if config:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({
            "partner_id": config.partner_id,
            "name": config.name,
            "content_filters": config.content_filters,
            "ranking_weights": config.ranking_weights,
            "license_regions": config.license_regions,
            "partner_content_boost": config.partner_content_boost,
            "competitor_exclude": config.competitor_exclude,
            "max_runtime_minutes": config.max_runtime_minutes,
            "allowed_content_ratings": [r.value for r in config.allowed_content_ratings],
            "offline_required": config.offline_required,
            "session_limits": config.session_limits
        }, indent=2))])
    else:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": "Partner not found"}))])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)