import json
import uuid
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod


class A2AMessageType(str, Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_STATUS = "task_status"
    AGENT_DISCOVERY = "agent_discovery"
    AGENT_CAPABILITIES = "agent_capabilities"
    ERROR = "error"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class A2AMessage:
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: A2AMessageType = A2AMessageType.TASK_REQUEST
    sender_id: str = ""
    recipient_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None


@dataclass
class AgentCapability:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    tags: List[str] = field(default_factory=list)


@dataclass
class AgentInfo:
    agent_id: str
    name: str
    description: str
    version: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    endpoints: Dict[str, str] = field(default_factory=dict)


class A2AAgent(ABC):
    def __init__(self, agent_id: str, name: str, description: str, version: str = "1.0.0"):
        self.agent_info = AgentInfo(
            agent_id=agent_id,
            name=name,
            description=description,
            version=version
        )
        self._handlers: Dict[A2AMessageType, Callable[[A2AMessage], Awaitable[A2AMessage]]] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        self._handlers[A2AMessageType.AGENT_DISCOVERY] = self._handle_discovery
        self._handlers[A2AMessageType.AGENT_CAPABILITIES] = self._handle_capabilities

    async def _handle_discovery(self, message: A2AMessage) -> A2AMessage:
        return A2AMessage(
            message_type=A2AMessageType.AGENT_CAPABILITIES,
            sender_id=self.agent_info.agent_id,
            recipient_id=message.sender_id,
            correlation_id=message.message_id,
            payload={"agent_info": asdict(self.agent_info)}
        )

    async def _handle_capabilities(self, message: A2AMessage) -> A2AMessage:
        capabilities = [
            {
                "name": cap.name,
                "description": cap.description,
                "input_schema": cap.input_schema,
                "output_schema": cap.output_schema,
                "tags": cap.tags
            }
            for cap in self.agent_info.capabilities
        ]
        return A2AMessage(
            message_type=A2AMessageType.TASK_RESPONSE,
            sender_id=self.agent_info.agent_id,
            recipient_id=message.sender_id,
            correlation_id=message.message_id,
            payload={"capabilities": capabilities}
        )

    def register_capability(self, capability: AgentCapability):
        self.agent_info.capabilities.append(capability)

    def register_handler(self, message_type: A2AMessageType, handler: Callable[[A2AMessage], Awaitable[A2AMessage]]):
        self._handlers[message_type] = handler

    async def handle_message(self, message: A2AMessage) -> A2AMessage:
        handler = self._handlers.get(message.message_type)
        if not handler:
            return A2AMessage(
                message_type=A2AMessageType.ERROR,
                sender_id=self.agent_info.agent_id,
                recipient_id=message.sender_id,
                correlation_id=message.message_id,
                payload={"error": f"No handler for message type: {message.message_type}"}
            )
        try:
            return await handler(message)
        except Exception as e:
            return A2AMessage(
                message_type=A2AMessageType.ERROR,
                sender_id=self.agent_info.agent_id,
                recipient_id=message.sender_id,
                correlation_id=message.message_id,
                payload={"error": str(e)}
            )


class MovieRecommenderA2AAgent(A2AAgent):
    def __init__(self):
        super().__init__(
            agent_id="movie-recommender-001",
            name="Movie Recommender Agent",
            description="Provides personalized movie recommendations for hospitality/airline businesses",
            version="1.0.0"
        )
        self._register_capabilities()
        self._register_task_handlers()

    def _register_capabilities(self):
        self.register_capability(AgentCapability(
            name="search_movies",
            description="Search movies by genre, region, language, and subscription availability",
            input_schema={
                "type": "object",
                "properties": {
                    "genre": {"type": "string"},
                    "region": {"type": "string"},
                    "language": {"type": "string"},
                    "subscription_services": {"type": "array", "items": {"type": "string"}},
                    "min_rating": {"type": "number"},
                    "max_results": {"type": "integer"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "movies": {"type": "array", "items": {"type": "object"}}
                }
            },
            tags=["search", "movies", "tmdb"]
        ))

        self.register_capability(AgentCapability(
            name="get_recommendations",
            description="Get personalized movie recommendations based on user watch history and natural language query",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "watch_history": {"type": "array", "items": {"type": "object"}},
                    "preferences": {"type": "object"},
                    "subscriptions": {"type": "array", "items": {"type": "string"}},
                    "region": {"type": "string"},
                    "language": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "natural_language": {"type": "string"},
                    "partner_id": {"type": "string"},
                    "session_id": {"type": "string"}
                },
                "required": ["user_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "recommendations": {"type": "array", "items": {"type": "object"}},
                    "reasoning": {"type": "string"},
                    "extracted_features": {"type": "object"}
                }
            },
            tags=["recommendations", "personalization", "llm", "natural-language"]
        ))

        self.register_capability(AgentCapability(
            name="analyze_preferences",
            description="Analyze user watch history to extract preference profile",
            input_schema={
                "type": "object",
                "properties": {
                    "watch_history": {"type": "array", "items": {"type": "object"}},
                    "ratings": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "preference_profile": {"type": "object"},
                    "top_genres": {"type": "array", "items": {"type": "string"}},
                    "top_keywords": {"type": "array", "items": {"type": "string"}},
                    "rating_patterns": {"type": "object"}
                }
            },
            tags=["analysis", "preferences", "user-profiling"]
        ))

        self.register_capability(AgentCapability(
            name="create_session",
            description="Create a new user session with context for personalized recommendations",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "partner_id": {"type": "string"},
                    "device_type": {"type": "string", "enum": ["seatback", "mobile", "tablet", "laptop", "tv", "vr"]},
                    "party_type": {"type": "string", "enum": ["solo", "couple", "family", "friends", "business"]},
                    "party_size": {"type": "integer"},
                    "region": {"type": "string"},
                    "language": {"type": "string"},
                    "subscriptions": {"type": "array", "items": {"type": "string"}},
                    "flight_duration_minutes": {"type": "integer"},
                    "flight_number": {"type": "string"},
                    "hotel_room_type": {"type": "string"},
                    "check_in_date": {"type": "string"},
                    "check_out_date": {"type": "string"},
                    "subscription_tier": {"type": "string"},
                    "offline_mode": {"type": "boolean"},
                    "metadata": {"type": "object"}
                },
                "required": ["user_id", "partner_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "session": {"type": "object"}
                }
            },
            tags=["session", "context", "personalization"]
        ))

        self.register_capability(AgentCapability(
            name="end_session",
            description="End a user session and record watch events for profile updates",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "watch_events": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["session_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "session": {"type": "object"},
                    "updated_profile": {"type": "object"}
                }
            },
            tags=["session", "analytics", "profile-update"]
        ))

        self.register_capability(AgentCapability(
            name="assess_churn_risk",
            description="Assess user churn risk and get recommended retention actions",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "partner_id": {"type": "string"},
                    "user_data": {"type": "object"}
                },
                "required": ["user_id", "partner_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "risk_score": {"type": "number"},
                    "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "top_risk_factors": {"type": "array", "items": {"type": "object"}},
                    "recommended_actions": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "predicted_churn_date": {"type": "string"},
                    "estimated_ltv_loss": {"type": "number"}
                }
            },
            tags=["churn", "retention", "business-intelligence"]
        ))

        self.register_capability(AgentCapability(
            name="trigger_business_action",
            description="Trigger a business action (upgrade offer, reengage notification, playlist, etc.)",
            input_schema={
                "type": "object",
                "properties": {
                    "action_type": {"type": "string", "enum": ["recommend_movie", "recommend_series", "offer_bundle_upgrade", "offer_discounted_subscription", "recommend_premium_content", "reengage_notification", "personalized_playlist"]},
                    "partner_id": {"type": "string"},
                    "user_id": {"type": "string"},
                    "trigger": {"type": "string", "enum": ["session_start", "session_end", "churn_risk_high", "content_completed", "search_failed", "subscription_expiring", "new_content_available", "manual_request"]},
                    "payload": {"type": "object"},
                    "reasoning": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "expires_in_hours": {"type": "integer"}
                },
                "required": ["action_type", "partner_id", "user_id", "trigger", "payload", "reasoning"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "action_id": {"type": "string"},
                    "result": {"type": "object"}
                }
            },
            tags=["business-actions", "retention", "upsell", "engagement"]
        ))

        self.register_capability(AgentCapability(
            name="ingest_events",
            description="Ingest analytics events for tracking and analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "event_type": {"type": "string"},
                                "partner_id": {"type": "string"},
                                "user_id": {"type": "string"},
                                "session_id": {"type": "string"},
                                "payload": {"type": "object"},
                                "metadata": {"type": "object"}
                            },
                            "required": ["event_type", "partner_id", "user_id", "payload"]
                        }
                    }
                },
                "required": ["events"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "ingested": {"type": "integer"},
                    "total": {"type": "integer"}
                }
            },
            tags=["analytics", "events", "tracking"]
        ))

        self.register_capability(AgentCapability(
            name="apply_business_rules",
            description="Apply partner-specific business rules to filter and rank movies",
            input_schema={
                "type": "object",
                "properties": {
                    "partner_id": {"type": "string"},
                    "context": {"type": "object"},
                    "movies": {"type": "array", "items": {"type": "object"}}
                },
                "required": ["partner_id", "context", "movies"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "filtered_movies": {"type": "array", "items": {"type": "object"}},
                    "filters_applied": {"type": "object"},
                    "weight_adjustments": {"type": "object"},
                    "triggered_actions": {"type": "array", "items": {"type": "object"}},
                    "warnings": {"type": "array", "items": {"type": "string"}}
                }
            },
            tags=["business-rules", "filtering", "ranking", "partner-config"]
        ))

        self.register_capability(AgentCapability(
            name="get_partner_config",
            description="Get partner configuration including business rules and content filters",
            input_schema={
                "type": "object",
                "properties": {
                    "partner_id": {"type": "string"}
                },
                "required": ["partner_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "partner_id": {"type": "string"},
                    "name": {"type": "string"},
                    "content_filters": {"type": "object"},
                    "ranking_weights": {"type": "object"},
                    "license_regions": {"type": "array", "items": {"type": "string"}},
                    "partner_content_boost": {"type": "object"},
                    "competitor_exclude": {"type": "array", "items": {"type": "string"}},
                    "max_runtime_minutes": {"type": "integer"},
                    "allowed_content_ratings": {"type": "array", "items": {"type": "string"}},
                    "offline_required": {"type": "boolean"},
                    "session_limits": {"type": "object"}
                }
            },
            tags=["partner-config", "business-rules", "content-filters"]
        ))

    def _register_task_handlers(self):
        self.register_handler(A2AMessageType.TASK_REQUEST, self._handle_task_request)

    async def _handle_task_request(self, message: A2AMessage) -> A2AMessage:
        task_name = message.payload.get("task")
        params = message.payload.get("params", {})
        
        if task_name == "search_movies":
            result = await self._search_movies(params)
        elif task_name == "get_recommendations":
            result = await self._get_recommendations(params)
        elif task_name == "analyze_preferences":
            result = await self._analyze_preferences(params)
        elif task_name == "create_session":
            result = await self._create_session(params)
        elif task_name == "end_session":
            result = await self._end_session(params)
        elif task_name == "assess_churn_risk":
            result = await self._assess_churn_risk(params)
        elif task_name == "trigger_business_action":
            result = await self._trigger_business_action(params)
        elif task_name == "ingest_events":
            result = await self._ingest_events(params)
        elif task_name == "apply_business_rules":
            result = await self._apply_business_rules(params)
        elif task_name == "get_partner_config":
            result = await self._get_partner_config(params)
        else:
            return A2AMessage(
                message_type=A2AMessageType.ERROR,
                sender_id=self.agent_info.agent_id,
                recipient_id=message.sender_id,
                correlation_id=message.message_id,
                payload={"error": f"Unknown task: {task_name}"}
            )

        return A2AMessage(
            message_type=A2AMessageType.TASK_RESPONSE,
            sender_id=self.agent_info.agent_id,
            recipient_id=message.sender_id,
            correlation_id=message.message_id,
            payload={"result": result}
        )

    async def _search_movies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from src.tmdb.client import TMDBClient
        from src.pipeline.recommender import MovieRecommender
        
        recommender = MovieRecommender()
        movies = await recommender.search_movies(
            genre=params.get("genre"),
            region=params.get("region"),
            language=params.get("language"),
            subscriptions=params.get("subscription_services", []),
            min_rating=params.get("min_rating", 0.0),
            max_results=params.get("max_results", 100)
        )
        return {"movies": movies}

    async def _get_recommendations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from src.pipeline.recommender import MovieRecommender
        
        recommender = MovieRecommender()
        result = await recommender.get_recommendations(
            user_id=params["user_id"],
            watch_history=params.get("watch_history", []),
            preferences=params.get("preferences", {}),
            subscriptions=params.get("subscriptions", []),
            region=params.get("region", "US"),
            language=params.get("language", "en"),
            top_k=params.get("top_k", 10),
            natural_language=params.get("natural_language")
        )
        return result

    async def _analyze_preferences(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from src.pipeline.recommender import MovieRecommender
        
        recommender = MovieRecommender()
        profile = recommender.analyze_user_preferences(
            watch_history=params.get("watch_history", []),
            ratings=params.get("ratings", {})
        )
        return {"preference_profile": profile}

    async def _create_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from src.business.session_context import SessionManager
        
        manager = SessionManager()
        session = manager.create_session(
            user_id=params["user_id"],
            partner_id=params["partner_id"],
            context_overrides={k: v for k, v in params.items() if k not in ["user_id", "partner_id"]}
        )
        return {"session_id": session.session_id, "session": session.to_dict()}

    async def _end_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from src.business.session_context import SessionManager
        
        manager = SessionManager()
        session = manager.end_session(params["session_id"], params.get("watch_events", []))
        if session:
            profile = manager.get_user_profile(session.user_id, session.partner_id)
            return {"session": session.to_dict(), "updated_profile": {
                "total_sessions": profile.total_sessions,
                "total_watch_time": profile.total_watch_time_minutes,
                "preferred_genres": profile.preferred_genres,
                "completion_rate": profile.completion_rate
            }}
        return {"error": "Session not found"}

    async def _assess_churn_risk(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from src.business.churn_risk import ChurnRiskService
        
        service = ChurnRiskService()
        user_data = params.get("user_data", {})
        user_data["user_id"] = params["user_id"]
        user_data["partner_id"] = params["partner_id"]
        prediction = service.assess_user(user_data)
        return {
            "risk_score": prediction.risk_score,
            "risk_level": prediction.risk_level.value,
            "top_risk_factors": prediction.top_risk_factors,
            "recommended_actions": [a.value for a in prediction.recommended_actions],
            "confidence": prediction.confidence,
            "predicted_churn_date": prediction.predicted_churn_date,
            "estimated_ltv_loss": prediction.estimated_ltv_loss
        }

    async def _trigger_business_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from src.business.actions import BusinessActionEngine, ActionType, TriggerType, ActionPriority, create_default_action_engine
        
        engine = create_default_action_engine()
        action = engine.create_action(
            action_type=ActionType(params["action_type"]),
            partner_id=params["partner_id"],
            user_id=params["user_id"],
            trigger=TriggerType(params["trigger"]),
            payload=params["payload"],
            reasoning=params["reasoning"],
            priority=ActionPriority(params.get("priority", "medium")),
            expires_in_hours=params.get("expires_in_hours")
        )
        result = engine.execute_action(action)
        return {"action_id": action.action_id, "result": result}

    async def _ingest_events(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from src.business.events import EventIngestion, build_event, EventType, create_ingestion_pipeline
        
        ingestion = create_ingestion_pipeline()
        ingestion.start()
        
        events = []
        for e in params["events"]:
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
        
        return {"ingested": count, "total": len(params["events"])}

    async def _apply_business_rules(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from src.business.rules_engine import create_default_rules_engine
        
        engine = create_default_rules_engine()
        filtered = engine.apply_content_filters(params["movies"], params["partner_id"], params["context"])
        eval_result = engine.evaluate(params["partner_id"], params["context"])
        
        return {
            "filtered_movies": filtered,
            "filters_applied": eval_result["filters"],
            "weight_adjustments": eval_result["weight_adjustments"],
            "triggered_actions": eval_result["actions"],
            "warnings": eval_result["warnings"]
        }

    async def _get_partner_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from src.business.rules_engine import create_default_rules_engine
        
        engine = create_default_rules_engine()
        config = engine.get_partner_config(params["partner_id"])
        if config:
            return {
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
            }
        return {"error": "Partner not found"}


class A2AClient:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._pending_requests: Dict[str, asyncio.Future] = {}

    async def send_request(
        self,
        recipient_id: str,
        task: str,
        params: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        message = A2AMessage(
            message_type=A2AMessageType.TASK_REQUEST,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            payload={"task": task, "params": params}
        )
        
        future = asyncio.Future()
        self._pending_requests[message.correlation_id] = future
        
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response.payload.get("result", {})
        finally:
            self._pending_requests.pop(message.correlation_id, None)

    async def discover_agent(self, recipient_id: str) -> AgentInfo:
        message = A2AMessage(
            message_type=A2AMessageType.AGENT_DISCOVERY,
            sender_id=self.agent_id,
            recipient_id=recipient_id
        )
        return None

    def handle_response(self, message: A2AMessage):
        future = self._pending_requests.get(message.correlation_id)
        if future and not future.done():
            future.set_result(message)


if __name__ == "__main__":
    agent = MovieRecommenderA2AAgent()
    print(f"Agent: {agent.agent_info.name}")
    print(f"Capabilities: {[c.name for c in agent.agent_info.capabilities]}")
