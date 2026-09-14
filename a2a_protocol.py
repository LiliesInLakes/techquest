import json
import uuid
import asyncio
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone
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
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
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
                    "natural_language": {"type": "string"}
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
        await recommender.initialize()
        profile = recommender.analyze_user_preferences(
            watch_history=params.get("watch_history", []),
            ratings=params.get("ratings", {})
        )
        return {"preference_profile": profile}


class A2AClient:
    def __init__(self, agent_id: str, target_agent: Optional[A2AAgent] = None):
        self.agent_id = agent_id
        self.target_agent = target_agent
        self._pending_requests: Dict[str, asyncio.Future] = {}

    async def send_request(
        self,
        recipient_id: str,
        task: str,
        params: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        req_id = str(uuid.uuid4())
        message = A2AMessage(
            message_id=req_id,
            message_type=A2AMessageType.TASK_REQUEST,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            correlation_id=req_id,
            payload={"task": task, "params": params}
        )
        
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_requests[message.correlation_id] = future
        
        try:
            if self.target_agent:
                response = await self.target_agent.handle_message(message)
                self.handle_response(response)

            response = await asyncio.wait_for(future, timeout=timeout)
            if response.message_type == A2AMessageType.ERROR:
                raise RuntimeError(response.payload.get("error", "Unknown A2A error"))
            return response.payload.get("result", {})
        finally:
            self._pending_requests.pop(message.correlation_id, None)

    async def discover_agent(self, recipient_id: str, timeout: float = 10.0) -> Optional[AgentInfo]:
        req_id = str(uuid.uuid4())
        message = A2AMessage(
            message_id=req_id,
            message_type=A2AMessageType.AGENT_DISCOVERY,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            correlation_id=req_id
        )
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_requests[message.correlation_id] = future

        try:
            if self.target_agent:
                response = await self.target_agent.handle_message(message)
                self.handle_response(response)

            response = await asyncio.wait_for(future, timeout=timeout)
            info_dict = response.payload.get("agent_info", {})
            return AgentInfo(**info_dict) if info_dict else None
        finally:
            self._pending_requests.pop(message.correlation_id, None)

    def handle_response(self, message: A2AMessage):
        corr_id = message.correlation_id or message.message_id
        future = self._pending_requests.get(corr_id)
        if future and not future.done():
            future.set_result(message)


if __name__ == "__main__":
    agent = MovieRecommenderA2AAgent()
    print(f"Agent: {agent.agent_info.name}")
    print(f"Capabilities: {[c.name for c in agent.agent_info.capabilities]}")
