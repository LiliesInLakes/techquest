import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod


class ActionType(str, Enum):
    RECOMMEND_MOVIE = "recommend_movie"
    RECOMMEND_SERIES = "recommend_series"
    OFFER_BUNDLE_UPGRADE = "offer_bundle_upgrade"
    OFFER_DISCOUNTED_SUBSCRIPTION = "offer_discounted_subscription"
    RECOMMEND_PREMIUM_CONTENT = "recommend_premium_content"
    REENGAGE_NOTIFICATION = "reengage_notification"
    CONTENT_WARNING = "content_warning"
    PERSONALIZED_PLAYLIST = "personalized_playlist"
    SOCIAL_RECOMMENDATION = "social_recommendation"
    CONTINUE_WATCHING = "continue_watching"


class ActionPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TriggerType(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CHURN_RISK_HIGH = "churn_risk_high"
    CONTENT_COMPLETED = "content_completed"
    SEARCH_FAILED = "search_failed"
    RECOMMENDATION_IGNORED = "recommendation_ignored"
    SUBSCRIPTION_EXPIRING = "subscription_expiring"
    NEW_CONTENT_AVAILABLE = "new_content_available"
    MANUAL_REQUEST = "manual_request"


@dataclass
class BusinessAction:
    action_id: str
    action_type: ActionType
    partner_id: str
    user_id: str
    trigger: TriggerType
    priority: ActionPriority
    payload: Dict[str, Any]
    reasoning: str
    expires_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    executed: bool = False
    executed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        d["trigger"] = self.trigger.value
        d["priority"] = self.priority.value
        return d


class ActionExecutor(ABC):
    @abstractmethod
    def execute(self, action: BusinessAction) -> Dict[str, Any]:
        pass

    @abstractmethod
    def can_execute(self, action: BusinessAction) -> bool:
        pass


class RecommendMovieExecutor(ActionExecutor):
    def __init__(self, recommender):
        self.recommender = recommender

    def can_execute(self, action: BusinessAction) -> bool:
        return action.action_type == ActionType.RECOMMEND_MOVIE

    def execute(self, action: BusinessAction) -> Dict[str, Any]:
        payload = action.payload
        return {
            "status": "queued",
            "recommendation_request": {
                "user_id": action.user_id,
                "natural_language": payload.get("query", ""),
                "context": payload.get("context", {}),
                "top_k": payload.get("top_k", 5)
            }
        }


class BundleUpgradeExecutor(ActionExecutor):
    def __init__(self, catalog_service=None):
        self.catalog_service = catalog_service

    def can_execute(self, action: BusinessAction) -> bool:
        return action.action_type == ActionType.OFFER_BUNDLE_UPGRADE

    def execute(self, action: BusinessAction) -> Dict[str, Any]:
        payload = action.payload
        current_tier = payload.get("current_tier", "basic")
        eligible_tiers = payload.get("eligible_tiers", ["standard", "premium"])

        upgrade_options = []
        for tier in eligible_tiers:
            if self._is_upgrade(current_tier, tier):
                upgrade_options.append({
                    "tier": tier,
                    "monthly_price": self._get_price(tier),
                    "benefits": self._get_benefits(tier),
                    "discount_percent": payload.get("discount_percent", 0)
                })

        return {
            "status": "success",
            "offer": {
                "type": "bundle_upgrade",
                "current_tier": current_tier,
                "options": upgrade_options,
                "expires_in_days": payload.get("expires_in_days", 7)
            }
        }

    def _is_upgrade(self, current: str, target: str) -> bool:
        tiers = ["basic", "standard", "premium", "vip"]
        return tiers.index(target) > tiers.index(current) if current in tiers and target in tiers else False

    def _get_price(self, tier: str) -> float:
        return {"basic": 9.99, "standard": 14.99, "premium": 22.99, "vip": 39.99}.get(tier, 9.99)

    def _get_benefits(self, tier: str) -> List[str]:
        benefits = {
            "basic": ["HD streaming", "1 device"],
            "standard": ["Full HD", "2 devices", "Download"],
            "premium": ["4K HDR", "4 devices", "Download", "Dolby Atmos"],
            "vip": ["4K HDR", "Unlimited devices", "Download", "Dolby Atmos", "Early access"]
        }
        return benefits.get(tier, [])


class DiscountedSubscriptionExecutor(ActionExecutor):
    def can_execute(self, action: BusinessAction) -> bool:
        return action.action_type == ActionType.OFFER_DISCOUNTED_SUBSCRIPTION

    def execute(self, action: BusinessAction) -> Dict[str, Any]:
        payload = action.payload
        return {
            "status": "success",
            "offer": {
                "type": "discounted_subscription",
                "discount_percent": payload.get("discount_percent", 20),
                "duration_months": payload.get("duration_months", 3),
                "applicable_tiers": payload.get("applicable_tiers", ["standard", "premium"]),
                "promo_code": f"SAVE{payload.get('discount_percent', 20)}",
                "expires_at": payload.get("expires_at")
            }
        }


class ReengageNotificationExecutor(ActionExecutor):
    def can_execute(self, action: BusinessAction) -> bool:
        return action.action_type == ActionType.REENGAGE_NOTIFICATION

    def execute(self, action: BusinessAction) -> Dict[str, Any]:
        payload = action.payload
        channel = payload.get("channel", "push")
        template = payload.get("template", "we_miss_you")

        templates = {
            "we_miss_you": "We miss you! Here's a movie picked just for you.",
            "new_content": "New {genre} movies added this week!",
            "continue_watching": "Pick up where you left off with {title}",
            "personalized": "Based on your love for {genre}, check out {title}",
            "discount": "Come back and get {discount}% off your next month!"
        }

        return {
            "status": "success",
            "notification": {
                "channel": channel,
                "template": template,
                "message": templates.get(template, templates["we_miss_you"]).format(**payload.get("template_vars", {})),
                "deep_link": payload.get("deep_link", "/recommendations"),
                "payload": payload.get("template_vars", {})
            }
        }


class PersonalizedPlaylistExecutor(ActionExecutor):
    def __init__(self, recommender):
        self.recommender = recommender

    def can_execute(self, action: BusinessAction) -> bool:
        return action.action_type == ActionType.PERSONALIZED_PLAYLIST

    def execute(self, action: BusinessAction) -> Dict[str, Any]:
        payload = action.payload
        return {
            "status": "queued",
            "playlist_request": {
                "user_id": action.user_id,
                "theme": payload.get("theme", "mix"),
                "duration_minutes": payload.get("duration_minutes", 120),
                "context": payload.get("context", {})
            }
        }


class BusinessActionEngine:
    def __init__(self):
        self.executors: Dict[ActionType, ActionExecutor] = {}
        self.action_queue: List[BusinessAction] = []
        self.action_history: List[BusinessAction] = []
        self.max_history = 10000

    def register_executor(self, executor: ActionExecutor):
        for action_type in ActionType:
            if executor.can_execute(BusinessAction(
                action_id="test", action_type=action_type, partner_id="", user_id="",
                trigger=TriggerType.MANUAL_REQUEST, priority=ActionPriority.LOW,
                payload={}, reasoning=""
            )):
                self.executors[action_type] = executor
                break

    def create_action(
        self,
        action_type: ActionType,
        partner_id: str,
        user_id: str,
        trigger: TriggerType,
        payload: Dict[str, Any],
        reasoning: str,
        priority: ActionPriority = ActionPriority.MEDIUM,
        expires_in_hours: Optional[int] = None
    ) -> BusinessAction:
        expires_at = None
        if expires_in_hours:
            from datetime import timedelta
            expires_at = (datetime.utcnow() + timedelta(hours=expires_in_hours)).isoformat()

        action = BusinessAction(
            action_id=str(uuid.uuid4()),
            action_type=action_type,
            partner_id=partner_id,
            user_id=user_id,
            trigger=trigger,
            priority=priority,
            payload=payload,
            reasoning=reasoning,
            expires_at=expires_at
        )

        self.action_queue.append(action)
        return action

    def execute_next(self) -> Optional[Dict[str, Any]]:
        if not self.action_queue:
            return None

        self.action_queue.sort(key=lambda a: {"critical": 0, "high": 1, "medium": 2, "low": 3}[a.priority.value])

        action = self.action_queue.pop(0)
        executor = self.executors.get(action.action_type)

        if not executor:
            action.executed = True
            action.executed_at = datetime.utcnow().isoformat()
            action.result = {"status": "error", "message": f"No executor for {action.action_type}"}
            self.action_history.append(action)
            return action.result

        try:
            result = executor.execute(action)
            action.executed = True
            action.executed_at = datetime.utcnow().isoformat()
            action.result = result
        except Exception as e:
            action.executed = True
            action.executed_at = datetime.utcnow().isoformat()
            action.result = {"status": "error", "message": str(e)}

        self.action_history.append(action)
        if len(self.action_history) > self.max_history:
            self.action_history = self.action_history[-self.max_history:]

        return action.result

    def execute_action(self, action: BusinessAction) -> Dict[str, Any]:
        executor = self.executors.get(action.action_type)
        if not executor:
            return {"status": "error", "message": f"No executor for {action.action_type}"}

        try:
            result = executor.execute(action)
            action.executed = True
            action.executed_at = datetime.utcnow().isoformat()
            action.result = result
            return result
        except Exception as e:
            action.executed = True
            action.executed_at = datetime.utcnow().isoformat()
            action.result = {"status": "error", "message": str(e)}
            return action.result

    def get_pending_actions(self, user_id: str = None, partner_id: str = None) -> List[BusinessAction]:
        actions = self.action_queue
        if user_id:
            actions = [a for a in actions if a.user_id == user_id]
        if partner_id:
            actions = [a for a in actions if a.partner_id == partner_id]
        return actions

    def get_action_history(self, user_id: str = None, limit: int = 100) -> List[BusinessAction]:
        actions = self.action_history
        if user_id:
            actions = [a for a in actions if a.user_id == user_id]
        return actions[-limit:]

    def cancel_action(self, action_id: str) -> bool:
        for i, action in enumerate(self.action_queue):
            if action.action_id == action_id:
                self.action_queue.pop(i)
                return True
        return False


ACTION_TRIGGER_MAP = {
    TriggerType.SESSION_START: [
        (ActionType.RECOMMEND_MOVIE, "Welcome back! Here are picks for your session"),
        (ActionType.CONTINUE_WATCHING, "Continue where you left off")
    ],
    TriggerType.CHURN_RISK_HIGH: [
        (ActionType.REENGAGE_NOTIFICATION, "High churn risk detected"),
        (ActionType.OFFER_DISCOUNTED_SUBSCRIPTION, "Retention offer"),
        (ActionType.PERSONALIZED_PLAYLIST, "Curated content to re-engage")
    ],
    TriggerType.CONTENT_COMPLETED: [
        (ActionType.RECOMMEND_MOVIE, "Since you enjoyed that, try this"),
        (ActionType.PERSONALIZED_PLAYLIST, "More like this")
    ],
    TriggerType.SEARCH_FAILED: [
        (ActionType.RECOMMEND_MOVIE, "Couldn't find that, but you might like these"),
        (ActionType.REENGAGE_NOTIFICATION, "Search assistance")
    ],
    TriggerType.SUBSCRIPTION_EXPIRING: [
        (ActionType.OFFER_BUNDLE_UPGRADE, "Upgrade before expiry"),
        (ActionType.OFFER_DISCOUNTED_SUBSCRIPTION, "Renewal discount")
    ],
    TriggerType.NEW_CONTENT_AVAILABLE: [
        (ActionType.RECOMMEND_PREMIUM_CONTENT, "New premium content"),
        (ActionType.REENGAGE_NOTIFICATION, "New arrivals")
    ]
}


def create_default_action_engine(recommender=None) -> BusinessActionEngine:
    engine = BusinessActionEngine()
    engine.register_executor(RecommendMovieExecutor(recommender))
    engine.register_executor(BundleUpgradeExecutor())
    engine.register_executor(DiscountedSubscriptionExecutor())
    engine.register_executor(ReengageNotificationExecutor())
    engine.register_executor(PersonalizedPlaylistExecutor(recommender))
    return engine