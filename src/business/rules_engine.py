import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import copy


class ContentRating(str, Enum):
    G = "G"
    PG = "PG"
    PG13 = "PG-13"
    R = "R"
    NC17 = "NC-17"
    UNRATED = "UNRATED"


class ActionType(str, Enum):
    RECOMMEND_MOVIE = "recommend_movie"
    RECOMMEND_SERIES = "recommend_series"
    OFFER_BUNDLE_UPGRADE = "offer_bundle_upgrade"
    OFFER_DISCOUNTED_SUBSCRIPTION = "offer_discounted_subscription"
    RECOMMEND_PREMIUM_CONTENT = "recommend_premium_content"
    REENGAGE_NOTIFICATION = "reengage_notification"
    CONTENT_WARNING = "content_warning"


@dataclass
class BusinessRule:
    rule_id: str
    name: str
    description: str
    condition: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: int = 0
    enabled: bool = True
    partner_id: Optional[str] = None

    def matches(self, context: Dict[str, Any]) -> bool:
        for key, expected in self.condition.items():
            if key not in context:
                return False
            actual = context[key]
            if isinstance(expected, dict):
                if "operator" not in expected:
                    return False
                op = expected["operator"]
                val = expected["value"]
                if op == "eq" and actual != val:
                    return False
                elif op == "neq" and actual == val:
                    return False
                elif op == "gt" and not (actual > val):
                    return False
                elif op == "gte" and not (actual >= val):
                    return False
                elif op == "lt" and not (actual < val):
                    return False
                elif op == "lte" and not (actual <= val):
                    return False
                elif op == "in" and actual not in val:
                    return False
                elif op == "not_in" and actual in val:
                    return False
                elif op == "contains" and val not in str(actual):
                    return False
            elif actual != expected:
                return False
        return True


@dataclass
class PartnerConfig:
    partner_id: str
    name: str
    business_rules: List[BusinessRule] = field(default_factory=list)
    content_filters: Dict[str, Any] = field(default_factory=dict)
    ranking_weights: Dict[str, float] = field(default_factory=dict)
    license_regions: List[str] = field(default_factory=lambda: ["US"])
    partner_content_boost: Dict[str, float] = field(default_factory=dict)
    competitor_exclude: List[str] = field(default_factory=list)
    max_runtime_minutes: Optional[int] = None
    allowed_content_ratings: List[ContentRating] = field(default_factory=lambda: [ContentRating.G, ContentRating.PG, ContentRating.PG13])
    offline_required: bool = False
    session_limits: Dict[str, Any] = field(default_factory=dict)


class RulesEngine:
    def __init__(self):
        self.partner_configs: Dict[str, PartnerConfig] = {}
        self.global_rules: List[BusinessRule] = []

    def register_partner(self, config: PartnerConfig):
        self.partner_configs[config.partner_id] = config

    def get_partner_config(self, partner_id: str) -> Optional[PartnerConfig]:
        return self.partner_configs.get(partner_id)

    def add_global_rule(self, rule: BusinessRule):
        self.global_rules.append(rule)

    def evaluate(self, partner_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        config = self.get_partner_config(partner_id)
        if not config:
            return {"filters": {}, "weight_adjustments": {}, "actions": [], "warnings": []}

        all_rules = self.global_rules + config.business_rules
        all_rules.sort(key=lambda r: r.priority, reverse=True)

        result = {
            "filters": copy.deepcopy(config.content_filters),
            "weight_adjustments": copy.deepcopy(config.ranking_weights),
            "actions": [],
            "warnings": []
        }

        for rule in all_rules:
            if not rule.enabled:
                continue
            if rule.matches(context):
                for action in rule.actions:
                    action_type = action.get("type")
                    if action_type == "add_filter":
                        result["filters"].update(action.get("filters", {}))
                    elif action_type == "adjust_weight":
                        result["weight_adjustments"].update(action.get("weights", {}))
                    elif action_type == "trigger_action":
                        result["actions"].append(action.get("action"))
                    elif action_type == "warning":
                        result["warnings"].append(action.get("message"))

        return result

    def apply_content_filters(self, movies: List[Dict], partner_id: str, context: Dict) -> List[Dict]:
        config = self.get_partner_config(partner_id)
        if not config:
            return movies

        eval_result = self.evaluate(partner_id, context)
        filters = eval_result["filters"]

        filtered = []
        for movie in movies:
            if self._passes_filters(movie, filters, config, context):
                filtered.append(movie)

        return filtered

    def _passes_filters(self, movie: Dict, filters: Dict, config: PartnerConfig, context: Dict) -> bool:
        # Missing metadata = pass. Filters only exclude on positive evidence,
        # so movies without runtime/rating/offline info are never dropped
        # for lack of data (loose filtering).
        if config.max_runtime_minutes:
            runtime = movie.get("runtime")
            if runtime and runtime > config.max_runtime_minutes:
                return False

        if config.allowed_content_ratings:
            rating = movie.get("content_rating")
            if rating and rating not in [r.value for r in config.allowed_content_ratings]:
                return False

        if config.offline_required:
            if movie.get("offline_available") is False:
                return False

        if config.competitor_exclude:
            providers = movie.get("watch_providers", {})
            for comp in config.competitor_exclude:
                if comp.lower() in str(providers).lower():
                    return False

        if "license_regions" in filters:
            region = context.get("region", "US")
            if region not in filters["license_regions"]:
                return False

        if "exclude_genres" in filters:
            movie_genres = movie.get("genres", [])
            if any(g in filters["exclude_genres"] for g in movie_genres):
                return False

        if "required_genres" in filters:
            movie_genres = movie.get("genres", [])
            if not any(g in filters["required_genres"] for g in movie_genres):
                return False

        if "min_rating" in filters:
            if movie.get("vote_average", 0) < filters["min_rating"]:
                return False

        return True

    def apply_weight_adjustments(self, base_weights: Dict[str, float], partner_id: str, context: Dict) -> Dict[str, float]:
        config = self.get_partner_config(partner_id)
        if not config:
            return base_weights

        eval_result = self.evaluate(partner_id, context)
        adjusted = copy.deepcopy(base_weights)
        adjusted.update(eval_result.get("weight_adjustments", {}))

        for provider, boost in config.partner_content_boost.items():
            if provider in str(context).lower():
                for k in adjusted:
                    adjusted[k] *= boost

        return adjusted

    def get_triggered_actions(self, partner_id: str, context: Dict) -> List[Dict]:
        config = self.get_partner_config(partner_id)
        if not config:
            return []

        eval_result = self.evaluate(partner_id, context)
        return eval_result.get("actions", [])

    def load_from_file(self, filepath: str):
        with open(filepath) as f:
            data = json.load(f)

        for partner_data in data.get("partners", []):
            rules = []
            for rule_data in partner_data.get("business_rules", []):
                rules.append(BusinessRule(**rule_data))

            config = PartnerConfig(
                partner_id=partner_data["partner_id"],
                name=partner_data["name"],
                business_rules=rules,
                content_filters=partner_data.get("content_filters", {}),
                ranking_weights=partner_data.get("ranking_weights", {}),
                license_regions=partner_data.get("license_regions", ["US"]),
                partner_content_boost=partner_data.get("partner_content_boost", {}),
                competitor_exclude=partner_data.get("competitor_exclude", []),
                max_runtime_minutes=partner_data.get("max_runtime_minutes"),
                allowed_content_ratings=[ContentRating(r) for r in partner_data.get("allowed_content_ratings", ["G", "PG", "PG-13"])],
                offline_required=partner_data.get("offline_required", False),
                session_limits=partner_data.get("session_limits", {})
            )
            self.register_partner(config)

        for rule_data in data.get("global_rules", []):
            self.add_global_rule(BusinessRule(**rule_data))


DEFAULT_PARTNER_CONFIGS = {
    "airline_default": PartnerConfig(
        partner_id="airline_default",
        name="Airline Default Config",
        content_filters={},
        ranking_weights={"imdb_weight": 0.25, "tfidf_weight": 0.15, "watch_history_weight": 0.4, "recency_weight": 0.2},
        license_regions=["US", "EU", "ASIA"],
        max_runtime_minutes=180,
        allowed_content_ratings=[ContentRating.G, ContentRating.PG, ContentRating.PG13, ContentRating.R],
        offline_required=True,
        partner_content_boost={"hbo": 1.1, "disney": 1.1},
        session_limits={"max_recommendations": 20}
    ),
    "hotel_default": PartnerConfig(
        partner_id="hotel_default",
        name="Hotel Default Config",
        content_filters={},
        ranking_weights={"imdb_weight": 0.3, "tfidf_weight": 0.2, "watch_history_weight": 0.35, "family_friendly_weight": 0.15},
        license_regions=["US"],
        max_runtime_minutes=150,
        allowed_content_ratings=[ContentRating.G, ContentRating.PG, ContentRating.PG13],
        partner_content_boost={"disney": 1.2, "netflix": 1.0},
        competitor_exclude=[],
        session_limits={"max_recommendations": 15}
    ),
    "streaming_default": PartnerConfig(
        partner_id="streaming_default",
        name="Streaming Platform Default",
        content_filters={},
        ranking_weights={"imdb_weight": 0.2, "tfidf_weight": 0.25, "watch_history_weight": 0.45, "discovery_weight": 0.1},
        license_regions=["US"],
        allowed_content_ratings=[ContentRating.G, ContentRating.PG, ContentRating.PG13, ContentRating.R, ContentRating.NC17],
        partner_content_boost={},
        competitor_exclude=[],
        session_limits={"max_recommendations": 50}
    )
}


def create_default_rules_engine() -> RulesEngine:
    engine = RulesEngine()
    for config in DEFAULT_PARTNER_CONFIGS.values():
        engine.register_partner(config)

    engine.add_global_rule(BusinessRule(
        rule_id="family_mode",
        name="Family Mode Filter",
        description="Enable family-friendly filtering when children present",
        condition={"context.party_type": {"operator": "eq", "value": "family"}},
        actions=[
            {"type": "add_filter", "filters": {"exclude_genres": ["Horror", "Thriller", "Crime"], "required_content_ratings": ["G", "PG", "PG-13"]}}
        ],
        priority=100
    ))

    engine.add_global_rule(BusinessRule(
        rule_id="short_flight",
        name="Short Flight Duration Limit",
        description="Limit movie length for short flights",
        condition={"context.flight_duration_minutes": {"operator": "lt", "value": 120}},
        actions=[
            {"type": "add_filter", "filters": {"max_runtime_minutes": 100}}
        ],
        priority=90
    ))

    engine.add_global_rule(BusinessRule(
        rule_id="premium_user_boost",
        name="Premium Content Boost for Premium Users",
        description="Boost premium content for premium tier users",
        condition={"context.subscription_tier": {"operator": "in", "value": ["premium", "vip"]}},
        actions=[
            {"type": "adjust_weight", "weights": {"premium_content_weight": 1.3}}
        ],
        priority=50
    ))

    return engine