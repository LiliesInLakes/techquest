import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import math


class ChurnRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendedAction(str, Enum):
    NO_ACTION = "no_action"
    REENGAGE_PUSH = "reengage_push"
    REENGAGE_EMAIL = "reengage_email"
    PERSONALIZED_RECOMMENDATIONS = "personalized_recommendations"
    CONTENT_DISCOVERY = "content_discovery"
    OFFER_DISCOUNT = "offer_discount"
    OFFER_FREE_TRIAL = "offer_free_trial"
    UPGRADE_PROMPT = "upgrade_prompt"
    SUPPORT_OUTREACH = "support_outreach"
    WINBACK_CAMPAIGN = "winback_campaign"


@dataclass
class ChurnFeatures:
    user_id: str
    partner_id: str

    days_since_last_session: float = 999
    days_since_last_watch: float = 999
    sessions_last_7_days: int = 0
    sessions_last_30_days: int = 0
    watch_time_last_7_days: int = 0
    watch_time_last_30_days: int = 0
    avg_session_length_minutes: float = 0
    completion_rate: float = 1.0
    skip_rate: float = 0.0
    rewatch_rate: float = 0.0
    discovery_rate: float = 0.0

    genre_diversity: float = 0.0
    genre_concentration_hhi: float = 0.0
    unique_genres_watched: int = 0

    search_count_last_30_days: int = 0
    search_failure_rate: float = 0.0
    recommendation_impression_count: int = 0
    recommendation_click_rate: float = 0.0

    subscription_tier: str = "basic"
    subscription_age_days: int = 0
    payment_failures_last_90_days: int = 0
    support_tickets_last_30_days: int = 0

    competitor_app_detected: bool = False
    device_count: int = 1
    platform_engagement_score: float = 0.0

    computed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ChurnPrediction:
    user_id: str
    partner_id: str
    risk_score: float
    risk_level: ChurnRiskLevel
    top_risk_factors: List[Dict[str, Any]]
    recommended_actions: List[RecommendedAction]
    confidence: float
    feature_contributions: Dict[str, float]
    predicted_churn_date: Optional[str] = None
    estimated_ltv_loss: float = 0.0


class ChurnRiskModel:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.weights = self.config.get("weights", {})
        self.thresholds = self.config.get("thresholds", {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.85
        })

    def _default_config(self) -> Dict[str, Any]:
        return {
            "weights": {
                "recency": 0.25,
                "frequency": 0.20,
                "engagement": 0.15,
                "satisfaction": 0.15,
                "diversity": 0.10,
                "subscription": 0.10,
                "external": 0.05
            },
            "thresholds": {
                "low": 0.3,
                "medium": 0.5,
                "high": 0.7,
                "critical": 0.85
            },
            "feature_normalization": {
                "days_since_last_session": {"type": "log", "max": 90},
                "days_since_last_watch": {"type": "log", "max": 90},
                "sessions_last_7_days": {"type": "linear", "max": 14},
                "sessions_last_30_days": {"type": "linear", "max": 60},
                "watch_time_last_7_days": {"type": "log", "max": 600},
                "avg_session_length_minutes": {"type": "linear", "max": 180},
                "completion_rate": {"type": "inverse", "max": 1.0},
                "skip_rate": {"type": "linear", "max": 1.0},
                "genre_concentration_hhi": {"type": "linear", "max": 1.0},
                "search_failure_rate": {"type": "linear", "max": 1.0},
                "recommendation_click_rate": {"type": "inverse", "max": 1.0},
                "subscription_age_days": {"type": "inverse_log", "max": 1000},
                "payment_failures_last_90_days": {"type": "linear", "max": 5},
                "support_tickets_last_30_days": {"type": "linear", "max": 10}
            }
        }

    def compute_features(self, user_data: Dict[str, Any]) -> ChurnFeatures:
        return ChurnFeatures(
            user_id=user_data.get("user_id", ""),
            partner_id=user_data.get("partner_id", ""),
            days_since_last_session=user_data.get("days_since_last_session", 999),
            days_since_last_watch=user_data.get("days_since_last_watch", 999),
            sessions_last_7_days=user_data.get("sessions_last_7_days", 0),
            sessions_last_30_days=user_data.get("sessions_last_30_days", 0),
            watch_time_last_7_days=user_data.get("watch_time_last_7_days", 0),
            watch_time_last_30_days=user_data.get("watch_time_last_30_days", 0),
            avg_session_length_minutes=user_data.get("avg_session_length_minutes", 0),
            completion_rate=user_data.get("completion_rate", 1.0),
            skip_rate=user_data.get("skip_rate", 0.0),
            rewatch_rate=user_data.get("rewatch_rate", 0.0),
            discovery_rate=user_data.get("discovery_rate", 0.0),
            genre_diversity=user_data.get("genre_diversity", 0.0),
            genre_concentration_hhi=user_data.get("genre_concentration_hhi", 0.0),
            unique_genres_watched=user_data.get("unique_genres_watched", 0),
            search_count_last_30_days=user_data.get("search_count_last_30_days", 0),
            search_failure_rate=user_data.get("search_failure_rate", 0.0),
            recommendation_impression_count=user_data.get("recommendation_impression_count", 0),
            recommendation_click_rate=user_data.get("recommendation_click_rate", 0.0),
            subscription_tier=user_data.get("subscription_tier", "basic"),
            subscription_age_days=user_data.get("subscription_age_days", 0),
            payment_failures_last_90_days=user_data.get("payment_failures_last_90_days", 0),
            support_tickets_last_30_days=user_data.get("support_tickets_last_30_days", 0),
            competitor_app_detected=user_data.get("competitor_app_detected", False),
            device_count=user_data.get("device_count", 1),
            platform_engagement_score=user_data.get("platform_engagement_score", 0.0)
        )

    def _normalize_feature(self, value: float, norm_config: Dict) -> float:
        norm_type = norm_config.get("type", "linear")
        max_val = norm_config.get("max", 1.0)

        if max_val == 0:
            return 0.0

        if norm_type == "linear":
            return min(value / max_val, 1.0)
        elif norm_type == "log":
            return min(math.log1p(value) / math.log1p(max_val), 1.0)
        elif norm_type == "inverse":
            return min((max_val - value) / max_val, 1.0)
        elif norm_type == "inverse_log":
            return min((math.log1p(max_val) - math.log1p(value)) / math.log1p(max_val), 1.0)
        else:
            return min(value / max_val, 1.0)

    def _compute_category_score(self, features: ChurnFeatures, category: str) -> Tuple[float, Dict[str, float]]:
        norm_config = self.config["feature_normalization"]
        contributions = {}

        if category == "recency":
            vals = {
                "days_since_last_session": self._normalize_feature(features.days_since_last_session, norm_config["days_since_last_session"]),
                "days_since_last_watch": self._normalize_feature(features.days_since_last_watch, norm_config["days_since_last_watch"])
            }
            score = np.mean(list(vals.values()))
            contributions = {f"recency_{k}": v * self.weights.get("recency", 0.25) / len(vals) for k, v in vals.items()}

        elif category == "frequency":
            vals = {
                "sessions_last_7_days": 1 - self._normalize_feature(features.sessions_last_7_days, norm_config["sessions_last_7_days"]),
                "sessions_last_30_days": 1 - self._normalize_feature(features.sessions_last_30_days, norm_config["sessions_last_30_days"]),
                "watch_time_last_7_days": 1 - self._normalize_feature(features.watch_time_last_7_days, norm_config["watch_time_last_7_days"])
            }
            score = np.mean(list(vals.values()))
            contributions = {f"frequency_{k}": v * self.weights.get("frequency", 0.20) / len(vals) for k, v in vals.items()}

        elif category == "engagement":
            vals = {
                "avg_session_length": 1 - self._normalize_feature(features.avg_session_length_minutes, norm_config["avg_session_length_minutes"]),
                "completion_rate": self._normalize_feature(features.completion_rate, norm_config["completion_rate"]),
                "skip_rate": self._normalize_feature(features.skip_rate, norm_config["skip_rate"]),
                "rewatch_rate": 1 - self._normalize_feature(features.rewatch_rate, {"type": "linear", "max": 0.5}),
                "discovery_rate": 1 - self._normalize_feature(features.discovery_rate, {"type": "linear", "max": 0.5})
            }
            score = np.mean(list(vals.values()))
            contributions = {f"engagement_{k}": v * self.weights.get("engagement", 0.15) / len(vals) for k, v in vals.items()}

        elif category == "satisfaction":
            vals = {
                "search_failure_rate": self._normalize_feature(features.search_failure_rate, norm_config["search_failure_rate"]),
                "recommendation_click_rate": self._normalize_feature(features.recommendation_click_rate, norm_config["recommendation_click_rate"]),
                "support_tickets": self._normalize_feature(features.support_tickets_last_30_days, norm_config["support_tickets_last_30_days"])
            }
            score = np.mean(list(vals.values()))
            contributions = {f"satisfaction_{k}": v * self.weights.get("satisfaction", 0.15) / len(vals) for k, v in vals.items()}

        elif category == "diversity":
            vals = {
                "genre_concentration": self._normalize_feature(features.genre_concentration_hhi, norm_config["genre_concentration_hhi"]),
                "unique_genres": 1 - min(features.unique_genres_watched / 20, 1.0)
            }
            score = np.mean(list(vals.values()))
            contributions = {f"diversity_{k}": v * self.weights.get("diversity", 0.10) / len(vals) for k, v in vals.items()}

        elif category == "subscription":
            tier_risk = {"basic": 0.3, "standard": 0.2, "premium": 0.1, "vip": 0.05}.get(features.subscription_tier, 0.3)
            vals = {
                "tier_risk": tier_risk,
                "subscription_age": self._normalize_feature(features.subscription_age_days, norm_config["subscription_age_days"]),
                "payment_failures": self._normalize_feature(features.payment_failures_last_90_days, norm_config["payment_failures_last_90_days"])
            }
            score = np.mean(list(vals.values()))
            contributions = {f"subscription_{k}": v * self.weights.get("subscription", 0.10) / len(vals) for k, v in vals.items()}

        elif category == "external":
            vals = {
                "competitor_app": 1.0 if features.competitor_app_detected else 0.0,
                "device_count": 1 - min(features.device_count / 5, 1.0),
                "platform_engagement": 1 - self._normalize_feature(features.platform_engagement_score, {"type": "linear", "max": 1.0})
            }
            score = np.mean(list(vals.values()))
            contributions = {f"external_{k}": v * self.weights.get("external", 0.05) / len(vals) for k, v in vals.items()}

        else:
            score = 0.0

        return score, contributions

    def predict(self, features: ChurnFeatures) -> ChurnPrediction:
        category_scores = {}
        all_contributions = {}

        for category in ["recency", "frequency", "engagement", "satisfaction", "diversity", "subscription", "external"]:
            score, contributions = self._compute_category_score(features, category)
            category_scores[category] = score
            all_contributions.update(contributions)

        risk_score = sum(category_scores.get(cat, 0) * self.weights.get(cat, 0) for cat in category_scores)

        if risk_score >= self.thresholds["critical"]:
            risk_level = ChurnRiskLevel.CRITICAL
        elif risk_score >= self.thresholds["high"]:
            risk_level = ChurnRiskLevel.HIGH
        elif risk_score >= self.thresholds["medium"]:
            risk_level = ChurnRiskLevel.MEDIUM
        else:
            risk_level = ChurnRiskLevel.LOW

        top_factors = sorted(all_contributions.items(), key=lambda x: -x[1])[:5]
        top_risk_factors = [{"factor": k, "contribution": round(v, 4)} for k, v in top_factors]

        recommended_actions = self._determine_actions(risk_level, features, category_scores)

        confidence = min(0.9, 0.5 + abs(risk_score - 0.5))

        predicted_date = None
        if risk_level in [ChurnRiskLevel.HIGH, ChurnRiskLevel.CRITICAL]:
            days_to_churn = max(7, int(30 * (1 - risk_score)))
            predicted_date = (datetime.utcnow() + timedelta(days=days_to_churn)).isoformat()

        ltv_loss = self._estimate_ltv_loss(features, risk_score)

        return ChurnPrediction(
            user_id=features.user_id,
            partner_id=features.partner_id,
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            top_risk_factors=top_risk_factors,
            recommended_actions=recommended_actions,
            confidence=round(confidence, 4),
            feature_contributions=all_contributions,
            predicted_churn_date=predicted_date,
            estimated_ltv_loss=round(ltv_loss, 2)
        )

    def _determine_actions(
        self,
        risk_level: ChurnRiskLevel,
        features: ChurnFeatures,
        category_scores: Dict[str, float]
    ) -> List[RecommendedAction]:
        actions = []

        if risk_level == ChurnRiskLevel.LOW:
            return [RecommendedAction.NO_ACTION]

        if category_scores.get("recency", 0) > 0.6:
            actions.append(RecommendedAction.REENGAGE_PUSH)

        if category_scores.get("frequency", 0) > 0.5:
            actions.append(RecommendedAction.PERSONALIZED_RECOMMENDATIONS)

        if category_scores.get("engagement", 0) > 0.5:
            actions.append(RecommendedAction.CONTENT_DISCOVERY)

        if category_scores.get("satisfaction", 0) > 0.5:
            actions.append(RecommendedAction.SUPPORT_OUTREACH)

        if category_scores.get("subscription", 0) > 0.5:
            if features.subscription_tier == "basic":
                actions.append(RecommendedAction.OFFER_FREE_TRIAL)
            else:
                actions.append(RecommendedAction.OFFER_DISCOUNT)

        if category_scores.get("external", 0) > 0.5:
            actions.append(RecommendedAction.WINBACK_CAMPAIGN)

        if risk_level == ChurnRiskLevel.CRITICAL:
            actions.append(RecommendedAction.UPGRADE_PROMPT)

        return list(dict.fromkeys(actions))[:3]

    def _estimate_ltv_loss(self, features: ChurnFeatures, risk_score: float) -> float:
        monthly_value = {"basic": 10, "standard": 15, "premium": 25, "vip": 50}.get(features.subscription_tier, 10)
        expected_months = max(1, 12 * (1 - risk_score))
        return monthly_value * expected_months * risk_score


class ChurnRiskService:
    def __init__(self, model: ChurnRiskModel = None):
        self.model = model or ChurnRiskModel()
        self.predictions_cache: Dict[str, ChurnPrediction] = {}

    def assess_user(self, user_data: Dict[str, Any]) -> ChurnPrediction:
        features = self.model.compute_features(user_data)
        prediction = self.model.predict(features)
        cache_key = f"{features.partner_id}:{features.user_id}"
        self.predictions_cache[cache_key] = prediction
        return prediction

    def get_cached_prediction(self, partner_id: str, user_id: str) -> Optional[ChurnPrediction]:
        return self.predictions_cache.get(f"{partner_id}:{user_id}")

    def get_high_risk_users(self, partner_id: str, threshold: float = 0.7) -> List[ChurnPrediction]:
        return [
            p for k, p in self.predictions_cache.items()
            if k.startswith(f"{partner_id}:") and p.risk_score >= threshold
        ]

    def batch_assess(self, users_data: List[Dict[str, Any]]) -> List[ChurnPrediction]:
        return [self.assess_user(u) for u in users_data]


def main():
    model = ChurnRiskModel()

    test_user = {
        "user_id": "user_123",
        "partner_id": "streaming_default",
        "days_since_last_session": 45,
        "days_since_last_watch": 30,
        "sessions_last_7_days": 0,
        "sessions_last_30_days": 2,
        "watch_time_last_7_days": 0,
        "watch_time_last_30_days": 120,
        "avg_session_length_minutes": 45,
        "completion_rate": 0.4,
        "skip_rate": 0.6,
        "rewatch_rate": 0.1,
        "discovery_rate": 0.05,
        "genre_diversity": 0.2,
        "genre_concentration_hhi": 0.8,
        "unique_genres_watched": 2,
        "search_count_last_30_days": 10,
        "search_failure_rate": 0.7,
        "recommendation_impression_count": 50,
        "recommendation_click_rate": 0.05,
        "subscription_tier": "basic",
        "subscription_age_days": 180,
        "payment_failures_last_90_days": 1,
        "support_tickets_last_30_days": 2,
        "competitor_app_detected": True,
        "device_count": 1,
        "platform_engagement_score": 0.2
    }

    prediction = model.predict(model.compute_features(test_user))
    print(f"Risk Score: {prediction.risk_score}")
    print(f"Risk Level: {prediction.risk_level}")
    print(f"Top Factors: {prediction.top_risk_factors}")
    print(f"Actions: {[a.value for a in prediction.recommended_actions]}")
    print(f"LTV Loss: ${prediction.estimated_ltv_loss}")


if __name__ == "__main__":
    main()