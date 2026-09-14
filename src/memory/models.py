from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import time


class EventType(str, Enum):
    MOVIE_CLICKED = "movie_clicked"
    MOVIE_VIEWED = "movie_viewed"
    ADDED_TO_WATCHLIST = "added_to_watchlist"
    MARKED_WATCHED = "marked_watched"
    RATED = "rated"


class FeatureType(str, Enum):
    GENRE = "genre"
    KEYWORD = "keyword"
    DIRECTOR = "director"
    CAST = "cast"
    DECADE = "decade"
    LANGUAGE = "language"


@dataclass
class WatchEvent:
    user_id: str
    movie_id: int
    event_type: EventType
    signal_weight: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None


@dataclass
class RatingEvent:
    user_id: str
    movie_id: int
    rating: float  # 0.0 to 5.0
    signal_target: float  # mapped value
    timestamp: float = field(default_factory=time.time)
    review_text: Optional[str] = None
    id: Optional[int] = None


@dataclass
class FeaturePreference:
    feature_type: FeatureType
    feature_value: str
    score: float  # [0.0, 1.0]
    confidence: float  # [0.0, 1.0]
    interaction_count: int
    last_updated: float = field(default_factory=time.time)


@dataclass
class NegativePreference:
    feature_type: FeatureType
    feature_value: str
    penalty_weight: float  # [0.0, 1.0]
    interaction_count: int
    last_updated: float = field(default_factory=time.time)


@dataclass
class UserProfileMemory:
    user_id: str
    total_interactions: int = 0
    short_term_preferences: Dict[str, Dict[str, float]] = field(default_factory=dict)
    long_term_preferences: Dict[str, Dict[str, FeaturePreference]] = field(default_factory=dict)
    negative_preferences: Dict[str, Dict[str, NegativePreference]] = field(default_factory=dict)
    subscriptions: List[str] = field(default_factory=lambda: ["Netflix", "Prime Video", "Disney+"])
    updated_at: float = field(default_factory=time.time)
    version: int = 1


