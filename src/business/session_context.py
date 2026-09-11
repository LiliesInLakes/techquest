import uuid
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import threading


class PartyType(str, Enum):
    SOLO = "solo"
    COUPLE = "couple"
    FAMILY = "family"
    FRIENDS = "friends"
    BUSINESS = "business"


class DeviceType(str, Enum):
    SEATBACK = "seatback"
    MOBILE = "mobile"
    TABLET = "tablet"
    LAPTOP = "laptop"
    TV = "tv"
    VR = "vr"


class SessionState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    EXPIRED = "expired"


@dataclass
class SessionContext:
    session_id: str
    user_id: str
    partner_id: str
    device_type: DeviceType
    party_type: PartyType
    party_size: int = 1
    region: str = "US"
    language: str = "en"
    subscriptions: List[str] = field(default_factory=list)
    flight_duration_minutes: Optional[int] = None
    flight_number: Optional[str] = None
    hotel_room_type: Optional[str] = None
    check_in_date: Optional[str] = None
    check_out_date: Optional[str] = None
    subscription_tier: str = "basic"
    time_of_day: str = "evening"
    day_of_week: str = "weekday"
    timezone: str = "UTC"
    offline_mode: bool = False
    bandwidth_mbps: Optional[float] = None
    screen_size_inches: Optional[float] = None
    audio_output: str = "headphones"
    accessibility: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    state: SessionState = SessionState.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["device_type"] = self.device_type.value
        d["party_type"] = self.party_type.value
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionContext":
        data = data.copy()
        data["device_type"] = DeviceType(data.get("device_type", "mobile"))
        data["party_type"] = PartyType(data.get("party_type", "solo"))
        data["state"] = SessionState(data.get("state", "active"))
        return cls(**data)


@dataclass
class UserSessionProfile:
    user_id: str
    partner_id: str
    total_sessions: int = 0
    total_watch_time_minutes: int = 0
    avg_session_length_minutes: float = 0.0
    preferred_genres: Dict[str, float] = field(default_factory=dict)
    preferred_decades: Dict[str, float] = field(default_factory=dict)
    preferred_runtime_range: tuple = (90, 120)
    preferred_party_types: Dict[str, int] = field(default_factory=dict)
    preferred_devices: Dict[str, int] = field(default_factory=dict)
    time_of_day_distribution: Dict[str, int] = field(default_factory=dict)
    day_of_week_distribution: Dict[str, int] = field(default_factory=dict)
    completion_rate: float = 1.0
    skip_rate: float = 0.0
    rewatch_rate: float = 0.0
    discovery_rate: float = 0.0
    last_active: Optional[str] = None
    churn_risk_score: float = 0.0
    lifetime_value_estimate: float = 0.0


class SessionManager:
    def __init__(self, ttl_hours: int = 24):
        self.sessions: Dict[str, SessionContext] = {}
        self.user_profiles: Dict[str, UserSessionProfile] = {}
        self.ttl_seconds = ttl_hours * 3600
        self._lock = threading.RLock()

    def create_session(
        self,
        user_id: str,
        partner_id: str,
        context_overrides: Dict[str, Any] = None
    ) -> SessionContext:
        with self._lock:
            session_id = str(uuid.uuid4())
            now = datetime.utcnow()

            context = SessionContext(
                session_id=session_id,
                user_id=user_id,
                partner_id=partner_id,
                device_type=DeviceType(context_overrides.get("device_type", "mobile")) if context_overrides else DeviceType.MOBILE,
                party_type=PartyType(context_overrides.get("party_type", "solo")) if context_overrides else PartyType.SOLO,
                party_size=context_overrides.get("party_size", 1) if context_overrides else 1,
                region=context_overrides.get("region", "US") if context_overrides else "US",
                language=context_overrides.get("language", "en") if context_overrides else "en",
                subscriptions=context_overrides.get("subscriptions", []) if context_overrides else [],
                flight_duration_minutes=context_overrides.get("flight_duration_minutes") if context_overrides else None,
                flight_number=context_overrides.get("flight_number") if context_overrides else None,
                hotel_room_type=context_overrides.get("hotel_room_type") if context_overrides else None,
                check_in_date=context_overrides.get("check_in_date") if context_overrides else None,
                check_out_date=context_overrides.get("check_out_date") if context_overrides else None,
                subscription_tier=context_overrides.get("subscription_tier", "basic") if context_overrides else "basic",
                time_of_day=self._get_time_of_day(now),
                day_of_week=self._get_day_of_week(now),
                timezone=context_overrides.get("timezone", "UTC") if context_overrides else "UTC",
                offline_mode=context_overrides.get("offline_mode", False) if context_overrides else False,
                bandwidth_mbps=context_overrides.get("bandwidth_mbps") if context_overrides else None,
                screen_size_inches=context_overrides.get("screen_size_inches") if context_overrides else None,
                audio_output=context_overrides.get("audio_output", "headphones") if context_overrides else "headphones",
                accessibility=context_overrides.get("accessibility", {}) if context_overrides else {},
                metadata=context_overrides.get("metadata", {}) if context_overrides else {}
            )

            self.sessions[session_id] = context
            self._ensure_user_profile(user_id, partner_id)
            return context

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session and self._is_expired(session):
                session.state = SessionState.EXPIRED
            return session

    def update_session(self, session_id: str, updates: Dict[str, Any]) -> Optional[SessionContext]:
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return None

            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)

            session.updated_at = datetime.utcnow().isoformat()
            return session

    def end_session(self, session_id: str, watch_events: List[Dict] = None) -> Optional[SessionContext]:
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return None

            session.state = SessionState.ENDED
            session.updated_at = datetime.utcnow().isoformat()

            if watch_events:
                self._update_user_profile(session, watch_events)

            return session

    def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[SessionContext]:
        with self._lock:
            sessions = [s for s in self.sessions.values() if s.user_id == user_id]
            if active_only:
                sessions = [s for s in sessions if s.state == SessionState.ACTIVE and not self._is_expired(s)]
            return sessions

    def get_user_profile(self, user_id: str, partner_id: str) -> UserSessionProfile:
        with self._lock:
            key = f"{partner_id}:{user_id}"
            return self.user_profiles.get(key, UserSessionProfile(user_id=user_id, partner_id=partner_id))

    def _ensure_user_profile(self, user_id: str, partner_id: str):
        key = f"{partner_id}:{user_id}"
        if key not in self.user_profiles:
            self.user_profiles[key] = UserSessionProfile(user_id=user_id, partner_id=partner_id)

    def _update_user_profile(self, session: SessionContext, watch_events: List[Dict]):
        key = f"{session.partner_id}:{session.user_id}"
        profile = self.user_profiles.get(key)
        if not profile:
            return

        profile.total_sessions += 1
        profile.last_active = datetime.utcnow().isoformat()

        total_watch_time = 0
        completed = 0
        skipped = 0
        rewatched = 0
        new_content = 0

        for event in watch_events:
            watch_time = event.get("watch_time_minutes", 0)
            total_watch_time += watch_time

            if event.get("completed", False):
                completed += 1
            if event.get("skipped", False):
                skipped += 1
            if event.get("rewatch", False):
                rewatched += 1
            if event.get("first_watch", True):
                new_content += 1

            genres = event.get("genres", [])
            for g in genres:
                profile.preferred_genres[g] = profile.preferred_genres.get(g, 0) + 1

            decade = event.get("decade")
            if decade:
                profile.preferred_decades[decade] = profile.preferred_decades.get(decade, 0) + 1

            runtime = event.get("runtime_minutes")
            if runtime:
                lo, hi = profile.preferred_runtime_range
                profile.preferred_runtime_range = (min(lo, runtime), max(hi, runtime))

        profile.total_watch_time_minutes += total_watch_time
        if profile.total_sessions > 0:
            profile.avg_session_length_minutes = profile.total_watch_time_minutes / profile.total_sessions

        total_events = len(watch_events)
        if total_events > 0:
            profile.completion_rate = completed / total_events
            profile.skip_rate = skipped / total_events
            profile.rewatch_rate = rewatched / total_events
            profile.discovery_rate = new_content / total_events

        profile.preferred_party_types[session.party_type.value] = profile.preferred_party_types.get(session.party_type.value, 0) + 1
        profile.preferred_devices[session.device_type.value] = profile.preferred_devices.get(session.device_type.value, 0) + 1
        profile.time_of_day_distribution[session.time_of_day] = profile.time_of_day_distribution.get(session.time_of_day, 0) + 1
        profile.day_of_week_distribution[session.day_of_week] = profile.day_of_week_distribution.get(session.day_of_week, 0) + 1

    def _is_expired(self, session: SessionContext) -> bool:
        try:
            updated = datetime.fromisoformat(session.updated_at.replace('Z', '+00:00'))
            return (datetime.utcnow() - updated).total_seconds() > self.ttl_seconds
        except:
            return False

    def _get_time_of_day(self, dt: datetime) -> str:
        hour = dt.hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"

    def _get_day_of_week(self, dt: datetime) -> str:
        return "weekend" if dt.weekday() >= 5 else "weekday"

    def cleanup_expired(self):
        with self._lock:
            expired = [sid for sid, s in self.sessions.items() if self._is_expired(s)]
            for sid in expired:
                self.sessions[sid].state = SessionState.EXPIRED


class ContextBuilder:
    @staticmethod
    def build_recommendation_context(session: SessionContext, user_profile: UserSessionProfile) -> Dict[str, Any]:
        return {
            "session": session.to_dict(),
            "user_profile": {
                "preferred_genres": dict(sorted(user_profile.preferred_genres.items(), key=lambda x: -x[1])[:10]),
                "preferred_decades": dict(sorted(user_profile.preferred_decades.items(), key=lambda x: -x[1])[:5]),
                "preferred_runtime_range": user_profile.preferred_runtime_range,
                "completion_rate": user_profile.completion_rate,
                "skip_rate": user_profile.skip_rate,
                "discovery_rate": user_profile.discovery_rate,
                "churn_risk_score": user_profile.churn_risk_score
            },
            "context": {
                "party_type": session.party_type.value,
                "party_size": session.party_size,
                "device_type": session.device_type.value,
                "flight_duration_minutes": session.flight_duration_minutes,
                "hotel_room_type": session.hotel_room_type,
                "subscription_tier": session.subscription_tier,
                "time_of_day": session.time_of_day,
                "day_of_week": session.day_of_week,
                "offline_mode": session.offline_mode,
                "bandwidth_mbps": session.bandwidth_mbps,
                "accessibility": session.accessibility
            }
        }


def main():
    manager = SessionManager()

    session = manager.create_session(
        user_id="user_123",
        partner_id="airline_default",
        context_overrides={
            "device_type": "seatback",
            "party_type": "solo",
            "flight_duration_minutes": 300,
            "flight_number": "AA123",
            "subscription_tier": "premium",
            "offline_mode": True
        }
    )

    print(f"Created session: {session.session_id}")
    print(f"Context: {session.to_dict()}")

    context = ContextBuilder.build_recommendation_context(
        session,
        manager.get_user_profile("user_123", "airline_default")
    )
    print(f"Recommendation context keys: {context.keys()}")


if __name__ == "__main__":
    main()