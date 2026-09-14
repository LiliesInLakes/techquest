import math
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from src.memory.models import (
    EventType, FeatureType, WatchEvent, RatingEvent,
    FeaturePreference, NegativePreference, UserProfileMemory
)
from src.memory.decay import compute_exponential_decay, aggregate_decayed_signals


# Default event weights
EVENT_WEIGHTS = {
    EventType.MOVIE_CLICKED: 0.10,
    EventType.MOVIE_VIEWED: 0.15,
    EventType.ADDED_TO_WATCHLIST: 0.30,
    EventType.MARKED_WATCHED: 0.50,
}

# Rating mapping spectrum
# (is_positive, positive_weight, is_negative, negative_penalty, is_neutral)
RATING_MAP = {
    5.0: (True, 1.0, False, 0.0, False),
    4.0: (True, 0.6, False, 0.0, False),
    3.0: (False, 0.0, False, 0.0, True),   # Neutral: increments interaction count, 0 weight, no penalty
    2.0: (False, 0.0, True, 0.4, False),
    1.0: (False, 0.0, True, 1.0, False),
    0.0: (False, 0.0, True, 1.0, False),
}


def map_rating_to_signals(rating: float) -> Tuple[bool, float, bool, float, bool]:
    """
    Maps 0-5 (or 0-10) star rating to (is_positive, pos_weight, is_negative, neg_penalty, is_neutral).
    Handles non-integer ratings by rounding to nearest half/whole star.
    """
    r_val = rating / 2.0 if rating > 5.0 else rating
    r = round(r_val)
    if r >= 5:
        return (True, 1.0, False, 0.0, False)
    elif r == 4:
        return (True, 0.6, False, 0.0, False)
    elif r == 3:
        return (False, 0.0, False, 0.0, True)
    elif r == 2:
        return (False, 0.0, True, 0.4, False)
    else:  # <= 1
        return (False, 0.0, True, 1.0, False)


def compute_beta(n_total_interactions: int) -> float:
    """
    Computes whiplash-free beta transition curve:
    beta(N) = 0.40 + 0.60 * max(0, 1 - N / 10)
    """
    ratio = max(0.0, 1.0 - (n_total_interactions / 10.0))
    return 0.40 + 0.60 * ratio


def compute_confidence(n_feature_interactions: int, k: float = 10.0) -> float:
    """
    Bayesian confidence smoothing:
    confidence(f) = N_f / (N_f + K)
    """
    if n_feature_interactions <= 0:
        return 0.0
    return float(n_feature_interactions) / (n_feature_interactions + k)


def compute_effective_binge_weight(sum_weight_24h: float) -> float:
    """
    Squashes 24-hour accumulated weight logarithmically:
    Effective_Weight = log2(1 + sum(Weight_24h))
    """
    if sum_weight_24h <= 0.0:
        return 0.0
    return math.log2(1.0 + sum_weight_24h)


def asymptotic_addition_update(current_score: float, signal_weight: float, alpha: float = 0.10) -> float:
    """
    Asymptotic addition for long-term preference:
    P_new = P_old + alpha * signal_weight * (1.0 - P_old)
    """
    current_score = max(0.0, min(1.0, current_score))
    signal_weight = max(0.0, signal_weight)
    headroom = 1.0 - current_score
    updated = current_score + alpha * signal_weight * headroom
    return max(0.0, min(1.0, updated))


class UserMemoryEngine:
    def __init__(self, alpha: float = 0.10, confidence_k: float = 10.0, half_life_days: float = 7.0):
        self.alpha = alpha
        self.confidence_k = confidence_k
        self.half_life_days = half_life_days

    def extract_features_from_metadata(self, metadata: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extracts categorical features from TMDB/Movie metadata dictionary.
        Returns dict of feature_type -> list of values.
        """
        extracted = defaultdict(list)
        
        # Genres
        genres = metadata.get("genres", [])
        if isinstance(genres, list):
            for g in genres:
                if isinstance(g, dict) and "name" in g:
                    extracted[FeatureType.GENRE.value].append(g["name"].strip().lower())
                elif isinstance(g, str) and g.strip():
                    extracted[FeatureType.GENRE.value].append(g.strip().lower())
        
        # Keywords
        keywords = metadata.get("keywords", [])
        if isinstance(keywords, list):
            for kw in keywords:
                if isinstance(kw, dict) and "name" in kw:
                    extracted[FeatureType.KEYWORD.value].append(kw["name"].strip().lower())
                elif isinstance(kw, str) and kw.strip():
                    extracted[FeatureType.KEYWORD.value].append(kw.strip().lower())
        
        # Director
        directors = metadata.get("director", [])
        if isinstance(directors, list):
            for d in directors:
                if isinstance(d, str) and d.strip():
                    extracted[FeatureType.DIRECTOR.value].append(d.strip().lower())
        elif isinstance(directors, str) and directors.strip():
            extracted[FeatureType.DIRECTOR.value].append(directors.strip().lower())
            
        # Cast (top 5)
        cast = metadata.get("cast", [])
        if isinstance(cast, list):
            for c in cast[:5]:
                if isinstance(c, str) and c.strip():
                    extracted[FeatureType.CAST.value].append(c.strip().lower())
        
        # Decade
        rel_date = metadata.get("release_date", "")
        if rel_date and len(rel_date) >= 4:
            try:
                year = int(rel_date[:4])
                decade = f"{(year // 10) * 10}s"
                extracted[FeatureType.DECADE.value].append(decade)
            except:
                pass

        # Language
        lang = metadata.get("original_language", metadata.get("language", ""))
        if lang and isinstance(lang, str):
            extracted[FeatureType.LANGUAGE.value].append(lang.strip().lower())

        return dict(extracted)

    def process_watch_event(
        self,
        event: WatchEvent,
        profile: UserProfileMemory,
        movie_metadata: Dict[str, Any],
        recent_same_movie_events: List[WatchEvent] = None,
        rolling_24h_feature_weights: Dict[Tuple[str, str], float] = None,
        current_time: Optional[float] = None
    ) -> UserProfileMemory:
        """
        Updates profile given a new watch event.
        Includes 30-min duplicate damping and 24-hr feature binge damping with asymptotic addition.
        """
        current_time = current_time or event.timestamp
        features = self.extract_features_from_metadata(movie_metadata)
        if not features:
            return profile

        # Base event weight
        base_weight = event.signal_weight
        if base_weight <= 0:
            base_weight = EVENT_WEIGHTS.get(event.event_type, 0.15)

        # 30-min deduplication damping
        is_duplicate = False
        if recent_same_movie_events:
            for past in recent_same_movie_events:
                if past.id != event.id and (event.timestamp - past.timestamp) <= 1800.0 and (event.timestamp >= past.timestamp):
                    is_duplicate = True
                    break
        
        effective_event_weight = base_weight * 0.25 if is_duplicate else base_weight

        profile.total_interactions += 1
        profile.updated_at = current_time

        # Update long-term & short-term for each feature
        for f_type, f_values in features.items():
            if f_type not in profile.long_term_preferences:
                profile.long_term_preferences[f_type] = {}
            if f_type not in profile.short_term_preferences:
                profile.short_term_preferences[f_type] = {}

            for f_val in f_values:
                # Long term preference lookup
                pref = profile.long_term_preferences[f_type].get(f_val)
                curr_score = pref.score if pref else 0.0
                curr_count = pref.interaction_count if pref else 0

                # 24-hour rolling weight calculation
                past_24h_w = 0.0
                if rolling_24h_feature_weights:
                    past_24h_w = rolling_24h_feature_weights.get((f_type, f_val), 0.0)
                total_24h_w = past_24h_w + effective_event_weight
                effective_binge_w = compute_effective_binge_weight(total_24h_w)

                # Asymptotic addition
                new_score = asymptotic_addition_update(curr_score, effective_binge_w, self.alpha)
                new_count = curr_count + 1
                new_conf = compute_confidence(new_count, self.confidence_k)

                profile.long_term_preferences[f_type][f_val] = FeaturePreference(
                    feature_type=FeatureType(f_type) if f_type in [ft.value for ft in FeatureType] else FeatureType.GENRE,
                    feature_value=f_val,
                    score=new_score,
                    confidence=new_conf,
                    interaction_count=new_count,
                    last_updated=current_time
                )

        return profile

    def process_rating_event(
        self,
        event: RatingEvent,
        profile: UserProfileMemory,
        movie_metadata: Dict[str, Any],
        current_time: Optional[float] = None
    ) -> UserProfileMemory:
        """
        Processes an explicit 0-5 star rating event.
        - 5 stars: +1.0 positive
        - 4 stars: +0.6 positive
        - 3 stars: 0.0 neutral (increments interaction count for confidence, 0 weight, no negative penalty)
        - 2 stars: +0.4 negative penalty
        - 1 & 0 stars: +1.0 negative penalty
        """
        current_time = current_time or event.timestamp
        features = self.extract_features_from_metadata(movie_metadata)
        if not features:
            return profile

        is_pos, pos_weight, is_neg, neg_penalty, is_neutral = map_rating_to_signals(event.rating)
        profile.total_interactions += 1
        profile.updated_at = current_time

        for f_type, f_values in features.items():
            if f_type not in profile.long_term_preferences:
                profile.long_term_preferences[f_type] = {}
            if is_neg and f_type not in profile.negative_preferences:
                profile.negative_preferences[f_type] = {}

            for f_val in f_values:
                # Handle positive & neutral (neutral increments count and confidence, without adding score)
                pref = profile.long_term_preferences[f_type].get(f_val)
                curr_score = pref.score if pref else 0.0
                curr_count = pref.interaction_count if pref else 0

                if is_pos:
                    new_score = asymptotic_addition_update(curr_score, pos_weight, self.alpha)
                    new_count = curr_count + 1
                elif is_neutral:
                    new_score = curr_score
                    new_count = curr_count + 1
                else:
                    new_score = curr_score
                    new_count = curr_count

                new_conf = compute_confidence(new_count, self.confidence_k)
                profile.long_term_preferences[f_type][f_val] = FeaturePreference(
                    feature_type=FeatureType(f_type) if f_type in [ft.value for ft in FeatureType] else FeatureType.GENRE,
                    feature_value=f_val,
                    score=new_score,
                    confidence=new_conf,
                    interaction_count=new_count,
                    last_updated=current_time
                )

                # Handle negative penalty tracking
                if is_neg:
                    neg_pref = profile.negative_preferences[f_type].get(f_val)
                    curr_penalty = neg_pref.penalty_weight if neg_pref else 0.0
                    curr_neg_count = neg_pref.interaction_count if neg_pref else 0

                    # Accumulate negative penalty asymptotically
                    new_penalty = asymptotic_addition_update(curr_penalty, neg_penalty, self.alpha * 2.0)
                    profile.negative_preferences[f_type][f_val] = NegativePreference(
                        feature_type=FeatureType(f_type) if f_type in [ft.value for ft in FeatureType] else FeatureType.GENRE,
                        feature_value=f_val,
                        penalty_weight=max(curr_penalty, new_penalty),
                        interaction_count=curr_neg_count + 1,
                        last_updated=current_time
                    )

        return profile

    def compute_short_term_profile(
        self,
        events: List[WatchEvent],
        movie_metadata_map: Dict[int, Dict[str, Any]],
        current_time: Optional[float] = None,
        window_days: float = 30.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes decayed short-term preferences over the recent window (e.g. 30 days).
        Uses exponential decay H = 7.0 days.
        """
        current_time = current_time or time.time()
        window_seconds = window_days * 86400.0
        earliest_ts = current_time - window_seconds

        # Map (feature_type, feature_val) -> list of (weight, timestamp)
        feature_signals = defaultdict(lambda: defaultdict(list))

        for ev in events:
            if ev.timestamp < earliest_ts:
                continue
            meta = movie_metadata_map.get(ev.movie_id, ev.metadata)
            if not meta:
                continue
            features = self.extract_features_from_metadata(meta)
            weight = ev.signal_weight if ev.signal_weight > 0 else EVENT_WEIGHTS.get(ev.event_type, 0.15)
            
            for f_type, f_vals in features.items():
                for f_val in f_vals:
                    feature_signals[f_type][f_val].append((weight, ev.timestamp))

        short_term = {}
        for f_type, val_dict in feature_signals.items():
            short_term[f_type] = {}
            for f_val, sig_list in val_dict.items():
                aggregated = aggregate_decayed_signals(sig_list, current_time, self.half_life_days)
                short_term[f_type][f_val] = aggregated

        return short_term

    def get_combined_score(
        self,
        feature_type: str,
        feature_val: str,
        profile: UserProfileMemory
    ) -> float:
        """
        Computes combined preference score:
        P_combined(f) = beta(N) * P_short_term(f) + (1 - beta(N)) * (confidence(f) * P_long_term(f))
        """
        beta = compute_beta(profile.total_interactions)
        
        # Short term score
        st_score = profile.short_term_preferences.get(feature_type, {}).get(feature_val, 0.0)
        
        # Long term score
        lt_pref = profile.long_term_preferences.get(feature_type, {}).get(feature_val)
        if lt_pref:
            lt_score = lt_pref.score
            lt_conf = lt_pref.confidence
        else:
            lt_score = 0.0
            lt_conf = 0.0
            
        combined = beta * st_score + (1.0 - beta) * (lt_conf * lt_score)
        return max(0.0, min(1.0, combined))

    def rebuild_profile_from_events(
        self,
        user_id: str,
        events: List[WatchEvent],
        ratings: List[RatingEvent],
        movie_metadata_map: Dict[int, Dict[str, Any]],
        current_time: Optional[float] = None
    ) -> UserProfileMemory:
        """
        Deterministically replays all watch events and rating events chronologically to reconstruct UserProfileMemory.
        """
        current_time = current_time or time.time()
        profile = UserProfileMemory(user_id=user_id)

        # Combine events and ratings chronologically
        timeline = []
        for ev in events:
            timeline.append(("event", ev.timestamp, ev))
        for r in ratings:
            timeline.append(("rating", r.timestamp, r))
        timeline.sort(key=lambda x: x[1])

        # Track 24h rolling weights during replay
        recent_events_by_movie = defaultdict(list)
        rolling_feature_weights = defaultdict(list)  # (f_type, f_val) -> list of (weight, ts)

        for item_type, ts, obj in timeline:
            if item_type == "event":
                ev: WatchEvent = obj
                meta = movie_metadata_map.get(ev.movie_id, ev.metadata)
                recent_same = recent_events_by_movie[ev.movie_id]
                
                # Compute rolling 24h weights
                cutoff_24h = ts - 86400.0
                r24_map = {}
                for (ft, fv), w_list in list(rolling_feature_weights.items()):
                    # Purge older than 24h
                    valid = [(w, w_ts) for w, w_ts in w_list if w_ts >= cutoff_24h]
                    rolling_feature_weights[(ft, fv)] = valid
                    if valid:
                        r24_map[(ft, fv)] = sum(w for w, _ in valid)

                profile = self.process_watch_event(
                    event=ev,
                    profile=profile,
                    movie_metadata=meta,
                    recent_same_movie_events=recent_same,
                    rolling_24h_feature_weights=r24_map,
                    current_time=ts
                )
                recent_events_by_movie[ev.movie_id].append(ev)
                
                # Add to rolling
                features = self.extract_features_from_metadata(meta)
                w = ev.signal_weight if ev.signal_weight > 0 else EVENT_WEIGHTS.get(ev.event_type, 0.15)
                for ft, fvs in features.items():
                    for fv in fvs:
                        rolling_feature_weights[(ft, fv)].append((w, ts))

            elif item_type == "rating":
                r: RatingEvent = obj
                meta = movie_metadata_map.get(r.movie_id, {})
                profile = self.process_rating_event(
                    event=r,
                    profile=profile,
                    movie_metadata=meta,
                    current_time=ts
                )

        # Compute short term profile
        profile.short_term_preferences = self.compute_short_term_profile(
            events=events,
            movie_metadata_map=movie_metadata_map,
            current_time=current_time
        )
        return profile
