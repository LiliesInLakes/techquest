import json
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict

from src.memory.models import (
    EventType, FeatureType, WatchEvent, RatingEvent,
    FeaturePreference, NegativePreference, UserProfileMemory
)
from src.memory.storage import MemoryStorage
from src.memory.engine import UserMemoryEngine, EVENT_WEIGHTS, map_rating_to_signals, compute_beta
from src.features.preference_extractor import UserPreferenceProfile


class UserMemoryService:
    def __init__(
        self,
        db_path: str = "data/memory.db",
        feature_matrix: Optional[Any] = None,
        alpha: float = 0.10,
        confidence_k: float = 10.0,
        half_life_days: float = 7.0
    ):
        self.storage = MemoryStorage(db_path=db_path)
        self.engine = UserMemoryEngine(alpha=alpha, confidence_k=confidence_k, half_life_days=half_life_days)
        self.feature_matrix = feature_matrix

    def set_feature_matrix(self, feature_matrix: Any):
        self.feature_matrix = feature_matrix

    def _ensure_feature_matrix(self):
        if self.feature_matrix is None:
            try:
                from src.features.matrix_builder import FeatureMatrixBuilder
                _, self.feature_matrix, _ = FeatureMatrixBuilder.load()
            except Exception:
                pass

    def _get_movie_metadata(self, movie_id: int, fallback_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if fallback_meta:
            return fallback_meta
        stored_meta = self.storage.get_movie_metadata_fallback(movie_id)
        if stored_meta:
            return stored_meta
        self._ensure_feature_matrix()
        if self.feature_matrix and hasattr(self.feature_matrix, "movie_metadata"):
            if movie_id in self.feature_matrix.movie_metadata:
                return self.feature_matrix.movie_metadata[movie_id]
        return {}

    def _serialize_profile(self, profile: UserProfileMemory) -> str:
        data = {
            "user_id": profile.user_id,
            "total_interactions": profile.total_interactions,
            "updated_at": profile.updated_at,
            "version": profile.version,
            "short_term_preferences": profile.short_term_preferences,
            "long_term_preferences": {
                ft: {
                    fv: {
                        "feature_type": pref.feature_type.value if hasattr(pref.feature_type, "value") else str(pref.feature_type),
                        "feature_value": pref.feature_value,
                        "score": pref.score,
                        "confidence": pref.confidence,
                        "interaction_count": pref.interaction_count,
                        "last_updated": pref.last_updated
                    } for fv, pref in val_dict.items()
                } for ft, val_dict in profile.long_term_preferences.items()
            },
            "negative_preferences": {
                ft: {
                    fv: {
                        "feature_type": pref.feature_type.value if hasattr(pref.feature_type, "value") else str(pref.feature_type),
                        "feature_value": pref.feature_value,
                        "penalty_weight": pref.penalty_weight,
                        "interaction_count": pref.interaction_count,
                        "last_updated": pref.last_updated
                    } for fv, pref in val_dict.items()
                } for ft, val_dict in profile.negative_preferences.items()
            },
            "subscriptions": getattr(profile, "subscriptions", ["Netflix", "Prime Video", "Disney+"])
        }
        return json.dumps(data)

    def _deserialize_profile(self, profile_json: str) -> UserProfileMemory:
        data = json.loads(profile_json)
        lt_prefs = {}
        for ft, val_dict in data.get("long_term_preferences", {}).items():
            lt_prefs[ft] = {}
            for fv, item in val_dict.items():
                lt_prefs[ft][fv] = FeaturePreference(
                    feature_type=FeatureType(item["feature_type"]) if item["feature_type"] in [e.value for e in FeatureType] else FeatureType.GENRE,
                    feature_value=item["feature_value"],
                    score=float(item["score"]),
                    confidence=float(item["confidence"]),
                    interaction_count=int(item["interaction_count"]),
                    last_updated=float(item.get("last_updated", time.time()))
                )

        neg_prefs = {}
        for ft, val_dict in data.get("negative_preferences", {}).items():
            neg_prefs[ft] = {}
            for fv, item in val_dict.items():
                neg_prefs[ft][fv] = NegativePreference(
                    feature_type=FeatureType(item["feature_type"]) if item["feature_type"] in [e.value for e in FeatureType] else FeatureType.GENRE,
                    feature_value=item["feature_value"],
                    penalty_weight=float(item["penalty_weight"]),
                    interaction_count=int(item["interaction_count"]),
                    last_updated=float(item.get("last_updated", time.time()))
                )

        return UserProfileMemory(
            user_id=data["user_id"],
            total_interactions=data.get("total_interactions", 0),
            short_term_preferences=data.get("short_term_preferences", {}),
            long_term_preferences=lt_prefs,
            negative_preferences=neg_prefs,
            subscriptions=data.get("subscriptions", ["Netflix", "Prime Video", "Disney+"]),
            updated_at=data.get("updated_at", time.time()),
            version=data.get("version", 1)
        )

    def update_user_subscriptions(self, user_id: str, subscriptions: List[str]) -> UserProfileMemory:
        profile = self.get_user_profile(user_id)
        profile.subscriptions = subscriptions
        profile.updated_at = time.time()
        self.storage.save_profile_json(user_id, self._serialize_profile(profile), updated_at=profile.updated_at)
        return profile

    def record_interaction(
        self,
        user_id: str,
        movie_id: int,
        event_type: EventType | str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> WatchEvent:
        ts = timestamp or time.time()
        if isinstance(event_type, str):
            event_type = EventType(event_type) if event_type in [e.value for e in EventType] else EventType.MOVIE_VIEWED

        signal_weight = EVENT_WEIGHTS.get(event_type, 0.15)
        meta = self._get_movie_metadata(movie_id, metadata)

        event = WatchEvent(
            user_id=user_id,
            movie_id=movie_id,
            event_type=event_type,
            signal_weight=signal_weight,
            timestamp=ts,
            metadata=meta
        )
        event_id = self.storage.record_event(event)
        event.id = event_id

        # Update and save profile
        profile = self.get_user_profile(user_id)
        recent_events = self.storage.get_recent_movie_events(user_id, movie_id, window_seconds=1800.0, current_time=ts)
        
        # Calculate rolling 24h weights for features
        past_events_24h = self.storage.get_events(user_id, since_timestamp=ts - 86400.0)
        rolling_24h = defaultdict(float)
        for past_ev in past_events_24h:
            if past_ev.id != event.id:
                p_meta = self._get_movie_metadata(past_ev.movie_id, past_ev.metadata)
                p_features = self.engine.extract_features_from_metadata(p_meta)
                for ft, fvs in p_features.items():
                    for fv in fvs:
                        rolling_24h[(ft, fv)] += past_ev.signal_weight

        profile = self.engine.process_watch_event(
            event=event,
            profile=profile,
            movie_metadata=meta,
            recent_same_movie_events=recent_events,
            rolling_24h_feature_weights=rolling_24h,
            current_time=ts
        )
        
        # Update short term preferences
        events_all = self.storage.get_events(user_id)
        movie_meta_map = {ev.movie_id: self._get_movie_metadata(ev.movie_id, ev.metadata) for ev in events_all}
        profile.short_term_preferences = self.engine.compute_short_term_profile(
            events=events_all,
            movie_metadata_map=movie_meta_map,
            current_time=ts
        )

        self.storage.save_profile_json(user_id, self._serialize_profile(profile), updated_at=ts)
        return event

    def record_rejection(
        self,
        user_id: str,
        movie_id: int,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[WatchEvent, UserProfileMemory]:
        """
        Records an explicit movie rejection / not-interested event, updates negative preference
        penalties in the user profile memory, and returns the event and updated profile.
        """
        ts = timestamp or time.time()
        meta = self._get_movie_metadata(movie_id, metadata)
        if reason:
            meta["rejection_reason"] = reason

        event = WatchEvent(
            user_id=user_id,
            movie_id=movie_id,
            event_type=EventType.REJECTED,
            signal_weight=0.5,
            timestamp=ts,
            metadata=meta
        )
        event_id = self.storage.record_event(event)
        event.id = event_id

        # Update profile with negative penalties
        profile = self.get_user_profile(user_id)
        profile = self.engine.process_rejection_event(
            event=event,
            profile=profile,
            movie_metadata=meta,
            penalty=0.5,
            current_time=ts
        )

        self.storage.save_profile_json(user_id, self._serialize_profile(profile), updated_at=ts)
        return event, profile

    def record_rating(
        self,
        user_id: str,
        movie_id: int,
        rating: float,
        review_text: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RatingEvent:
        if isinstance(review_text, dict):
            metadata = review_text
            review_text = None

        ts = timestamp or time.time()
        _, pos_weight, _, neg_penalty, _ = map_rating_to_signals(rating)
        signal_target = pos_weight if pos_weight > 0 else (1.0 - neg_penalty)

        rating_event = RatingEvent(
            user_id=user_id,
            movie_id=movie_id,
            rating=rating,
            signal_target=signal_target,
            timestamp=ts,
            review_text=review_text
        )
        r_id = self.storage.record_rating(rating_event)
        rating_event.id = r_id

        # Update profile
        profile = self.get_user_profile(user_id)
        meta = self._get_movie_metadata(movie_id, metadata)
        profile = self.engine.process_rating_event(
            event=rating_event,
            profile=profile,
            movie_metadata=meta,
            current_time=ts
        )
        self.storage.save_profile_json(user_id, self._serialize_profile(profile), updated_at=ts)
        return rating_event

    def record_search_query(
        self,
        user_id: str,
        query_text: str,
        extracted_features: Optional[Any] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Processes a search exploration query to dynamically update user taste profile
        with extracted tropes, themes, keywords, and genres.
        """
        ts = timestamp or time.time()
        profile = self.get_user_profile(user_id)
        
        keywords = []
        genres = []
        if extracted_features:
            if hasattr(extracted_features, "keywords"):
                keywords.extend(extracted_features.keywords)
            elif isinstance(extracted_features, dict):
                keywords.extend(extracted_features.get("keywords", []))
                
            if hasattr(extracted_features, "genres"):
                genres.extend(extracted_features.genres)
            elif isinstance(extracted_features, dict):
                genres.extend(extracted_features.get("genres", []))

        # NOTE: We do NOT fall back to raw query word tokenization here.
        # Only structured features extracted by NLFeatureExtractor (keywords, tmdb_keywords, genres)
        # should influence taste memory.  Falling back to raw word splitting causes generic verbs
        # and pronouns ("Tonight", "Give", "Something") to leak into learned keyword preferences.
        if not keywords and not genres:
            return

        profile.total_interactions += 1
        profile.updated_at = ts

        # Update long-term & short-term preferences for keywords
        if FeatureType.KEYWORD.value not in profile.long_term_preferences:
            profile.long_term_preferences[FeatureType.KEYWORD.value] = {}
        if FeatureType.KEYWORD.value not in profile.short_term_preferences:
            profile.short_term_preferences[FeatureType.KEYWORD.value] = {}

        for kw in keywords:
            kw_clean = str(kw).strip().lower()
            if not kw_clean or len(kw_clean) < 2:
                continue
            
            # Short-term update
            prev_st = profile.short_term_preferences[FeatureType.KEYWORD.value].get(kw_clean, 0.0)
            profile.short_term_preferences[FeatureType.KEYWORD.value][kw_clean] = prev_st + 0.35

            # Long-term update
            pref = profile.long_term_preferences[FeatureType.KEYWORD.value].get(kw_clean)
            curr_score = pref.score if pref else 0.0
            curr_count = pref.interaction_count if pref else 0
            
            from src.memory.engine import asymptotic_addition_update, compute_confidence
            new_score = asymptotic_addition_update(curr_score, 0.30, self.engine.alpha)
            new_count = curr_count + 1
            new_conf = compute_confidence(new_count, self.engine.confidence_k)
            
            profile.long_term_preferences[FeatureType.KEYWORD.value][kw_clean] = FeaturePreference(
                feature_type=FeatureType.KEYWORD,
                feature_value=kw_clean,
                score=new_score,
                confidence=new_conf,
                interaction_count=new_count,
                last_updated=ts
            )

        # Update long-term & short-term preferences for genres
        if FeatureType.GENRE.value not in profile.long_term_preferences:
            profile.long_term_preferences[FeatureType.GENRE.value] = {}
        if FeatureType.GENRE.value not in profile.short_term_preferences:
            profile.short_term_preferences[FeatureType.GENRE.value] = {}

        for g in genres:
            g_clean = str(g).strip().lower()
            if not g_clean:
                continue
            prev_st = profile.short_term_preferences[FeatureType.GENRE.value].get(g_clean, 0.0)
            profile.short_term_preferences[FeatureType.GENRE.value][g_clean] = prev_st + 0.40

            pref = profile.long_term_preferences[FeatureType.GENRE.value].get(g_clean)
            curr_score = pref.score if pref else 0.0
            curr_count = pref.interaction_count if pref else 0

            from src.memory.engine import asymptotic_addition_update, compute_confidence
            new_score = asymptotic_addition_update(curr_score, 0.35, self.engine.alpha)
            new_count = curr_count + 1
            new_conf = compute_confidence(new_count, self.engine.confidence_k)

            profile.long_term_preferences[FeatureType.GENRE.value][g_clean] = FeaturePreference(
                feature_type=FeatureType.GENRE,
                feature_value=g_clean,
                score=new_score,
                confidence=new_conf,
                interaction_count=new_count,
                last_updated=ts
            )

        self.storage.save_profile_json(user_id, self._serialize_profile(profile), updated_at=ts)

    def get_user_profile(self, user_id: str) -> UserProfileMemory:
        raw_json = self.storage.get_profile_json(user_id)
        if raw_json:
            try:
                return self._deserialize_profile(raw_json)
            except:
                pass
        # Empty profile fallback
        return UserProfileMemory(user_id=user_id)

    def rebuild_user_profile(self, user_id: str) -> UserProfileMemory:
        events = self.storage.get_events(user_id)
        ratings = self.storage.get_ratings(user_id)
        movie_meta_map = {}
        for ev in events:
            movie_meta_map[ev.movie_id] = self._get_movie_metadata(ev.movie_id, ev.metadata)
        for r in ratings:
            if r.movie_id not in movie_meta_map:
                movie_meta_map[r.movie_id] = self._get_movie_metadata(r.movie_id)

        profile = self.engine.rebuild_profile_from_events(
            user_id=user_id,
            events=events,
            ratings=ratings,
            movie_metadata_map=movie_meta_map
        )
        self.storage.save_profile_json(user_id, self._serialize_profile(profile))
        return profile

    def get_combined_preference_profile(self, user_id: str) -> UserPreferenceProfile:
        """
        Creates a UserPreferenceProfile adapter object configured with combined short/long term scores
        and negative preference penalties, perfectly compatible with MovieRanker.
        """
        self._ensure_feature_matrix()
        profile = self.get_user_profile(user_id)
        
        # Aggregate top genres, keywords, directors, cast, decades, languages
        def get_sorted_feature_list(f_type: str, top_n: int = 10) -> List[Tuple[str, float]]:
            scores = {}
            all_keys = set(profile.short_term_preferences.get(f_type, {}).keys()) | set(profile.long_term_preferences.get(f_type, {}).keys())
            for k in all_keys:
                scores[k] = self.engine.get_combined_score(f_type, k, profile)
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_items[:top_n]

        top_genres = get_sorted_feature_list(FeatureType.GENRE.value, top_n=10)
        top_tags = get_sorted_feature_list(FeatureType.TAG.value, top_n=15)
        top_themes = get_sorted_feature_list(FeatureType.THEME.value, top_n=10)
        top_tones = get_sorted_feature_list(FeatureType.TONE.value, top_n=10)
        top_keywords = get_sorted_feature_list(FeatureType.KEYWORD.value, top_n=20)
        top_directors = get_sorted_feature_list(FeatureType.DIRECTOR.value, top_n=10)
        top_cast = get_sorted_feature_list(FeatureType.CAST.value, top_n=10)
        top_decades = get_sorted_feature_list(FeatureType.DECADE.value, top_n=5)
        top_languages = get_sorted_feature_list(FeatureType.LANGUAGE.value, top_n=5)

        genre_weights = {k: self.engine.get_combined_score(FeatureType.GENRE.value, k, profile) for k, _ in top_genres}
        keyword_weights = {k: self.engine.get_combined_score(FeatureType.KEYWORD.value, k, profile) for k, _ in top_keywords}

        # Build feature vector if feature matrix is present
        vec_len = self.feature_matrix.count_matrix.shape[1] if self.feature_matrix and hasattr(self.feature_matrix, "count_matrix") else 1000
        feature_vector = np.zeros(vec_len)
        if self.feature_matrix and hasattr(self.feature_matrix, "feature_names"):
            for i, name in enumerate(self.feature_matrix.feature_names):
                name_lower = name.lower()
                if name_lower in keyword_weights:
                    feature_vector[i] = keyword_weights[name_lower]
                elif name_lower in genre_weights:
                    feature_vector[i] = genre_weights[name_lower]

        # Ratings stats
        ratings = self.storage.get_ratings(user_id)
        rating_vals = [r.rating for r in ratings]
        avg_rating = float(np.mean(rating_vals)) if rating_vals else 0.0
        rating_std = float(np.std(rating_vals)) if rating_vals else 0.0
        
        rating_dist = {str(i): 0.0 for i in range(1, 6)}
        if rating_vals:
            for v in rating_vals:
                bucket = str(max(1, min(5, round(v))))
                rating_dist[bucket] = rating_dist.get(bucket, 0.0) + 1.0 / len(rating_vals)

        # Prepare negative preferences lookup: feature_value -> penalty_weight
        negative_lookup = {}
        for ft, val_dict in profile.negative_preferences.items():
            for fv, neg_pref in val_dict.items():
                negative_lookup[fv.lower()] = neg_pref.penalty_weight

        adapter_profile = UserPreferenceProfile(
            user_id=user_id,
            top_genres=top_genres,
            top_keywords=top_keywords,
            top_directors=top_directors,
            top_cast=top_cast,
            rating_distribution=rating_dist,
            avg_rating=avg_rating,
            rating_std=rating_std,
            preferred_decades=top_decades,
            preferred_languages=top_languages,
            watch_count=profile.total_interactions,
            feature_vector=feature_vector,
            keyword_weights=keyword_weights,
            genre_weights=genre_weights,
            top_tags=top_tags,
            top_themes=top_themes,
            top_tones=top_tones,
            negative_preferences=negative_lookup
        )
        setattr(adapter_profile, "raw_memory_profile", profile)
        return adapter_profile

    def get_user_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Returns structured multi-dimensional summary including genres, tags, themes,
        tones, keywords, active negative penalties, and learning curve metrics.
        """
        profile = self.get_user_profile(user_id)
        combined = self.get_combined_preference_profile(user_id)
        
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

        negative_penalties = []
        for ft, val_dict in profile.negative_preferences.items():
            for fv, neg_pref in val_dict.items():
                if neg_pref.penalty_weight > 0.05:
                    negative_penalties.append({
                        "feature_type": ft,
                        "name": fv,
                        "penalty": round(neg_pref.penalty_weight, 3),
                        "count": neg_pref.interaction_count
                    })
        negative_penalties.sort(key=lambda x: x["penalty"], reverse=True)

        return {
            "user_id": user_id,
            "genres": [{"name": g, "score": round(score, 3)} for g, score in combined.top_genres if score > 0],
            "tags": [{"name": t, "score": round(score, 3)} for t, score in combined.top_tags if score > 0],
            "themes": [{"name": th, "score": round(score, 3)} for th, score in combined.top_themes if score > 0],
            "tones": [{"name": tn, "score": round(score, 3)} for tn, score in combined.top_tones if score > 0],
            "keywords": [{"name": k, "score": round(score, 3)} for k, score in combined.top_keywords if score > 0],
            "negative_penalties": negative_penalties,
            "subscriptions": getattr(profile, "subscriptions", ["Netflix", "Prime Video", "Disney+"]),
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
