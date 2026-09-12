import asyncio
import sys
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
import yaml

from src.tmdb.client import TMDBClient, Movie
from src.features.matrix_builder import FeatureMatrixBuilder, FeatureMatrix
from src.features.semantic_search import SemanticKeywordSearch, HybridKeywordMatcher
from src.features.preference_extractor import PreferenceExtractor, UserPreferenceProfile
from src.ranking.ranker import MovieRanker, RankingConfig, RankedMovie, DiversityFilter
from src.llm.reasoning import DeepSeekReasoner, ReasoningTemplateEngine, RecommendationReasoning
from src.llm.nl_feature_extractor import NLFeatureExtractor, ExtractedFeatures


@dataclass
class RecommendationResult:
    movie: Dict[str, Any]
    reasoning: str
    score: float
    rank: int


class MovieRecommender:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.rec_config = self.config.get("recommendation", {})

        self.tmdb_client: Optional[TMDBClient] = None
        self.feature_matrix: Optional[FeatureMatrix] = None
        self.semantic_index: Optional[Dict] = None
        self.matrix_builder: Optional[FeatureMatrixBuilder] = None
        self.semantic_search: Optional[SemanticKeywordSearch] = None
        self.hybrid_matcher: Optional[HybridKeywordMatcher] = None
        self.preference_extractor: Optional[PreferenceExtractor] = None
        self.ranker: Optional[MovieRanker] = None
        self.reasoner: Optional[DeepSeekReasoner] = None
        self.nl_extractor: Optional[NLFeatureExtractor] = None
        self.template_engine = ReasoningTemplateEngine()
        self.diversity_filter = DiversityFilter()
        self._initialized = False

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    async def initialize(self):
        if self._initialized:
            return
        
        try:
            self.tmdb_client = TMDBClient(config_path=self.config_path)
        except ValueError as e:
            # No/invalid TMDB key: live discovery disabled, local
            # feature-matrix pool is used instead (see _get_local_candidates).
            print(f"TMDB not configured ({e}), using local candidate pool only", file=sys.stderr)
            self.tmdb_client = None
        self.matrix_builder = FeatureMatrixBuilder(config_path=self.config_path)
        
        try:
            self.matrix_builder, self.feature_matrix, self.semantic_index = \
                FeatureMatrixBuilder.load(self.config_path)
            print("Loaded precomputed feature matrix", file=sys.stderr)
        except FileNotFoundError:
            print("Precomputed matrix not found, building...", file=sys.stderr)
            self.feature_matrix = self.matrix_builder.build_and_save()
            self.semantic_index = self.matrix_builder.build_semantic_index(self.feature_matrix)
        
        self.semantic_search = SemanticKeywordSearch(threshold=0.65)
        all_keywords = []
        for meta in self.feature_matrix.movie_metadata.values():
            all_keywords.extend(meta.get('keywords', []))
            all_keywords.extend(meta.get('genres', []))
        self.semantic_search.build_index(all_keywords)
        
        self.hybrid_matcher = HybridKeywordMatcher(
            self.semantic_search, self.feature_matrix.feature_names
        )
        
        self.preference_extractor = PreferenceExtractor(
            feature_matrix=self.feature_matrix,
            semantic_search=self.semantic_search
        )
        
        self.ranker = MovieRanker(config_path=self.config_path)
        self.ranker.set_dependencies(
            self.preference_extractor, self.feature_matrix, self.semantic_search
        )
        
        try:
            self.reasoner = DeepSeekReasoner(config_path=self.config_path)
        except ValueError:
            print("DeepSeek not configured, using template engine for reasoning", file=sys.stderr)
            self.reasoner = None

        self.nl_extractor = NLFeatureExtractor(
            config_path=self.config_path,
            semantic_search=self.semantic_search
        )
        
        self._initialized = True

    async def search_movies(
        self,
        genre: Optional[str] = None,
        region: str = "US",
        language: str = "en",
        subscriptions: List[str] = None,
        min_rating: float = 0.0,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        await self.initialize()

        if self.tmdb_client is None:
            # Local fallback: filter the feature-matrix pool by genre/rating.
            pool = self._get_local_candidates(None)
            if genre:
                pool = [c for c in pool if genre.lower() in [g.lower() for g in c.get('genres', [])]]
            pool = [c for c in pool if c.get('vote_average', 0) >= min_rating]
            return pool[:max_results]

        try:
            async with self.tmdb_client as client:
                movies = await client.discover_movies(
                    genre=genre,
                    region=region,
                    language=language,
                    min_rating=min_rating,
                    with_watch_providers=subscriptions,
                    watch_region=region,
                    sort_by="popularity.desc"
                )

                results = []
                for m in movies[:max_results]:
                    results.append({
                        'id': m.id,
                        'title': m.title,
                        'overview': m.overview,
                        'genres': m.genres,
                        'release_date': m.release_date,
                        'vote_average': m.vote_average,
                        'vote_count': m.vote_count,
                        'popularity': m.popularity,
                        'original_language': m.original_language,
                        'poster_path': m.poster_path
                    })

                return results
        except Exception as e:
            print(f"TMDB unavailable ({e}), falling back to local pool", file=sys.stderr)
            pool = self._get_local_candidates(None)
            if genre:
                pool = [c for c in pool if genre.lower() in [g.lower() for g in c.get('genres', [])]]
            return pool[:max_results]

    async def get_recommendations(
        self,
        user_id: str,
        watch_history: List[Dict[str, Any]],
        preferences: Dict[str, Any] = None,
        subscriptions: List[str] = None,
        region: str = "US",
        language: str = "en",
        top_k: int = 10,
        natural_language: str = None
    ) -> Dict[str, Any]:
        await self.initialize()
        
        preferences = preferences or {}
        subscriptions = subscriptions or []
        
        ratings = {}
        for entry in watch_history:
            movie_id = entry.get('movie_id') or entry.get('id')
            rating = entry.get('rating', 0)
            if rating > 0:
                ratings[movie_id] = rating
        
        user_profile = self.preference_extractor.extract_preferences(
            watch_history, ratings, self.feature_matrix
        )
        
        extracted_features = None
        if natural_language:
            context = {
                "subscriptions": subscriptions,
                "region": region,
                "language": language,
                "watch_history_summary": self._summarize_watch_history(watch_history)
            }
            extracted_features = await self.nl_extractor.extract(natural_language, context)
            
            if self.feature_matrix:
                vocab = set(self.feature_matrix.feature_names)
                extracted_features = self.nl_extractor.map_to_vocabulary(extracted_features, vocab)
        
        candidate_movies = await self._get_candidate_movies_stage1(
            extracted_features, subscriptions, region, language, user_profile
        )
        
        if not candidate_movies:
            return {
                'recommendations': [],
                'user_profile': {
                    'top_genres': user_profile.top_genres[:5],
                    'top_keywords': user_profile.top_keywords[:10],
                    'avg_rating': user_profile.avg_rating,
                    'watch_count': user_profile.watch_count
                }
            }
        
        query_text = extracted_features.query_text if extracted_features else self._build_query_from_profile(user_profile, preferences)
        query_count, query_tfidf, query_embedding = self.matrix_builder.transform_query(query_text)
        
        stage1_ranked = self.ranker.rank_movies_stage1(
            candidate_movies, query_embedding, query_tfidf
        )
        
        stage1_top = self.ranker.get_top_k(stage1_ranked, self.rec_config.get("stage1_candidates", 100))
        
        final_ranked = self.ranker.rank_movies_stage2(
            stage1_top, user_profile, extracted_features, self.feature_matrix,
            history_boost_factor=self.rec_config.get("history_boost_factor", 0.3)
        )
        
        final_ranked = self.diversity_filter.filter(final_ranked, self.feature_matrix)
        top_movies = self.ranker.get_top_k(final_ranked, top_k)
        
        movie_dicts = []
        ranking_factors = []
        for rm in top_movies:
            meta = self.feature_matrix.movie_metadata[rm.movie_id]
            movie_dicts.append({
                'id': rm.movie_id,
                'title': rm.title,
                'overview': rm.overview,
                'genres': rm.genres,
                'vote_average': rm.vote_average,
                'vote_count': rm.vote_count,
                'release_date': meta.get('release_date', ''),
                'director': meta.get('director', []),
                'cast': meta.get('cast', [])
            })
            ranking_factors.append(rm.reasoning_factors)
        
        watch_history_summary = self._summarize_watch_history(watch_history)
        user_query = natural_language if natural_language else (preferences.get('query', '') if preferences else '')
        
        reasonings = await self._generate_reasonings(
            movie_dicts, user_profile, ranking_factors, user_query, watch_history_summary, extracted_features
        )
        
        recommendations = []
        for i, (movie, reasoning) in enumerate(zip(movie_dicts, reasonings)):
            recommendations.append(RecommendationResult(
                movie=movie,
                reasoning=reasoning.reasoning,
                score=top_movies[i].score,
                rank=i + 1
            ))
        
        return {
            'recommendations': [
                {
                    'rank': r.rank,
                    'movie': r.movie,
                    'reasoning': r.reasoning,
                    'score': round(r.score, 4)
                }
                for r in recommendations
            ],
            'user_profile': {
                'top_genres': user_profile.top_genres[:5],
                'top_keywords': user_profile.top_keywords[:10],
                'avg_rating': user_profile.avg_rating,
                'watch_count': user_profile.watch_count
            },
            'extracted_features': extracted_features.to_dict() if extracted_features else None
        }

    async def _get_candidate_movies_stage1(
        self,
        extracted_features: Optional[ExtractedFeatures],
        subscriptions: List[str],
        region: str,
        language: str,
        user_profile: UserPreferenceProfile
    ) -> List[Dict[str, Any]]:
        # Stage 1: TMDB returns a LOOSELY sorted pool only (popularity.desc).
        # No strict filtering at the API level besides the primary genre;
        # exclusions are applied in Python. If TMDB is unreachable
        # (no/invalid key, offline), fall back to the local feature-matrix pool.
        if extracted_features:
            if self.tmdb_client is None:
                return self._get_local_candidates(extracted_features)
            try:
                async with self.tmdb_client as client:
                    tmdb_params = self.nl_extractor.build_tmdb_discover_params(
                        extracted_features, subscriptions, region, language
                    )
                    movies = await client.discover_movies(**tmdb_params)

                    candidates = []
                    for m in movies[:self.rec_config.get("stage1_max_tmdb_results", 200)]:
                        if self._passes_exclusions(m, extracted_features):
                            candidates.append({
                                'id': m.id,
                                'title': m.title,
                                'overview': m.overview,
                                'genres': m.genres,
                                'release_date': m.release_date,
                                'vote_average': m.vote_average,
                                'vote_count': m.vote_count,
                                'popularity': m.popularity,
                                'original_language': m.original_language
                            })

                    if len(candidates) < 50:
                        popular = await client.get_popular_movies(region=region, language=language, page=1)
                        for m in popular[:100]:
                            if m.id not in {c['id'] for c in candidates} and self._passes_exclusions(m, extracted_features):
                                candidates.append({
                                    'id': m.id,
                                    'title': m.title,
                                    'overview': m.overview,
                                    'genres': m.genres,
                                    'release_date': m.release_date,
                                    'vote_average': m.vote_average,
                                    'vote_count': m.vote_count,
                                    'popularity': m.popularity,
                                    'original_language': m.original_language
                                })

                    if candidates:
                        return candidates[:self.rec_config.get("stage1_candidates", 100)]
            except Exception as e:
                print(f"TMDB unavailable ({e}), falling back to local candidate pool", file=sys.stderr)

            return self._get_local_candidates(extracted_features)
        else:
            try:
                return await self._get_candidate_movies(subscriptions, region, language, user_profile)
            except Exception as e:
                print(f"TMDB unavailable ({e}), falling back to local candidate pool", file=sys.stderr)
                return self._get_local_candidates(None)

    def _get_local_candidates(
        self, extracted_features: Optional[ExtractedFeatures]
    ) -> List[Dict[str, Any]]:
        # Local fallback: every movie in the precomputed feature matrix,
        # minus hard exclusions. Loose pool — ranking happens in stage 1/2.
        candidates = []
        for movie_id, meta in self.feature_matrix.movie_metadata.items():
            movie = {
                'id': movie_id,
                'title': meta.get('title', ''),
                'overview': meta.get('overview', ''),
                'genres': meta.get('genres', []),
                'release_date': meta.get('release_date', ''),
                'vote_average': meta.get('vote_average', 0),
                'vote_count': meta.get('vote_count', 0),
                'popularity': meta.get('popularity', 0),
                'original_language': meta.get('original_language', 'en'),
            }
            if self._passes_exclusions(movie, extracted_features):
                candidates.append(movie)
        candidates.sort(key=lambda c: (c.get('vote_average', 0), c.get('popularity', 0)), reverse=True)
        return candidates

    def _passes_exclusions(self, movie, extracted_features: ExtractedFeatures) -> bool:
        if not extracted_features:
            return True

        if isinstance(movie, dict):
            genres = movie.get('genres', [])
            overview = movie.get('overview', '')
        else:
            genres = movie.genres
            overview = movie.overview

        movie_genres_lower = [g.lower() for g in genres]
        for excl in extracted_features.exclude_genres:
            if excl.lower() in movie_genres_lower:
                return False

        movie_overview_lower = (overview or '').lower()
        for excl in extracted_features.exclude_keywords:
            if excl.lower() in movie_overview_lower:
                return False

        return True

    async def _get_candidate_movies(
        self,
        subscriptions: List[str],
        region: str,
        language: str,
        user_profile: UserPreferenceProfile
    ) -> List[Dict[str, Any]]:
        async with self.tmdb_client as client:
            candidates = []
            
            for genre, weight in user_profile.top_genres[:5]:
                movies = await client.discover_movies(
                    genre=genre,
                    region=region,
                    language=language,
                    min_rating=6.0,
                    with_watch_providers=subscriptions,
                    watch_region=region,
                    sort_by="vote_average.desc",
                    page=1
                )
                for m in movies[:30]:
                    candidates.append({
                        'id': m.id,
                        'title': m.title,
                        'overview': m.overview,
                        'genres': m.genres,
                        'release_date': m.release_date,
                        'vote_average': m.vote_average,
                        'vote_count': m.vote_count,
                        'popularity': m.popularity,
                        'original_language': m.original_language
                    })
            
            popular = await client.get_popular_movies(region=region, language=language, page=1)
            for m in popular[:50]:
                candidates.append({
                    'id': m.id,
                    'title': m.title,
                    'overview': m.overview,
                    'genres': m.genres,
                    'release_date': m.release_date,
                    'vote_average': m.vote_average,
                    'vote_count': m.vote_count,
                    'popularity': m.popularity,
                    'original_language': m.original_language
                })
            
            seen = set()
            unique_candidates = []
            for c in candidates:
                if c['id'] not in seen:
                    seen.add(c['id'])
                    unique_candidates.append(c)
            
            return unique_candidates[:200]

    def _build_query_from_profile(
        self,
        user_profile: UserPreferenceProfile,
        preferences: Dict[str, Any] = None
    ) -> str:
        parts = []
        
        for genre, _ in user_profile.top_genres[:5]:
            parts.append(genre)
        
        for keyword, _ in user_profile.top_keywords[:10]:
            parts.append(keyword)
        
        if preferences and preferences.get('query'):
            parts.append(preferences['query'])
        
        if preferences and preferences.get('mood'):
            parts.append(preferences['mood'])
        
        return ' '.join(parts)

    def _summarize_watch_history(self, watch_history: List[Dict[str, Any]]) -> str:
        if not watch_history:
            return "No watch history available"
        
        total = len(watch_history)
        rated = sum(1 for e in watch_history if e.get('rating', 0) > 0)
        avg_rating = np.mean([e.get('rating', 0) for e in watch_history if e.get('rating', 0) > 0]) if rated > 0 else 0
        
        return f"User has watched {total} movies, rated {rated} of them (avg {avg_rating:.1f}/10)"

    async def _generate_reasonings(
        self,
        movies: List[Dict[str, Any]],
        user_profile: UserPreferenceProfile,
        ranking_factors: List[Dict[str, float]],
        user_query: str,
        watch_history_summary: str,
        extracted_features: Optional[ExtractedFeatures] = None
    ) -> List[RecommendationReasoning]:
        if self.reasoner:
            return await self.reasoner.generate_batch_reasoning(
                movies, user_profile, ranking_factors, user_query, watch_history_summary
            )
        else:
            return [
                RecommendationReasoning(
                    movie_title=m['title'],
                    reasoning=self.template_engine.generate(m, user_profile, f),
                    key_factors=[],
                    user_preferences_matched=[]
                )
                for m, f in zip(movies, ranking_factors)
            ]

    def analyze_user_preferences(
        self,
        watch_history: List[Dict[str, Any]],
        ratings: Dict[int, float] = None
    ) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Recommender not initialized. Call initialize() first.")
        
        ratings = ratings or {}
        profile = self.preference_extractor.extract_preferences(
            watch_history, ratings, self.feature_matrix
        )
        
        return {
            'top_genres': profile.top_genres,
            'top_keywords': profile.top_keywords,
            'top_directors': profile.top_directors,
            'top_cast': profile.top_cast,
            'rating_distribution': profile.rating_distribution,
            'avg_rating': profile.avg_rating,
            'rating_std': profile.rating_std,
            'preferred_decades': profile.preferred_decades,
            'preferred_languages': profile.preferred_languages,
            'watch_count': profile.watch_count
        }

    async def get_movie_details(self, movie_id: int) -> Dict[str, Any]:
        await self.initialize()
        
        async with self.tmdb_client as client:
            movie = await client.get_movie_details(movie_id)
            meta = self.feature_matrix.movie_metadata.get(movie_id, {})
            
            return {
                'id': movie.id,
                'title': movie.title,
                'overview': movie.overview,
                'genres': movie.genres,
                'release_date': movie.release_date,
                'vote_average': movie.vote_average,
                'vote_count': movie.vote_count,
                'popularity': movie.popularity,
                'original_language': movie.original_language,
                'poster_path': movie.poster_path,
                'backdrop_path': movie.backdrop_path,
                'keywords': movie.keywords,
                'credits': movie.credits,
                'watch_providers': movie.watch_providers,
                'similar_movies': self.semantic_index.get(movie_id, [])[:10] if self.semantic_index else []
            }


class BusinessMovieRecommender(MovieRecommender):
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        self.rules_engine = None
        self.session_manager = None
        self.churn_service = None
        self.action_engine = None
        self.event_ingestion = None
        self._business_initialized = False

    async def initialize(self):
        await super().initialize()
        if self._business_initialized:
            return

        from src.business.rules_engine import create_default_rules_engine
        from src.business.session_context import SessionManager
        from src.business.churn_risk import ChurnRiskService
        from src.business.actions import create_default_action_engine
        from src.business.events import create_ingestion_pipeline

        self.rules_engine = create_default_rules_engine()
        self.session_manager = SessionManager()
        self.churn_service = ChurnRiskService()
        self.action_engine = create_default_action_engine(self)
        self.event_ingestion = create_ingestion_pipeline()
        self.event_ingestion.start()

        self._business_initialized = True

    async def get_recommendations_with_business_context(
        self,
        user_id: str,
        partner_id: str,
        watch_history: List[Dict[str, Any]],
        preferences: Dict[str, Any] = None,
        subscriptions: List[str] = None,
        region: str = "US",
        language: str = "en",
        top_k: int = 10,
        natural_language: str = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        await self.initialize()

        preferences = preferences or {}
        subscriptions = subscriptions or []

        session_context = None
        if session_id:
            session = self.session_manager.get_session(session_id)
            if session:
                session_context = session

        if not session_context:
            session_context = self.session_manager.create_session(
                user_id=user_id,
                partner_id=partner_id,
                context_overrides={
                    "region": region,
                    "language": language,
                    "subscriptions": subscriptions,
                    **preferences.get("session_context", {})
                }
            )
            session_id = session_context.session_id

        ratings = {}
        for entry in watch_history:
            movie_id = entry.get('movie_id') or entry.get('id')
            rating = entry.get('rating', 0)
            if rating > 0:
                ratings[movie_id] = rating

        user_profile = self.preference_extractor.extract_preferences(
            watch_history, ratings, self.feature_matrix
        )

        extracted_features = None
        if natural_language:
            context = {
                "subscriptions": subscriptions,
                "region": region,
                "language": language,
                "watch_history_summary": self._summarize_watch_history(watch_history)
            }
            extracted_features = await self.nl_extractor.extract(natural_language, context)

            if self.feature_matrix:
                vocab = set(self.feature_matrix.feature_names)
                extracted_features = self.nl_extractor.map_to_vocabulary(extracted_features, vocab)

        candidate_movies = await self._get_candidate_movies_stage1(
            extracted_features, subscriptions, region, language, user_profile
        )

        if candidate_movies:
            rule_context = self._build_rule_context(session_context, user_profile, extracted_features)
            candidate_movies = self.rules_engine.apply_content_filters(
                candidate_movies, partner_id, rule_context
            )
            base_weights = self.rec_config.get("ranking_weights", {})
            adjusted_weights = self.rules_engine.apply_weight_adjustments(
                base_weights, partner_id, rule_context
            )
            self._apply_weight_adjustments(adjusted_weights)

        if not candidate_movies:
            return self._empty_result(user_profile, session_id, extracted_features)

        query_text = extracted_features.query_text if extracted_features else self._build_query_from_profile(user_profile, preferences)
        query_count, query_tfidf, query_embedding = self.matrix_builder.transform_query(query_text)

        stage1_ranked = self.ranker.rank_movies_stage1(
            candidate_movies, query_embedding, query_tfidf
        )

        stage1_top = self.ranker.get_top_k(stage1_ranked, self.rec_config.get("stage1_candidates", 100))

        final_ranked = self.ranker.rank_movies_stage2(
            stage1_top, user_profile, extracted_features, self.feature_matrix,
            history_boost_factor=self.rec_config.get("history_boost_factor", 0.3)
        )

        final_ranked = self.diversity_filter.filter(final_ranked, self.feature_matrix)
        top_movies = self.ranker.get_top_k(final_ranked, top_k)

        movie_dicts = []
        ranking_factors = []
        for rm in top_movies:
            meta = self.feature_matrix.movie_metadata[rm.movie_id]
            movie_dicts.append({
                'id': rm.movie_id,
                'title': rm.title,
                'overview': rm.overview,
                'genres': rm.genres,
                'vote_average': rm.vote_average,
                'vote_count': rm.vote_count,
                'release_date': meta.get('release_date', ''),
                'director': meta.get('director', []),
                'cast': meta.get('cast', [])
            })
            ranking_factors.append(rm.reasoning_factors)

        watch_history_summary = self._summarize_watch_history(watch_history)
        user_query = natural_language if natural_language else (preferences.get('query', '') if preferences else '')

        reasonings = await self._generate_reasonings(
            movie_dicts, user_profile, ranking_factors, user_query, watch_history_summary, extracted_features
        )

        recommendations = []
        for i, (movie, reasoning) in enumerate(zip(movie_dicts, reasonings)):
            recommendations.append(RecommendationResult(
                movie=movie,
                reasoning=reasoning.reasoning,
                score=top_movies[i].score,
                rank=i + 1
            ))

        self._ingest_recommendation_events(
            partner_id, user_id, session_id, movie_dicts, ranking_factors
        )

        triggered_actions = self.rules_engine.get_triggered_actions(partner_id, rule_context)
        for action in triggered_actions:
            self.action_engine.create_action(
                action_type=action.get("type", "recommend_movie"),
                partner_id=partner_id,
                user_id=user_id,
                trigger="session_start",
                payload=action.get("payload", {}),
                reasoning=action.get("reasoning", "Business rule triggered"),
                priority="medium"
            )

        return {
            'recommendations': [
                {
                    'rank': r.rank,
                    'movie': r.movie,
                    'reasoning': r.reasoning,
                    'score': round(r.score, 4)
                }
                for r in recommendations
            ],
            'user_profile': {
                'top_genres': user_profile.top_genres[:5],
                'top_keywords': user_profile.top_keywords[:10],
                'avg_rating': user_profile.avg_rating,
                'watch_count': user_profile.watch_count
            },
            'extracted_features': extracted_features.to_dict() if extracted_features else None,
            'session_id': session_id,
            'business_actions_triggered': len(triggered_actions)
        }

    def _build_rule_context(
        self,
        session_context,
        user_profile,
        extracted_features
    ) -> Dict[str, Any]:
        context = {
            "partner_id": session_context.partner_id if session_context else "default",
            "user_id": session_context.user_id if session_context else "unknown",
            "context": {}
        }
        if session_context:
            context["context"] = {
                "party_type": session_context.party_type.value,
                "party_size": session_context.party_size,
                "device_type": session_context.device_type.value,
                "flight_duration_minutes": session_context.flight_duration_minutes,
                "hotel_room_type": session_context.hotel_room_type,
                "subscription_tier": session_context.subscription_tier,
                "time_of_day": session_context.time_of_day,
                "day_of_week": session_context.day_of_week,
                "offline_mode": session_context.offline_mode,
                "bandwidth_mbps": session_context.bandwidth_mbps,
                "accessibility": session_context.accessibility,
                "region": session_context.region
            }
        if extracted_features:
            context["context"]["extracted_genres"] = extracted_features.genres
            context["context"]["extracted_decades"] = extracted_features.decades
            context["context"]["extracted_mood"] = extracted_features.mood
        return context

    def _apply_weight_adjustments(self, weights: Dict[str, float]):
        if not weights:
            return
        for key, value in weights.items():
            if hasattr(self.ranker.config, key):
                setattr(self.ranker.config, key, value)

    def _ingest_recommendation_events(
        self,
        partner_id: str,
        user_id: str,
        session_id: str,
        movie_dicts: List[Dict],
        ranking_factors: List[Dict]
    ):
        from src.business.events import build_event, EventType

        events = []
        for i, (movie, factors) in enumerate(zip(movie_dicts, ranking_factors)):
            events.append(build_event(
                EventType.RECOMMENDATION_IMPRESSION,
                partner_id, user_id, session_id,
                {"recommendation_id": movie['id'], "position": i, "score": factors.get('imdb', 0)}
            ))

        if events:
            self.event_ingestion.ingest_batch(events)

    def _empty_result(self, user_profile, session_id, extracted_features):
        return {
            'recommendations': [],
            'user_profile': {
                'top_genres': user_profile.top_genres[:5],
                'top_keywords': user_profile.top_keywords[:10],
                'avg_rating': user_profile.avg_rating,
                'watch_count': user_profile.watch_count
            },
            'extracted_features': extracted_features.to_dict() if extracted_features else None,
            'session_id': session_id,
            'business_actions_triggered': 0
        }

    async def assess_churn_risk(self, user_id: str, partner_id: str, user_data: Dict = None) -> Dict:
        await self.initialize()
        data = user_data or {}
        data["user_id"] = user_id
        data["partner_id"] = partner_id
        prediction = self.churn_service.assess_user(data)
        return {
            "risk_score": prediction.risk_score,
            "risk_level": prediction.risk_level.value,
            "top_risk_factors": prediction.top_risk_factors,
            "recommended_actions": [a.value for a in prediction.recommended_actions],
            "confidence": prediction.confidence,
            "predicted_churn_date": prediction.predicted_churn_date,
            "estimated_ltv_loss": prediction.estimated_ltv_loss
        }

    async def trigger_business_action(
        self,
        action_type: str,
        partner_id: str,
        user_id: str,
        trigger: str,
        payload: Dict,
        reasoning: str,
        priority: str = "medium"
    ) -> Dict:
        await self.initialize()
        from src.business.actions import ActionType, TriggerType, ActionPriority
        action = self.action_engine.create_action(
            action_type=ActionType(action_type),
            partner_id=partner_id,
            user_id=user_id,
            trigger=TriggerType(trigger),
            payload=payload,
            reasoning=reasoning,
            priority=ActionPriority(priority)
        )
        result = self.action_engine.execute_action(action)
        return {"action_id": action.action_id, "result": result}

    async def ingest_events(self, events: List[Dict]) -> Dict:
        await self.initialize()
        from src.business.events import build_event, EventType

        analytics_events = []
        for e in events:
            event = build_event(
                event_type=EventType(e["event_type"]),
                partner_id=e["partner_id"],
                user_id=e["user_id"],
                session_id=e.get("session_id"),
                payload=e["payload"],
                metadata=e.get("metadata", {})
            )
            analytics_events.append(event)
        
        count = self.event_ingestion.ingest_batch(analytics_events)
        return {"ingested": count, "total": len(events)}

    async def apply_business_rules(self, partner_id: str, context: Dict, movies: List[Dict]) -> Dict:
        await self.initialize()
        filtered = self.rules_engine.apply_content_filters(movies, partner_id, context)
        eval_result = self.rules_engine.evaluate(partner_id, context)
        return {
            "filtered_movies": filtered,
            "filters_applied": eval_result["filters"],
            "weight_adjustments": eval_result["weight_adjustments"],
            "triggered_actions": eval_result["actions"],
            "warnings": eval_result["warnings"]
        }

    async def get_partner_config(self, partner_id: str) -> Dict:
        await self.initialize()
        config = self.rules_engine.get_partner_config(partner_id)
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

    def shutdown(self):
        if self.event_ingestion:
            self.event_ingestion.stop()


async def main():
    recommender = MovieRecommender()
    await recommender.initialize()
    
    sample_history = [
        {"movie_id": 550, "rating": 9.0},
        {"movie_id": 680, "rating": 8.5},
        {"movie_id": 13, "rating": 8.0},
    ]
    
    result = await recommender.get_recommendations(
        user_id="test_user",
        watch_history=sample_history,
        subscriptions=["netflix", "disney"],
        region="US",
        top_k=5,
        natural_language="I want a funny 90s space comedy with Tom Hanks"
    )
    
    for rec in result['recommendations']:
        print(f"#{rec['rank']}: {rec['movie']['title']} - {rec['reasoning']}", file=sys.stderr)


async def main_business():
    recommender = BusinessMovieRecommender()
    await recommender.initialize()
    
    sample_history = [
        {"movie_id": 550, "rating": 9.0},
        {"movie_id": 680, "rating": 8.5},
        {"movie_id": 13, "rating": 8.0},
    ]
    
    result = await recommender.get_recommendations_with_business_context(
        user_id="test_user",
        partner_id="airline_default",
        watch_history=sample_history,
        subscriptions=["netflix", "disney"],
        region="US",
        top_k=5,
        natural_language="I want a funny 90s space comedy with Tom Hanks"
    )
    
    for rec in result['recommendations']:
        print(f"#{rec['rank']}: {rec['movie']['title']} - {rec['reasoning']}", file=sys.stderr)
    
    recommender.shutdown()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "business":
        asyncio.run(main_business())
    else:
        asyncio.run(main())
