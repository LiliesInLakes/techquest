import asyncio
import sys
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import yaml

from src.tmdb.client import TMDBClient, Movie
from src.features.matrix_builder import FeatureMatrixBuilder, FeatureMatrix
from src.features.semantic_search import SemanticKeywordSearch, HybridKeywordMatcher
from src.features.preference_extractor import PreferenceExtractor, UserPreferenceProfile
from src.ranking.ranker import MovieRanker, RankingConfig, RankedMovie, DiversityFilter
from src.llm.reasoning import DeepSeekReasoner, ReasoningTemplateEngine, RecommendationReasoning
from src.llm.nl_feature_extractor import NLFeatureExtractor, ExtractedFeatures
from src.memory.service import UserMemoryService


@dataclass
class RecommendationResult:
    movie: Dict[str, Any]
    reasoning: str
    score: float
    rank: int


def _sanitize_keywords(kw_data: Any) -> List[str]:
    if not kw_data:
        return []
    if isinstance(kw_data, str):
        kw_data = kw_data.strip()
        if not kw_data:
            return []
        try:
            parsed = json.loads(kw_data)
            if isinstance(parsed, list):
                return [str(k).strip().strip('[]"\'') for k in parsed if str(k).strip().strip('[]"\'')]
        except Exception:
            pass
        return [k.strip().strip('[]"\'') for k in kw_data.replace('|', ',').split(',') if k.strip().strip('[]"\'')]
    if isinstance(kw_data, list):
        cleaned = []
        for item in kw_data:
            if isinstance(item, str) and item.strip().startswith('[') and item.strip().endswith(']'):
                try:
                    p = json.loads(item)
                    if isinstance(p, list):
                        cleaned.extend([str(k).strip().strip('[]"\'') for k in p if str(k).strip().strip('[]"\'')])
                        continue
                except Exception:
                    pass
            c_str = str(item).strip().strip('[]"\'')
            if c_str:
                cleaned.append(c_str)
        return cleaned
    return []


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
        self.memory_service: Optional[UserMemoryService] = None
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
            self.matrix_builder, self.feature_matrix, _ = \
                FeatureMatrixBuilder.load(self.config_path)
            print("Loaded precomputed feature matrix", file=sys.stderr)
        except (FileNotFoundError, ValueError) as e:
            print(f"Precomputed matrix issue ({e}), building from {self.matrix_builder.db_path}...", file=sys.stderr)
            self.feature_matrix = self.matrix_builder.build_and_save()
        self.semantic_index = None
        
        model_name = self.rec_config.get("semantic_model_name", "multi-qa-mpnet-base-dot-v1")
        kw_path = Path("data/processed/keyword_embeddings.pkl")
        if kw_path.exists():
            try:
                self.semantic_search = SemanticKeywordSearch.load(str(kw_path), model_name=model_name)
            except Exception:
                self.semantic_search = SemanticKeywordSearch(
                    model_name=model_name,
                    threshold=self.rec_config.get("semantic_similarity_threshold", 0.65)
                )
                self.semantic_search.build_index(self.feature_matrix.feature_names)
                self.semantic_search.save(str(kw_path))
        else:
            self.semantic_search = SemanticKeywordSearch(
                model_name=model_name,
                threshold=self.rec_config.get("semantic_similarity_threshold", 0.65)
            )
            self.semantic_search.build_index(self.feature_matrix.feature_names)
            self.semantic_search.save(str(kw_path))
        
        self.hybrid_matcher = HybridKeywordMatcher(
            self.semantic_search, self.feature_matrix.feature_names
        )
        
        self.preference_extractor = PreferenceExtractor(
            feature_matrix=self.feature_matrix,
            semantic_search=self.semantic_search
        )
        
        self.memory_service = UserMemoryService(
            db_path=self.config.get("memory", {}).get("db_path", "data/memory.db"),
            feature_matrix=self.feature_matrix
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
        watch_history: List[Dict[str, Any]] = None,
        preferences: Dict[str, Any] = None,
        subscriptions: List[str] = None,
        region: str = "US",
        language: str = "en",
        top_k: Optional[int] = None,
        natural_language: str = None,
        excluded_movie_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        await self.initialize()
        if top_k is None:
            top_k = int(self.rec_config.get("final_recommendations", 1))
        
        preferences = preferences or {}
        subscriptions = subscriptions or []
        watch_history = watch_history or []
        
        if watch_history:
            ratings = {}
            for entry in watch_history:
                movie_id = entry.get('movie_id') or entry.get('id')
                rating = entry.get('rating', 0)
                if rating > 0:
                    ratings[movie_id] = rating
            
            user_profile = self.preference_extractor.extract_preferences(
                watch_history, ratings, self.feature_matrix
            )
            if self.memory_service:
                persisted_prof = self.memory_service.get_combined_preference_profile(user_id)
                if hasattr(persisted_prof, "negative_preferences"):
                    setattr(user_profile, "negative_preferences", getattr(persisted_prof, "negative_preferences"))
        else:
            if self.memory_service:
                user_profile = self.memory_service.get_combined_preference_profile(user_id)
            else:
                user_profile = self.preference_extractor.extract_preferences([], {}, self.feature_matrix)
        
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
        
        resolved_ref = None
        if extracted_features and extracted_features.reference_movies:
            for ref_name in extracted_features.reference_movies:
                resolved_ref = await self._resolve_reference_movie(ref_name)
                if resolved_ref:
                    break

        candidate_movies = await self._get_candidate_movies_stage1(
            extracted_features, subscriptions, region, language, user_profile, resolved_ref=resolved_ref
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

        if resolved_ref and resolved_ref.get('idx') is not None and self.feature_matrix is not None and hasattr(self.feature_matrix, "semantic_embeddings"):
            ref_emb = self.feature_matrix.semantic_embeddings[resolved_ref['idx']].reshape(1, -1)
            if query_text and any(w in query_text.lower() for w in ["dark", "funny", "scary", "thriller", "action", "horror", "sci-fi", "space", "animation"]):
                query_embedding = 0.75 * ref_emb + 0.25 * query_embedding
                norm = np.linalg.norm(query_embedding)
                if norm > 0:
                    query_embedding = query_embedding / norm
            else:
                query_embedding = ref_emb
        
        stage1_ranked = self.ranker.rank_movies_stage1(
            candidate_movies, query_embedding, query_tfidf
        )
        
        stage1_top = self.ranker.get_top_k(stage1_ranked, self.rec_config.get("stage1_candidates", 100))
        
        final_ranked = self.ranker.rank_movies_stage2(
            stage1_top, user_profile, extracted_features, self.feature_matrix,
            history_boost_factor=self.rec_config.get("history_boost_factor", 0.3)
        )
        
        final_ranked = self.diversity_filter.filter(final_ranked, self.feature_matrix)
        excluded_set = set(excluded_movie_ids) if excluded_movie_ids else set()
        if resolved_ref and resolved_ref.get('id'):
            excluded_set.add(resolved_ref['id'])
        if excluded_set:
            final_ranked = [m for m in final_ranked if m.movie_id not in excluded_set]
        top_movies = self.ranker.get_top_k(final_ranked, top_k)
        
        movie_dicts = []
        ranking_factors = []
        candidate_lookup = {c.get('id') or c.get('movie_id'): c for c in candidate_movies}
        for rm in top_movies:
            meta = self.feature_matrix.movie_metadata.get(rm.movie_id, {}) if self.feature_matrix and hasattr(self.feature_matrix, "movie_metadata") else {}
            cand = candidate_lookup.get(rm.movie_id, {})
            
            genres_list = rm.genres or cand.get('genres', []) or meta.get('genres', [])
            
            # Genuine streaming platforms if available from real candidate data, otherwise empty list
            platforms = []
            if cand.get('streaming_platforms') and isinstance(cand.get('streaming_platforms'), list):
                platforms = list(cand.get('streaming_platforms'))
            elif cand.get('watch_providers') and isinstance(cand.get('watch_providers'), dict):
                wp = cand.get('watch_providers')
                flatrate = wp.get('flatrate') or []
                platforms = [p.get('provider_name') for p in flatrate if isinstance(p, dict) and p.get('provider_name')]
            platforms = list(dict.fromkeys([p for p in platforms if isinstance(p, str) and p.strip()]))

            movie_dicts.append({
                'id': rm.movie_id,
                'title': rm.title,
                'overview': rm.overview or cand.get('overview', '') or meta.get('overview', ''),
                'genres': genres_list,
                'vote_average': rm.vote_average,
                'vote_count': rm.vote_count,
                'release_date': cand.get('release_date', '') or meta.get('release_date', ''),
                'director': cand.get('director', []) or meta.get('director', []),
                'cast': cand.get('cast', []) or meta.get('cast', []),
                'keywords': _sanitize_keywords(cand.get('keywords') or meta.get('keywords', [])),
                'streaming_platforms': platforms,
                'poster_path': cand.get('poster_path') or meta.get('poster_path') or None,
                'original_language': cand.get('original_language') or meta.get('original_language', ''),
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
        
        top_pick_verdict = None
        if movie_dicts:
            top_pick_verdict = await self._generate_top_pick_verdict(
                movie_dicts[0],
                user_profile,
                ranking_factors[0] if ranking_factors else {},
                user_query,
                watch_history_summary
            )

        single_rec = {
            'rank': 1,
            'title': recommendations[0].movie.get('title'),
            'movie': recommendations[0].movie,
            'reasoning': recommendations[0].reasoning,
            'verdict': top_pick_verdict,
            'score': round(recommendations[0].score, 4)
        } if recommendations else None

        return {
            'recommendations': [
                {
                    'rank': r.rank,
                    'movie': r.movie,
                    'reasoning': r.reasoning,
                    'verdict': top_pick_verdict if r.rank == 1 else None,
                    'score': round(r.score, 4)
                }
                for r in recommendations
            ],
            'recommendation': single_rec,
            'top_pick': single_rec,
            'user_profile': {
                'top_genres': user_profile.top_genres[:5],
                'top_keywords': user_profile.top_keywords[:10],
                'top_tags': getattr(user_profile, 'top_tags', [])[:10],
                'top_themes': getattr(user_profile, 'top_themes', [])[:10],
                'top_tones': getattr(user_profile, 'top_tones', [])[:10],
                'negative_preferences': getattr(user_profile, 'negative_preferences', {}),
                'avg_rating': user_profile.avg_rating,
                'watch_count': user_profile.watch_count
            },
            'extracted_features': extracted_features.to_dict() if extracted_features else None
        }

    async def get_replacement_recommendation(
        self,
        user_id: str,
        excluded_movie_ids: List[int],
        natural_language: Optional[str] = None,
        subscriptions: List[str] = None,
        region: str = "US",
        language: str = "en"
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches the next highest ranked recommendation that is not in excluded_movie_ids.
        Used for background replacement when a user skips / rejects a movie.
        """
        recs_data = await self.get_recommendations(
            user_id=user_id,
            subscriptions=subscriptions or [],
            region=region,
            language=language,
            top_k=1,
            natural_language=natural_language,
            excluded_movie_ids=excluded_movie_ids
        )
        recs = recs_data.get("recommendations", [])
        if recs:
            rec_item = recs[0]
            if recs_data.get("top_pick") and recs_data["top_pick"].get("verdict"):
                rec_item["verdict"] = recs_data["top_pick"]["verdict"]
            return rec_item
        return None

    async def _resolve_reference_movie(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Resolves a reference movie title against canonical SQLite database or TMDB search.
        Resolution policy:
        1. Exact match (case-insensitive).
        2. Normalized title match (removing trailing/leading articles, release years, and punctuation).
        3. Franchise/Prefix match with popularity disambiguation (e.g. 'Harry Potter' -> highest popularity canonical entry).
        4. TMDB live search fallback.
        """
        if not title or not self.feature_matrix or not hasattr(self.feature_matrix, "movie_metadata"):
            return None

        title_clean = title.strip().lower()
        metadata = self.feature_matrix.movie_metadata

        # 1. Exact title match
        for mid, meta in metadata.items():
            t = (meta.get('title') or '').strip().lower()
            if t == title_clean:
                idx = self.feature_matrix.tmdb_id_to_idx.get(mid)
                return {'id': mid, 'title': meta.get('title'), 'meta': meta, 'idx': idx}

        # 2. Normalized title match
        def normalize_title(t_str: str) -> str:
            t_str = re.sub(r'\s*\([^)]*\)', '', t_str)  # remove parenthesized years or alternate titles
            t_str = t_str.lower().strip()
            if t_str.endswith(', the'):
                t_str = 'the ' + t_str[:-5].strip()
            elif t_str.endswith(', a'):
                t_str = 'a ' + t_str[:-3].strip()
            t_str = re.sub(r'[^a-z0-9\s]', ' ', t_str)
            return ' '.join(t_str.split())

        norm_target = normalize_title(title)
        for mid, meta in metadata.items():
            norm_db = normalize_title(meta.get('title') or '')
            if norm_db == norm_target:
                idx = self.feature_matrix.tmdb_id_to_idx.get(mid)
                return {'id': mid, 'title': meta.get('title'), 'meta': meta, 'idx': idx}

        # 3. Franchise / Prefix match with popularity disambiguation
        prefix_matches = []
        for mid, meta in metadata.items():
            norm_db = normalize_title(meta.get('title') or '')
            if norm_db.startswith(norm_target) or (norm_target in norm_db and len(norm_target) >= 5):
                idx = self.feature_matrix.tmdb_id_to_idx.get(mid)
                if idx is not None:
                    prefix_matches.append({
                        'id': mid,
                        'title': meta.get('title'),
                        'meta': meta,
                        'idx': idx,
                        'vote_count': meta.get('vote_count', 0),
                        'popularity': meta.get('popularity', 0.0)
                    })
        if prefix_matches:
            prefix_matches.sort(key=lambda x: (x['vote_count'], x['popularity']), reverse=True)
            best = prefix_matches[0]
            return {'id': best['id'], 'title': best['title'], 'meta': best['meta'], 'idx': best['idx']}

        # 4. TMDB title search fallback
        if self.tmdb_client is not None:
            try:
                async with self.tmdb_client as client:
                    results = await client.search_movies(title, page=1)
                    for m in results:
                        if m.id in self.feature_matrix.tmdb_id_to_idx:
                            idx = self.feature_matrix.tmdb_id_to_idx[m.id]
                            meta = metadata.get(m.id, {})
                            return {'id': m.id, 'title': meta.get('title') or m.title, 'meta': meta, 'idx': idx}
            except Exception:
                pass

        return None

    async def _get_candidate_movies_stage1(
        self,
        extracted_features: Optional[ExtractedFeatures],
        subscriptions: List[str],
        region: str,
        language: str,
        user_profile: UserPreferenceProfile,
        resolved_ref: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # Stage 1: Candidate discovery.
        # If a reference movie is resolved, similarity candidates are derived directly from the reference
        # movie's precomputed 768-d semantic embedding, filtered by hard constraints, and excluding the reference movie itself.
        sqlite_metadata = self.feature_matrix.movie_metadata if self.feature_matrix else {}
        sqlite_id_set = set(self.feature_matrix.tmdb_id_to_idx.keys()) if (self.feature_matrix and hasattr(self.feature_matrix, "tmdb_id_to_idx")) else set(sqlite_metadata.keys())

        if resolved_ref and resolved_ref.get('idx') is not None and self.feature_matrix is not None and hasattr(self.feature_matrix, "semantic_embeddings"):
            ref_id = resolved_ref['id']
            ref_idx = resolved_ref['idx']
            ref_emb = self.feature_matrix.semantic_embeddings[ref_idx]

            sims = np.dot(ref_emb, self.feature_matrix.semantic_embeddings.T)
            top_indices = np.argsort(sims)[::-1]

            candidates = []
            seen_ids = {ref_id}

            for ti in top_indices:
                cand_id = self.feature_matrix.idx_to_tmdb_id.get(ti, self.feature_matrix.movie_ids[ti])
                if cand_id in seen_ids:
                    continue
                seen_ids.add(cand_id)
                meta = sqlite_metadata.get(cand_id, {})
                movie = {
                    'id': cand_id,
                    'title': meta.get('title', ''),
                    'overview': meta.get('overview', ''),
                    'genres': meta.get('genres', []),
                    'release_date': meta.get('release_date', ''),
                    'vote_average': meta.get('vote_average', 0.0),
                    'vote_count': meta.get('vote_count', 0),
                    'popularity': meta.get('popularity', 0),
                    'original_language': meta.get('original_language', 'en'),
                    'poster_path': meta.get('poster_path') or None
                }
                if self._passes_all_hard_filters(movie, extracted_features):
                    candidates.append(movie)
                    if len(candidates) >= self.rec_config.get("stage1_candidates", 100):
                        break

            return candidates

        if extracted_features:
            if self.tmdb_client is None:
                return self._get_local_candidates(extracted_features)
            try:
                async with self.tmdb_client as client:
                    tmdb_params = self.nl_extractor.build_tmdb_discover_params(
                        extracted_features, subscriptions, region, language
                    )
                    
                    total_tmdb_candidates = 0
                    candidates = []
                    seen_ids = set()

                    # Fetch TMDB discovery candidates (up to 5 pages if needed to satisfy candidate target)
                    max_pages = 5
                    for page_num in range(1, max_pages + 1):
                        page_params = tmdb_params.copy()
                        if page_num > 1:
                            page_params["page"] = page_num
                            
                        page_movies = await client.discover_movies(**page_params)
                        if not page_movies:
                            break
                            
                        total_tmdb_candidates += len(page_movies)
                        
                        for m in page_movies:
                            if m.id in seen_ids:
                                continue
                            seen_ids.add(m.id)
                            
                            # Strict SQLite membership filter: candidate MUST exist in movie_features.db
                            if m.id not in sqlite_id_set:
                                continue
                                
                            meta = sqlite_metadata.get(m.id, {})
                            cand_dict = {
                                'id': m.id,
                                'title': meta.get('title') or m.title,
                                'overview': meta.get('overview') or m.overview,
                                'genres': meta.get('genres') or m.genres,
                                'release_date': meta.get('release_date') or m.release_date,
                                'vote_average': meta.get('vote_average') if meta.get('vote_average', 0) > 0 else m.vote_average,
                                'vote_count': meta.get('vote_count') if meta.get('vote_count', 0) > 0 else m.vote_count,
                                'popularity': m.popularity or meta.get('popularity', 0),
                                'original_language': m.original_language or meta.get('original_language', 'en'),
                                'poster_path': m.poster_path
                            }
                            if self._passes_all_hard_filters(cand_dict, extracted_features):
                                candidates.append(cand_dict)
                                
                        if len(candidates) >= self.rec_config.get("stage1_candidates", 50):
                            break

                    matched_count = len(candidates)
                    excluded_count = total_tmdb_candidates - matched_count
                    coverage_pct = (matched_count / total_tmdb_candidates * 100.0) if total_tmdb_candidates > 0 else 0.0
                    print(
                        f"[Candidate Discovery] TMDB candidates: {total_tmdb_candidates} | "
                        f"SQLite matches: {matched_count} | Excluded (out-of-database): {excluded_count} | "
                        f"Coverage: {coverage_pct:.1f}%",
                        file=sys.stderr
                    )

                    # If still insufficient after TMDB pagination, top up with SQLite local fallback matching hard constraints
                    if len(candidates) < 50:
                        fallback_pool = self._get_local_candidates(extracted_features)
                        for fb in fallback_pool:
                            if fb['id'] not in seen_ids:
                                candidates.append(fb)
                                seen_ids.add(fb['id'])
                            if len(candidates) >= self.rec_config.get("stage1_candidates", 100):
                                break

                    if candidates:
                        return candidates[:self.rec_config.get("stage1_candidates", 100)]
            except Exception as e:
                print(f"TMDB unavailable ({e}), falling back to local candidate pool", file=sys.stderr)

            return self._get_local_candidates(extracted_features)
        else:
            if self.tmdb_client is None:
                return self._get_local_candidates(None)
            try:
                return await self._get_candidate_movies(subscriptions, region, language, user_profile)
            except Exception as e:
                print(f"TMDB unavailable ({e}), falling back to local candidate pool", file=sys.stderr)
                return self._get_local_candidates(None)

    def _get_local_candidates(
        self, extracted_features: Optional[ExtractedFeatures]
    ) -> List[Dict[str, Any]]:
        # Offline fallback candidate pool directly from movie_features.db / feature_matrix
        # Applies all extracted explicit hard constraints
        candidates = []
        if not self.feature_matrix or not hasattr(self.feature_matrix, "movie_metadata"):
            return candidates

        for movie_id, meta in self.feature_matrix.movie_metadata.items():
            movie = {
                'id': movie_id,
                'title': meta.get('title', ''),
                'overview': meta.get('overview', ''),
                'genres': meta.get('genres', []),
                'release_date': meta.get('release_date', ''),
                'vote_average': meta.get('vote_average', 0.0),
                'vote_count': meta.get('vote_count', 0),
                'popularity': meta.get('popularity', 0),
                'original_language': meta.get('original_language', 'en'),
            }
            if self._passes_all_hard_filters(movie, extracted_features):
                candidates.append(movie)

        candidates.sort(key=lambda c: (c.get('vote_count', 0), c.get('vote_average', 0)), reverse=True)
        return candidates

    @staticmethod
    def _normalize_genre_name(g: str) -> str:
        g_low = str(g).lower().strip()
        if g_low in {"sci-fi", "scifi", "science fiction", "science-fiction"}:
            return "science fiction"
        if g_low in {"children", "kids", "family"}:
            return "family"
        if g_low in {"animated", "anime", "cartoon", "animation"}:
            return "animation"
        if g_low in {"romcom", "rom-com", "romantic comedy"}:
            return "comedy"
        return g_low

    def _passes_all_hard_filters(self, movie: Any, extracted_features: Optional[ExtractedFeatures]) -> bool:
        if not extracted_features:
            return True

        if not self._passes_exclusions(movie, extracted_features):
            return False

        if isinstance(movie, dict):
            genres = movie.get('genres', [])
            release_date = movie.get('release_date', '')
            original_language = movie.get('original_language', 'en')
            vote_average = movie.get('vote_average', 0.0)
        else:
            genres = getattr(movie, 'genres', [])
            release_date = getattr(movie, 'release_date', '')
            original_language = getattr(movie, 'original_language', 'en')
            vote_average = getattr(movie, 'vote_average', 0.0)

        # Explicit genres hard constraint (if explicitly requested, movie must match)
        if extracted_features.explicit_genres:
            req_genres = {self._normalize_genre_name(g) for g in extracted_features.explicit_genres}
            movie_genres = {self._normalize_genre_name(g) for g in genres}
            if not req_genres.intersection(movie_genres):
                return False

        # Release year constraints
        if release_date and len(str(release_date)) >= 4:
            try:
                year = int(str(release_date)[:4])
                if extracted_features.release_year_min and year < extracted_features.release_year_min:
                    return False
                if extracted_features.release_year_max and year > extracted_features.release_year_max:
                    return False
            except ValueError:
                pass
        elif extracted_features.release_year_min:
            # If movie has no valid release date and a minimum year is required, exclude it
            return False

        # Language constraint
        if extracted_features.original_language and extracted_features.original_language != "en":
            if original_language != extracted_features.original_language:
                return False

        # Explicit min rating (only if explicitly extracted/requested by user)
        if extracted_features.min_rating and extracted_features.min_rating > 0.0:
            if vote_average < extracted_features.min_rating:
                return False

        return True

    def _passes_exclusions(self, movie, extracted_features: ExtractedFeatures) -> bool:
        if not extracted_features:
            return True

        if isinstance(movie, dict):
            genres = movie.get('genres', [])
            overview = movie.get('overview', '')
        else:
            genres = movie.genres
            overview = movie.overview

        movie_genres_norm = {self._normalize_genre_name(g) for g in genres}
        for excl in extracted_features.exclude_genres:
            if self._normalize_genre_name(excl) in movie_genres_norm:
                return False

        if isinstance(overview, str):
            movie_overview_lower = overview.lower()
        else:
            movie_overview_lower = ""

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
        if self.tmdb_client is None:
            return self._get_local_candidates(None)

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
                        'original_language': m.original_language,
                        'poster_path': m.poster_path
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
                    'original_language': m.original_language,
                    'poster_path': m.poster_path
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

    async def _generate_top_pick_verdict(
        self,
        top_movie: Dict[str, Any],
        user_profile: UserPreferenceProfile,
        ranking_factors: Dict[str, float],
        user_query: str,
        watch_history_summary: str
    ) -> str:
        if not top_movie:
            return ""
        if self.reasoner:
            try:
                return await self.reasoner.generate_top_pick_verdict(
                    top_movie, user_profile, ranking_factors, user_query, watch_history_summary
                )
            except Exception:
                pass
        return self.template_engine.generate_top_pick(
            top_movie, user_profile, ranking_factors, user_query
        )

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
        
        meta = self.feature_matrix.movie_metadata.get(movie_id, {}) if self.feature_matrix else {}
        similar = []
        if self.feature_matrix and hasattr(self.feature_matrix, "semantic_embeddings"):
            if hasattr(self.feature_matrix, "tmdb_id_to_idx") and self.feature_matrix.tmdb_id_to_idx:
                idx = self.feature_matrix.tmdb_id_to_idx.get(movie_id)
            else:
                idx = self.feature_matrix.movie_ids.index(movie_id) if movie_id in self.feature_matrix.movie_ids else None
                
            if idx is not None:
                sims = np.dot(self.feature_matrix.semantic_embeddings[idx], self.feature_matrix.semantic_embeddings.T)
                top_indices = np.argsort(sims)[::-1][1:11]
                for ti in top_indices:
                    sim_id = self.feature_matrix.idx_to_tmdb_id.get(ti, self.feature_matrix.movie_ids[ti])
                    similar.append((sim_id, float(sims[ti])))
        
        if self.tmdb_client is not None:
            try:
                async with self.tmdb_client as client:
                    movie = await client.get_movie_details(movie_id)
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
                        'similar_movies': similar
                    }
            except Exception as e:
                print(f"TMDB details fetch failed ({e}), using local metadata", file=sys.stderr)
        
        return {
            'id': movie_id,
            'title': meta.get('title', ''),
            'overview': meta.get('overview', ''),
            'genres': meta.get('genres', []),
            'release_date': meta.get('release_date', ''),
            'vote_average': meta.get('vote_average', 0),
            'vote_count': meta.get('vote_count', 0),
            'popularity': meta.get('popularity', 0),
            'original_language': meta.get('original_language', 'en'),
            'poster_path': None,
            'backdrop_path': None,
            'keywords': _sanitize_keywords(meta.get('keywords', [])),
            'credits': {'cast': meta.get('cast', []), 'crew': meta.get('director', [])},
            'watch_providers': {},
            'similar_movies': similar
        }


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


if __name__ == "__main__":
    asyncio.run(main())
