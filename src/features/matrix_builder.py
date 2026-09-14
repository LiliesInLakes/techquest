import sys
import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import yaml


@dataclass
class FeatureMatrix:
    movie_ids: List[int]
    movie_titles: List[str]
    count_matrix: np.ndarray
    tfidf_matrix: np.ndarray
    semantic_embeddings: np.ndarray
    feature_names: List[str]
    tfidf_feature_names: List[str]
    movie_metadata: Dict[int, Dict]


class FeatureMatrixBuilder:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.data_config = self.config.get("data", {})
        self.tmdb_5000_path = self.data_config.get("tmdb_5000_path", "data/tmdb_5000_movies.csv")
        self.tmdb_5000_credits_path = self.data_config.get("tmdb_5000_credits_path", "data/tmdb_5000_credits.csv")
        self.movielens_tags_path = self.data_config.get("movielens_tags_path", "data/genome-tags.csv")
        self.movielens_relevance_path = self.data_config.get("movielens_relevance_path", "data/genome-scores.csv")
        
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.count_vectorizer = CountVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.semantic_model_name = self.config.get("recommendation", {}).get(
            "semantic_model_name", "multi-qa-mpnet-base-dot-v1"
        )
        self.semantic_model = SentenceTransformer(self.semantic_model_name)

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def load_tmdb_5000(self) -> pd.DataFrame:
        movies_df = pd.read_csv(self.tmdb_5000_path)
        credits_df = pd.read_csv(self.tmdb_5000_credits_path)

        # credits CSV also has a 'title' column — drop it to avoid title_x/title_y
        credits_df = credits_df.drop(columns=['title'], errors='ignore')

        # credits CSV keys on 'movie_id', movies on 'id'
        if 'movie_id' in credits_df.columns:
            credits_df = credits_df.rename(columns={'movie_id': 'id'})

        movies_df = movies_df.merge(credits_df, on='id', how='left')
        
        movies_df['keywords'] = movies_df['keywords'].apply(self._parse_json_list)
        movies_df['genres'] = movies_df['genres'].apply(self._parse_json_list)
        movies_df['cast'] = movies_df['cast'].apply(self._parse_json_cast)
        movies_df['crew'] = movies_df['crew'].apply(self._parse_json_crew)
        
        movies_df['director'] = movies_df['crew'].apply(
            lambda x: [c['name'] for c in x if c.get('job') == 'Director']
        )
        movies_df['top_cast'] = movies_df['cast'].apply(
            lambda x: [c['name'] for c in x[:5]]
        )
        
        return movies_df

    def load_movielens_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        tags_df = pd.read_csv(self.movielens_tags_path)
        relevance_df = pd.read_csv(self.movielens_relevance_path)
        
        # Ensure consistent types for merge
        tags_df['tagId'] = tags_df['tagId'].astype(int)
        relevance_df['tagId'] = relevance_df['tagId'].astype(int)
        relevance_df['movieId'] = relevance_df['movieId'].astype(int)
        
        return tags_df, relevance_df

    def _parse_json_list(self, val):
        if pd.isna(val):
            return []
        try:
            import json, ast
            if isinstance(val, str):
                val = val.strip()
                if val.startswith('[') and val.endswith(']'):
                    try:
                        return [item['name'] for item in json.loads(val.replace("'", '"'))]
                    except:
                        try:
                            parsed = ast.literal_eval(val)
                            return [item['name'] for item in parsed if 'name' in item]
                        except:
                            return []
            elif isinstance(val, list):
                return [item['name'] for item in val if 'name' in item]
            return []
        except:
            return []
    
    def _parse_json_crew(self, val):
        if pd.isna(val):
            return []
        try:
            import json, ast
            if isinstance(val, str):
                val = val.strip()
                if val.startswith('[') and val.endswith(']'):
                    try:
                        return json.loads(val.replace("'", '"'))
                    except:
                        try:
                            return ast.literal_eval(val)
                        except:
                            return []
            elif isinstance(val, list):
                return val
            return []
        except:
            return []
    
    def _parse_json_cast(self, val):
        if pd.isna(val):
            return []
        try:
            import json, ast
            if isinstance(val, str):
                val = val.strip()
                if val.startswith('[') and val.endswith(']'):
                    try:
                        return json.loads(val.replace("'", '"'))
                    except:
                        try:
                            return ast.literal_eval(val)
                        except:
                            return []
            elif isinstance(val, list):
                return val
            return []
        except:
            return []

    def build_combined_text(self, movies_df: pd.DataFrame, tags_df: pd.DataFrame, relevance_df: pd.DataFrame) -> pd.DataFrame:
        tmdb_id_to_movielens = self._create_id_mapping(movies_df)

        # Top 20 tags per movie by relevance, then join tag names
        top_relevance = (
            relevance_df.sort_values('relevance', ascending=False)
            .groupby('movieId', as_index=False)
            .head(20)
        )
        merged = top_relevance.merge(tags_df, on='tagId', how='left')

        tag_text_map = merged.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.dropna())).to_dict()
        
        movies_df['movielens_tags'] = movies_df['id'].map(
            lambda x: tag_text_map.get(tmdb_id_to_movielens.get(x, -1), '')
        )
        
        movies_df['combined_features'] = (
            movies_df['overview'].fillna('') + ' ' +
            movies_df['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
            movies_df['genres'].apply(lambda x: ' '.join(x)) + ' ' +
            movies_df['director'].apply(lambda x: ' '.join(x)) + ' ' +
            movies_df['top_cast'].apply(lambda x: ' '.join(x)) + ' ' +
            movies_df['movielens_tags'].fillna('')
        )
        
        return movies_df

    def _create_id_mapping(self, movies_df: pd.DataFrame) -> Dict[int, int]:
        links_path = Path("data/links.csv")
        if links_path.exists():
            links_df = pd.read_csv(links_path)
            return dict(zip(links_df['tmdbId'], links_df['movieId']))
        return {}

    def build_matrices(self, movies_df: pd.DataFrame) -> FeatureMatrix:
        texts = movies_df['combined_features'].tolist()
        movie_ids = movies_df['id'].tolist()
        movie_titles = movies_df['title'].tolist()
        
        print("Building CountVectorizer matrix...", file=sys.stderr)
        count_matrix = self.count_vectorizer.fit_transform(texts)
        feature_names = self.count_vectorizer.get_feature_names_out().tolist()
        
        print("Building TF-IDF matrix...", file=sys.stderr)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
        
        print("Building semantic embeddings...", file=sys.stderr)
        semantic_embeddings = self.semantic_model.encode(texts, show_progress_bar=True, batch_size=32)
        
        movie_metadata = {}
        for _, row in movies_df.iterrows():
            movie_metadata[row['id']] = {
                'title': row['title'],
                'genres': row['genres'],
                'keywords': row['keywords'],
                'overview': row['overview'],
                'vote_average': row.get('vote_average', 0),
                'vote_count': row.get('vote_count', 0),
                'popularity': row.get('popularity', 0),
                'release_date': row.get('release_date', ''),
                'director': row['director'],
                'cast': row['top_cast'],
                'movielens_tags': row.get('movielens_tags', '')
            }
        
        return FeatureMatrix(
            movie_ids=movie_ids,
            movie_titles=movie_titles,
            count_matrix=count_matrix,
            tfidf_matrix=tfidf_matrix,
            semantic_embeddings=semantic_embeddings,
            feature_names=feature_names,
            tfidf_feature_names=tfidf_feature_names,
            movie_metadata=movie_metadata
        )

    def build_semantic_index(self, feature_matrix: FeatureMatrix) -> Dict:
        similarity_matrix = cosine_similarity(feature_matrix.semantic_embeddings)
        
        semantic_index = {}
        for i, movie_id in enumerate(feature_matrix.movie_ids):
            similar_indices = np.argsort(similarity_matrix[i])[::-1][1:51]
            semantic_index[movie_id] = [
                (feature_matrix.movie_ids[idx], similarity_matrix[i][idx])
                for idx in similar_indices
            ]
        
        return semantic_index

    def build_and_save(self) -> FeatureMatrix:
        print("Loading TMDB 5000 dataset...", file=sys.stderr)
        movies_df = self.load_tmdb_5000()
        print(f"Loaded {len(movies_df)} movies from TMDB 5000", file=sys.stderr)
        
        print("Loading MovieLens data...", file=sys.stderr)
        tags_df, relevance_df = self.load_movielens_data()
        print(f"Loaded {len(tags_df)} tags and {len(relevance_df)} relevance scores", file=sys.stderr)
        
        print("Building combined features...", file=sys.stderr)
        movies_df = self.build_combined_text(movies_df, tags_df, relevance_df)
        
        print("Building feature matrices...", file=sys.stderr)
        feature_matrix = self.build_matrices(movies_df)
        
        print("Building semantic similarity index...", file=sys.stderr)
        semantic_index = self.build_semantic_index(feature_matrix)
        
        print("Saving matrices...", file=sys.stderr)
        self._save_feature_matrix(feature_matrix, semantic_index)
        
        print("Done!", file=sys.stderr)
        return feature_matrix

    def _save_feature_matrix(self, feature_matrix: FeatureMatrix, semantic_index: Dict):
        with open(self.output_dir / "feature_matrix.pkl", "wb") as f:
            pickle.dump(feature_matrix, f)
        
        with open(self.output_dir / "semantic_index.pkl", "wb") as f:
            pickle.dump(semantic_index, f)
        
        with open(self.output_dir / "count_vectorizer.pkl", "wb") as f:
            pickle.dump(self.count_vectorizer, f)
        
        with open(self.output_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        print(f"Saved to {self.output_dir}", file=sys.stderr)

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> Tuple['FeatureMatrixBuilder', FeatureMatrix, Dict]:
        builder = cls(config_path)
        output_dir = Path("data/processed")
        
        with open(output_dir / "feature_matrix.pkl", "rb") as f:
            feature_matrix = pickle.load(f)
        
        with open(output_dir / "semantic_index.pkl", "rb") as f:
            semantic_index = pickle.load(f)
        
        with open(output_dir / "count_vectorizer.pkl", "rb") as f:
            builder.count_vectorizer = pickle.load(f)
        
        with open(output_dir / "tfidf_vectorizer.pkl", "rb") as f:
            builder.tfidf_vectorizer = pickle.load(f)
        
        return builder, feature_matrix, semantic_index

    def transform_query(self, query: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        count_vec = self.count_vectorizer.transform([query])
        tfidf_vec = self.tfidf_vectorizer.transform([query])
        semantic_vec = self.semantic_model.encode([query])
        return count_vec, tfidf_vec, semantic_vec


def main():
    builder = FeatureMatrixBuilder()
    builder.build_and_save()


if __name__ == "__main__":
    main()