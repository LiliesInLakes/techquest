import sys
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet
import pickle
from pathlib import Path


try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def _get_embedding_dimension(model: Any) -> int:
    """Safely retrieves model embedding dimension without deprecation warnings."""
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()


@dataclass
class SemanticSearchResult:
    keyword: str
    synonyms: List[str]
    similar_keywords: List[Tuple[str, float]]
    embedding: np.ndarray


class SemanticKeywordSearch:
    def __init__(self, model_name: str = 'multi-qa-mpnet-base-dot-v1', threshold: float = 0.7):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.keyword_embeddings: Dict[str, np.ndarray] = {}
        self.keyword_list: List[str] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        self.synonym_cache: Dict[str, List[str]] = {}

    def build_index(self, keywords: List[str]):
        unique_keywords = list(set(k.lower().strip() for k in keywords if k and k.strip()))
        self.keyword_list = unique_keywords
        
        print(f"Building embeddings for {len(unique_keywords)} keywords...", file=sys.stderr)
        embeddings = self.model.encode(unique_keywords, show_progress_bar=True, batch_size=64)
        
        self.keyword_embeddings = {kw: emb for kw, emb in zip(unique_keywords, embeddings)}
        self.embedding_matrix = np.array(embeddings)
        
        print("Keyword index built.", file=sys.stderr)

    def get_synonyms_wordnet(self, keyword: str) -> Set[str]:
        if keyword in self.synonym_cache:
            return set(self.synonym_cache[keyword])
        
        synonyms = set()
        for syn in wordnet.synsets(keyword.replace(' ', '_')):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != keyword.lower():
                    synonyms.add(name.lower())
        
        self.synonym_cache[keyword] = list(synonyms)
        return synonyms

    def find_similar_keywords(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.embedding_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embedding_matrix)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.threshold:
                results.append((self.keyword_list[idx], float(similarities[idx])))
        
        return results

    def expand_query_with_synonyms(self, query: str, max_synonyms: int = 5) -> List[str]:
        words = query.lower().split()
        expanded = set(words)
        
        for word in words:
            synonyms = self.get_synonyms_wordnet(word)
            for syn in list(synonyms)[:max_synonyms]:
                expanded.add(syn)
        
        similar = self.find_similar_keywords(query, top_k=max_synonyms)
        for kw, score in similar:
            if score >= self.threshold:
                expanded.add(kw)
        
        return list(expanded)

    def search(self, query: str, top_k: int = 20) -> SemanticSearchResult:
        synonyms = set()
        for word in query.lower().split():
            synonyms.update(self.get_synonyms_wordnet(word))
        
        similar_keywords = self.find_similar_keywords(query, top_k=top_k)
        
        query_embedding = self.model.encode([query])[0]
        
        return SemanticSearchResult(
            keyword=query,
            synonyms=list(synonyms),
            similar_keywords=similar_keywords,
            embedding=query_embedding
        )

    def get_keyword_embedding(self, keyword: str) -> Optional[np.ndarray]:
        return self.keyword_embeddings.get(keyword.lower())

    def compute_keyword_similarity(self, kw1: str, kw2: str) -> float:
        emb1 = self.get_keyword_embedding(kw1)
        emb2 = self.get_keyword_embedding(kw2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        return float(cosine_similarity([emb1], [emb2])[0][0])

    def save(self, path: str):
        data = {
            'keyword_embeddings': self.keyword_embeddings,
            'keyword_list': self.keyword_list,
            'synonym_cache': self.synonym_cache,
            'threshold': self.threshold
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, model_name: str = 'multi-qa-mpnet-base-dot-v1') -> 'SemanticKeywordSearch':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(model_name=model_name, threshold=data['threshold'])
        instance.keyword_embeddings = data['keyword_embeddings']
        instance.keyword_list = data['keyword_list']
        instance.synonym_cache = data.get('synonym_cache', {})
        instance.embedding_matrix = np.array(list(data['keyword_embeddings'].values()))
        
        expected_dim = _get_embedding_dimension(instance.model)
        if instance.embedding_matrix.shape[1] != expected_dim:
            raise ValueError(
                f"Keyword embedding dimension mismatch: found {instance.embedding_matrix.shape[1]}d, expected {expected_dim}d"
            )
        
        return instance

    @classmethod
    def from_feature_embeddings(
        cls,
        embeddings_path: str,
        feature_names: List[str],
        model_name: str = 'multi-qa-mpnet-base-dot-v1',
        threshold: float = 0.65
    ) -> 'SemanticKeywordSearch':
        """Quickly reconstructs SemanticKeywordSearch from precomputed feature embedding matrix."""
        arr = np.load(embeddings_path)
        instance = cls(model_name=model_name, threshold=threshold)
        cleaned_features = [f.lower().strip() for f in feature_names]
        instance.keyword_list = cleaned_features
        instance.embedding_matrix = arr
        instance.keyword_embeddings = {kw: arr[i] for i, kw in enumerate(cleaned_features)}
        return instance


class HybridKeywordMatcher:
    def __init__(
        self,
        semantic_search: SemanticKeywordSearch,
        feature_names: List[str],
        cache_path: str = "data/processed/keyword_feature_embeddings.npy"
    ):
        self.semantic_search = semantic_search
        self.feature_names = [f.lower() for f in feature_names]
        self.cache_path = Path(cache_path)
        self.feature_embeddings = None
        
        self._load_or_build_feature_embeddings()

    def _load_or_build_feature_embeddings(self):
        expected_count = len(self.feature_names)
        expected_dim = _get_embedding_dimension(self.semantic_search.model)
        
        # 1. Try to load cached embeddings
        if self.cache_path.exists():
            try:
                arr = np.load(self.cache_path)
                if (
                    arr.shape[0] == expected_count
                    and arr.shape[1] == expected_dim
                    and arr.dtype == np.float32
                    and not np.isnan(arr).any()
                ):
                    self.feature_embeddings = arr
                    return
            except Exception as e:
                print(f"Failed to load keyword feature cache ({e}), rebuilding...", file=sys.stderr)
        
        # 2. Rebuild if missing or incompatible
        print(f"Building embeddings for {expected_count} features ({expected_dim}d)...", file=sys.stderr)
        embeddings = self.semantic_search.model.encode(
            self.feature_names,
            show_progress_bar=False,
            batch_size=128,
            normalize_embeddings=True
        ).astype(np.float32)
        
        self.feature_embeddings = embeddings
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, embeddings)
        except Exception as e:
            print(f"Could not persist keyword feature cache ({e})", file=sys.stderr)

    def match_user_keywords(self, user_keywords: List[str], top_k: int = 50) -> Dict[str, List[Tuple[str, float]]]:
        results = {}
        
        for kw in user_keywords:
            expanded = self.semantic_search.expand_query_with_synonyms(kw)
            
            kw_embedding = self.semantic_search.model.encode([kw])[0]
            similarities = cosine_similarity([kw_embedding], self.feature_embeddings)[0]
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            matched_features = []
            for idx in top_indices:
                if similarities[idx] >= self.semantic_search.threshold:
                    matched_features.append((self.feature_names[idx], float(similarities[idx])))
            
            results[kw] = matched_features
        
        return results

    def get_feature_importance_for_keywords(self, user_keywords: List[str]) -> Dict[str, float]:
        matched = self.match_user_keywords(user_keywords)
        
        feature_scores = {}
        for kw, matches in matched.items():
            for feature, score in matches:
                if feature not in feature_scores or score > feature_scores[feature]:
                    feature_scores[feature] = score
        
        return feature_scores


def main():
    sample_keywords = [
        "action", "adventure", "comedy", "drama", "thriller", "horror",
        "romance", "sci-fi", "fantasy", "animation", "superhero", "space",
        "time travel", "artificial intelligence", "robot", "alien",
        "detective", "murder mystery", "heist", "war", "historical",
        "biography", "musical", "family", "kids", "teen", "coming of age",
        "revenge", "redemption", "survival", "post-apocalyptic", "dystopian",
        "cyberpunk", "steampunk", "noir", "western", "martial arts",
        "sports", "dance", "music", "fashion", "food", "travel"
    ]
    
    search = SemanticKeywordSearch(threshold=0.65)
    search.build_index(sample_keywords)
    
    query = "funny space adventure"
    result = search.search(query)
    
    print(f"Query: {query}", file=sys.stderr)
    print(f"Synonyms: {result.synonyms[:10]}", file=sys.stderr)
    print(f"Similar keywords: {result.similar_keywords[:10]}", file=sys.stderr)
    
    search.save("data/processed/semantic_keyword_index.pkl")


if __name__ == "__main__":
    main()