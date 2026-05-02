import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text
from embeddings import E5Retriever


class HybridRetriever:
    def __init__(self, tfidf_weight=0.4, e5_weight=0.6, min_score=0.20):
        self.tfidf_weight = tfidf_weight
        self.e5_weight = e5_weight
        self.min_score = min_score

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )

        self.e5 = E5Retriever()

        self.context_vectors = None
        self.context_embeddings = None
        self.contexts = None
        self.responses = None

    def fit(self, contexts, responses):
        if contexts is None or len(contexts) == 0:
            raise ValueError("Empty context data provided.")

        print("Training Hybrid Retriever...")

        self.contexts = contexts.reset_index(drop=True)
        self.responses = responses.reset_index(drop=True)

        clean_contexts = []
        valid_indices = []

        for i, c in enumerate(self.contexts):
            if isinstance(c, str) and c.strip():
                cleaned = clean_text(c)
                clean_contexts.append(cleaned)
                valid_indices.append(i)

        if not clean_contexts:
            raise ValueError("No valid context data after cleaning.")

        self.contexts = self.contexts.iloc[valid_indices].reset_index(drop=True)
        self.responses = self.responses.iloc[valid_indices].reset_index(drop=True)

        self.context_vectors = self.vectorizer.fit_transform(clean_contexts)

        # Use cleaned contexts for E5 for consistency
        cleaned_series = self.contexts.apply(clean_text)
        self.e5.fit(cleaned_series, self.responses)
        self.context_embeddings = self.e5.context_embeddings

        print(f"Hybrid Retriever trained. Total vectors: {len(self.contexts)}")

    def _normalize(self, arr):
        min_val = arr.min()
        max_val = arr.max()
        return (arr - min_val) / (max_val - min_val + 1e-8)

    def retrieve(self, query, top_k=5):
        if self.context_vectors is None or self.context_embeddings is None:
            raise ValueError("Model not trained. Call fit() first.")

        if not isinstance(query, str) or not query.strip():
            return []

        query_clean = clean_text(query)

        # TF-IDF
        query_vec = self.vectorizer.transform([query_clean])
        tfidf_scores = cosine_similarity(query_vec, self.context_vectors).flatten()

        # E5
        if hasattr(self.e5, "encode_query"):
            query_embedding = self.e5.encode_query(query_clean)
        else:
            query_embedding = self.e5.model.encode(
                ["query: " + query_clean],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

        e5_scores = cosine_similarity(query_embedding, self.context_embeddings).flatten()

        # Normalize
        tfidf_scores = self._normalize(tfidf_scores)
        e5_scores = self._normalize(e5_scores)

        # Hybrid
        final_scores = (
            self.tfidf_weight * tfidf_scores +
            self.e5_weight * e5_scores
        )

        # Fallback
        if len(final_scores) == 0 or final_scores.max() < self.min_score:
            return self.e5.retrieve(query, top_k)

        top_k = min(top_k, len(final_scores))

        top_indices = np.argpartition(final_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(final_scores[top_indices])[::-1]]

        return [
            {
                "context": self.contexts.iloc[idx],
                "response": self.responses.iloc[idx],
                "score": float(final_scores[idx]),
                "tfidf_score": float(tfidf_scores[idx]),
                "e5_score": float(e5_scores[idx])
            }
            for idx in top_indices
        ]


TfidfRetriever = HybridRetriever