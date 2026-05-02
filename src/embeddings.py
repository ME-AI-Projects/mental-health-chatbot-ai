from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

from preprocessing import clean_text


class E5Retriever:
    def __init__(self, model_name="intfloat/e5-base-v2"):
        print("Loading E5 model...")

        # Safe device selection
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.model = SentenceTransformer(model_name, device=device)

        self.context_embeddings = None
        self.contexts = None
        self.responses = None

        print(f"E5 model loaded successfully on {device}.")

    # -----------------------------
    # TRAIN / ENCODE
    # -----------------------------
    def fit(self, contexts, responses):
        if contexts is None or len(contexts) == 0:
            raise ValueError("Empty context data provided.")

        print("Encoding context data (E5)...")

        self.contexts = contexts.reset_index(drop=True)
        self.responses = responses.reset_index(drop=True)

        clean_contexts = []
        valid_indices = []

        for i, c in enumerate(self.contexts):
            if isinstance(c, str) and c.strip() != "":
                clean_contexts.append("passage: " + clean_text(c))
                valid_indices.append(i)

        if len(clean_contexts) == 0:
            raise ValueError("No valid context data after cleaning.")

        # Keep only valid rows
        self.contexts = self.contexts.iloc[valid_indices].reset_index(drop=True)
        self.responses = self.responses.iloc[valid_indices].reset_index(drop=True)

        # Encode passages
        self.context_embeddings = self.model.encode(
            clean_contexts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        print(f"Encoding completed. Total vectors: {len(self.context_embeddings)}")

    # -----------------------------
    # RETRIEVE
    # -----------------------------
    def retrieve(self, query, top_k=5):
        if self.context_embeddings is None:
            raise ValueError("Model not trained. Call fit() first.")

        if not isinstance(query, str) or query.strip() == "":
            raise ValueError("Invalid query input.")

        query = clean_text(query)

        # E5 format
        query_input = "query: " + query

        query_embedding = self.model.encode(
            [query_input],
            batch_size=1,  # ✅ consistency + clarity
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarities = cosine_similarity(query_embedding, self.context_embeddings).flatten()

        top_k = min(top_k, len(similarities))
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = [
            {
                "context": self.contexts.iloc[idx],
                "response": self.responses.iloc[idx],
                "score": float(similarities[idx])
            }
            for idx in top_indices
        ]

        return results