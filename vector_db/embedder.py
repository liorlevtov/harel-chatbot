"""BGE-M3 embedding wrapper for dense + sparse vectors."""

from typing import List, Tuple, Dict, Any
import numpy as np

from .config import EMBEDDING_MODEL

# Global model instance (lazy loaded)
_model = None


def get_model():
    """Get or initialize the BGE-M3 model (singleton)."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        from FlagEmbedding import BGEM3FlagModel
        _model = BGEM3FlagModel(
            EMBEDDING_MODEL,
            use_fp16=True,
            device="cpu"  # Change to "cuda" if GPU available
        )
        print("Model loaded successfully.")
    return _model


def embed_texts(texts: List[str]) -> Tuple[np.ndarray, List[Dict[int, float]]]:
    """
    Embed texts using BGE-M3.

    Args:
        texts: List of text strings to embed

    Returns:
        Tuple of:
        - dense_vectors: numpy array of shape (n, 1024)
        - sparse_vectors: list of dicts mapping token_id -> weight
    """
    if not texts:
        return np.array([]), []

    model = get_model()

    # BGE-M3 returns both dense and sparse embeddings
    output = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    # Ensure dense vectors are numpy arrays
    dense_vectors = output["dense_vecs"]
    if not isinstance(dense_vectors, np.ndarray):
        dense_vectors = np.array(dense_vectors)

    # Convert sparse to list of dicts format for Milvus
    sparse_vectors = []
    lexical_weights = output.get("lexical_weights", [])

    for sparse in lexical_weights:
        # sparse is already a dict of {token_id: weight}
        if isinstance(sparse, dict):
            sparse_vectors.append(sparse)
        else:
            sparse_vectors.append({})

    return dense_vectors, sparse_vectors


def embed_query(query: str) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Embed a single query text.

    Args:
        query: Query string

    Returns:
        Tuple of (dense_vector, sparse_vector)
    """
    dense_vectors, sparse_vectors = embed_texts([query])

    dense_vector = dense_vectors[0] if len(dense_vectors) > 0 else np.zeros(1024)
    sparse_vector = sparse_vectors[0] if sparse_vectors else {}

    return dense_vector, sparse_vector
