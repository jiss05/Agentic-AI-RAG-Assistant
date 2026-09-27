from sentence_transformers import SentenceTransformer
import numpy as np
import os

from chunking import chunks


embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    embedding_model_name
)

print("Embedding model:", embedding_model_name)

print(
    "Embedding dimensions:",
    embedding_model.get_sentence_embedding_dimension()
)


# Cache file
CACHE_DIR = "data/cache"
CACHE_FILE = os.path.join(
    CACHE_DIR,
    "embeddings.npy"
)


# Create cache directory if it doesn't exist
os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


if os.path.exists(CACHE_FILE):

    print("Loading cached embeddings...")

    embeddings = np.load(
        CACHE_FILE
    )

else:

    print("Generating embeddings...")

    chunk_texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    np.save(
        CACHE_FILE,
        embeddings
    )

    print("Embeddings saved to cache.")


print(
    "Embedding shape:",
    embeddings.shape
)