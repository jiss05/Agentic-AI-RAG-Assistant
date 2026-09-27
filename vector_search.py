import faiss
import numpy as np

from embeddings import embedding_model, embeddings
from chunking import chunks


# Create FAISS vector index
dimension = embeddings.shape[1]

vector_index = faiss.IndexFlatIP(dimension)

vector_index.add(embeddings)

print("Number of vectors:", vector_index.ntotal)


def vector_search(query, top_k=10):

    # Convert user query into an embedding
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Search for the most similar vectors
    scores, indices = vector_index.search(
        query_embedding,
        top_k
    )

    results = []

    # Get the original chunks
    for score, index in zip(
        scores[0],
        indices[0]
    ):

        chunk = chunks[int(index)]

        results.append({
            "score": float(score),
            "chunk_index": int(index),
            "id": chunk.metadata["document_id"],
            "title": chunk.metadata["title"],
            "source": chunk.metadata["source_file"],
            "page": chunk.metadata["page"],
            "text": chunk.page_content
        })

    return results


# Test vector search
if __name__ == "__main__":

    query = input("\nEnter your question: ")

    results = vector_search(
        query,
        top_k=5
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "=" * 60)

        print("Result:", i)

        print(
            "Score:",
            result["score"]
        )

        print(
            "Chunk Index:",
            result["chunk_index"]
        )

        print(
            "Document:",
            result["title"]
        )

        print(
            "Source:",
            result["source"]
        )

        print(
            "Page:",
            result["page"]
        )

        print("\nText:")

        print(
            result["text"]
        )