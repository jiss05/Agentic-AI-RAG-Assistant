from rank_bm25 import BM25Okapi

from chunking import chunks


# Prepare documents for BM25
tokenized_chunks = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

# Create BM25 index
bm25 = BM25Okapi(tokenized_chunks)

print("BM25 index created")


def keyword_search(query, top_k=10):

    # Tokenize query
    query_tokens = query.lower().split()

    # Get BM25 scores
    scores = bm25.get_scores(query_tokens)

    # Get top results
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for index in ranked_indices:

        chunk = chunks[index]

        results.append({
            "score": float(scores[index]),
            "chunk_index": int(index),
            "id": chunk.metadata["document_id"],
            "title": chunk.metadata["title"],
            "source": chunk.metadata["source_file"],
            "page": chunk.metadata["page"],
            "text": chunk.page_content
        })

    return results


# Test keyword search
if __name__ == "__main__":

    query = input("\nEnter your question: ")

    results = keyword_search(
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