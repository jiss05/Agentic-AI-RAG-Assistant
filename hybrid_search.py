
from vector_search import vector_search
from keyword_search import keyword_search


def hybrid_search(
    query,
    top_k=5,
    vector_weight=0.7,
    keyword_weight=0.3,
    min_vector_score=0.20
):
    """
    Perform hybrid search using vector similarity and BM25 keyword search.

    Args:
        query: User's search query.
        top_k: Maximum number of results to return.
        vector_weight: Weight assigned to vector ranking.
        keyword_weight: Weight assigned to keyword ranking.
        min_vector_score: Minimum raw vector similarity required
                          for a result to be retained.

    Returns:
        A list of filtered and ranked search results.
    """

    # 1. Retrieve candidates from both search methods
    vector_results = vector_search(query, top_k=10)
    keyword_results = keyword_search(query, top_k=10)

    combined_results = {}

    # 2. Add vector search results
    for rank, result in enumerate(vector_results, start=1):

        chunk_id = result["chunk_index"]

        combined_results[chunk_id] = {
            **result,

            # Preserve the original vector similarity
            "raw_vector_score": result["score"],

            # Convert rank into a reciprocal-rank score
            "vector_score": 1 / rank,

            # Initialize keyword scores
            "raw_keyword_score": 0.0,
            "keyword_score": 0.0
        }

    # 3. Add keyword search results and merge matching chunks
    for rank, result in enumerate(keyword_results, start=1):

        chunk_id = result["chunk_index"]
        raw_bm25_score = result["score"]

        if chunk_id in combined_results:

            # This chunk was already found by vector search
            combined_results[chunk_id]["raw_keyword_score"] = (
                raw_bm25_score
            )

            combined_results[chunk_id]["keyword_score"] = 1 / rank

        else:

            # This chunk was found only by keyword search
            combined_results[chunk_id] = {
                **result,

                "raw_vector_score": 0.0,
                "vector_score": 0.0,

                "raw_keyword_score": raw_bm25_score,
                "keyword_score": 1 / rank
            }

    # 4. Calculate the hybrid score using rank fusion
    for result in combined_results.values():

        result["hybrid_score"] = (
            vector_weight * result["vector_score"]
            + keyword_weight * result["keyword_score"]
        )

    # 5. Sort results by hybrid score, highest first
    ranked_results = sorted(
        combined_results.values(),
        key=lambda result: result["hybrid_score"],
        reverse=True
    )

    # 6. Filter results using the raw vector similarity
    filtered_results = [
        result
        for result in ranked_results
        if result["raw_vector_score"] >= min_vector_score
    ]

    # 7. Return the top results after filtering
    return filtered_results[:top_k]


# --------------------------------------------------
# Test the hybrid search
# --------------------------------------------------

if __name__ == "__main__":

    query = input("Enter your question: ")

    results = hybrid_search(query, top_k=5)

    if not results:

        print("\nNO_RELEVANT_INFORMATION")
        print(
            "No results passed the minimum vector similarity "
            "threshold."
        )

    else:

        for rank, result in enumerate(results, start=1):

            print("\n" + "=" * 60)
            print(f"Result: {rank}")

            print(f"Hybrid Score: {result['hybrid_score']:.4f}")
            print(
                f"Raw Vector Score: "
                f"{result['raw_vector_score']:.4f}"
            )
            print(
                f"Raw BM25 Score: "
                f"{result['raw_keyword_score']:.4f}"
            )

            print(f"Document: {result['id']}")
            print(f"Page: {result['page']}")

            print("\nText:")
            print(result["text"])

        print("\n" + "=" * 60)