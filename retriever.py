from hybrid_search import hybrid_search


def retrieve_context(query):

    results = hybrid_search(
        query,
        top_k=4
    )

    if not results:
        return "NO_RELEVANT_INFORMATION"

    context = ""

    for i, result in enumerate(
        results,
        start=1
    ):

        context += f"""
SOURCE {i}

Document ID: {result['id']}

Title: {result['title']}

Hybrid Score: {result['hybrid_score']:.3f}

Content:

{result['text']}

-------------------------
"""

    return context


if __name__ == "__main__":

    query = input("Enter your question: ")

    context = retrieve_context(query)

    print("\nRESULT:\n")

    print(context)