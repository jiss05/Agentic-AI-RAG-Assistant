from smolagents import tool

from retriever import retrieve_context


@tool
def knowledge_base_search(query: str) -> str:
    """
    Search the project knowledge base for relevant information.

    Args:
        query: The user's question or search query.
    """

    context = retrieve_context(query)

    return context

if __name__ == "__main__":
    result = knowledge_base_search("What is RAG?")
    print("\nTOOL RESULT:\n")
    print(result)