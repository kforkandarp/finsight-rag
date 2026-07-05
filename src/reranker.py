from sentence_transformers import CrossEncoder


# Load once at module level — avoids reloading on every query
# ms-marco-MiniLM-L-6-v2: trained specifically for passage relevance ranking
# Downloads ~80MB once, cached after that
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") # this loads the pretrained reranking model


def rerank(query: str, documents: list, top_n: int = 5) -> list:
    """
    Reranks retrieved documents using a cross-encoder model.

    How it works:
    - Takes (query, chunk) pairs
    - Cross-encoder sees BOTH together — much more accurate than vector similarity
    - Returns top_n most relevant chunks

    Why top_n=5 now?
    - Increased from 4 to give LLM slightly more context
    - Still tight enough to avoid token waste
    """

    if not documents:
        return []

    # Build (query, chunk_text) pairs for the cross-encoder
    pairs = [[query, doc.page_content] for doc in documents] # Because cross encoder expects pairs.

    # Score all pairs — higher score = more relevant
    scores = cross_encoder.predict(pairs)

    # Attach scores to documents
    scored_docs = list(zip(scores, documents)) # Zip combines two lists element-wise.

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True) # sort() -> Sort list in place.
    # key -> Tells Python what to sort by. # lambda is Small anonymous function. x:x[0] -> Take item x, return first element.
    # and we know that x is the tuple (score, doc), so x[0] is the score.

    # Print top 8 scores so we can diagnose what the reranker thinks is relevant
    # This is a debug view — helps us catch cases where irrelevant chunks score high
    print("\nReranker scores (top 8 candidates):")
    for score, doc in scored_docs[:8]:
        source = doc.metadata.get("source", "").split("\\")[-1]
        page = doc.metadata.get("page", "?")
        # Printing score to 4 decimal places for precision
        print(f"  Score: {score:.4f} | {source} | Page {page} | {doc.page_content[:80]}...")

    top_docs = [doc for _, doc in scored_docs[:top_n]]
    # Return only the documents, not the scores. As _ is a common convention in Python to indicate that we don't care about that value.
    # this is tuple unpacking. The _ is a throwaway variable for the score, and doc is the document we want to keep.
    # scored_docs[:top_n] -> Take the first top_n elements of the sorted list. top_n is declared in the function parameter.

    return top_docs


# Quick test
if __name__ == "__main__": # this block runs only when the script is executed directly, not when imported as a module.
    from ingestion import load_and_chunk
    from retriever import build_retriever

    chunks = load_and_chunk()
    retriever = build_retriever(chunks)

    # More specific query — "total revenue" gives reranker a clearer signal
    test_query = "What was Infosys total revenue in FY2024?"

    print(f"Query: {test_query}")
    print("\nFetching candidates from hybrid retriever...")
    candidates = retriever.invoke(test_query)

    print(f"Reranking {len(candidates)} candidates to top 5...")
    top_chunks = rerank(test_query, candidates, top_n=5)

    print(f"\nFinal top 5 after reranking:")
    for i, doc in enumerate(top_chunks):
        source = doc.metadata.get("source", "unknown").split("\\")[-1]
        page = doc.metadata.get("page", "?")
        print(f"\n[{i+1}] Source: {source} | Page: {page}")
        print(f"     Content: {doc.page_content[:200]}...")