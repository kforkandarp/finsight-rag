from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever      # EnsembleRetriever combines multiple retrievers into one, allowing you to leverage the strengths of each.
from langchain_huggingface import HuggingFaceEmbeddings


def build_retriever(chunks: list) -> EnsembleRetriever: # This function expects a list of Document objects (chunks) and returns an EnsembleRetriever object.
    """
    Builds a hybrid BM25 + FAISS retriever.

    Why hybrid?
    - FAISS = semantic search (finds conceptually similar chunks)
    - BM25  = keyword search (finds exact term matches)
    - Together they cover each other's blind spots

    For financial docs specifically:
    - "FY2023 revenue" -> BM25 nails the exact year reference
    - "profitability trends" -> FAISS finds semantically related chunks
    """

    print("Building embeddings model...")
    # all-MiniLM-L6-v2: small, fast, good quality — standard choice for RAG
    # Downloads once (~80MB), cached locally after first run
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}, # tells model where to run
        encode_kwargs={"normalize_embeddings": True}  # normalizing improves cosine similarity scores. This means to Scale vectors to unit length. 
        # Without normalization: Large magnitude vectors may dominate.
        # With normalization: Only direction / semantic meaning matters.
    )

    print("Building FAISS index...")
    # FAISS stores dense vector representations of all chunks
    # as_retriever(k=10) -> returns top 10 semantic matches
    faiss_vectorstore = FAISS.from_documents(chunks, embeddings) 
    # it takes 2 parameters: 1) list of Document objects (chunks), 2) embeddings model. It creates a FAISS index of the chunks.
    faiss_retriever = faiss_vectorstore.as_retriever(
        search_kwargs={"k": 10}
    ) 
    # .as_retriever() converts the FAISS index into a retriever object that can be queried. search_kwargs={"k": 10} means it will return the top 10 results for any query.
    # we convert the FAISS vector store to retriever object, Because LangChain components (chains / RAG pipelines) expect a standard interface. 
    # and ensemble retriever can combine multiple retrievers, but they all need to follow the same interface.

    print("Building BM25 index...")
    # BM25 works directly on raw text — no embeddings needed
    # k=10 -> returns top 10 keyword matches
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    print("Building EnsembleRetriever...")
    # weights=[0.5, 0.5] -> equal contribution from both retrievers
    # Reciprocal Rank Fusion merges and deduplicates results automatically
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    # this creates Hybrid Retriever that combines BM25 and FAISS. The weights parameter allows you to control the influence of each retriever in the final results. 
    # In this case, both retrievers contribute equally (0.5 each).

    print("Hybrid retriever ready.")
    return ensemble_retriever


# Quick test
if __name__ == "__main__": # this runs only if you execute this file directly, not if you import it as a module.
    from ingestion import load_and_chunk

    chunks = load_and_chunk()
    retriever = build_retriever(chunks)

    # Test with a financial query
    test_query = "What was Infosys revenue in FY2024?"
    print(f"\nTest query: {test_query}")
    results = retriever.invoke(test_query)

    print(f"\nTop {len(results)} chunks retrieved:")
    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "unknown").split("\\")[-1] # .get("source", "unknown") retrieves the value of the "source" key from the metadata dictionary. If the key doesn't exist, it returns "unknown". .split("\\")[-1] splits the source path by backslashes and takes the last part, which is typically the filename.
        page = doc.metadata.get("page", "?")
        print(f"\n[{i+1}] Source: {source} | Page: {page}")
        print(f"     Preview: {doc.page_content[:150]}...")