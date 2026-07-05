import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# Import our own modules — each one does one job
from ingestion import load_and_chunk       # loads + chunks PDFs
from retriever import build_retriever      # builds hybrid BM25+FAISS retriever
from reranker import rerank                # cross-encoder reranking

load_dotenv()  # loads GROQ_API_KEY and LANGCHAIN_API_KEY from .env


# ── PROMPT TEMPLATE ────────────────────────────────────────────────────────── 
# System Prompts are the permanent instructions / personality / rules for the LLM. System prompt constrains behavior.
# This is the system prompt that controls LLM behaviour.
# Key rules we enforce:
# 1. Answer ONLY from provided context — prevents hallucination
# 2. Always cite source document + page — enforces grounded citations
# 3. Say "I don't know" if context doesn't contain the answer — honest fallback
SYSTEM_PROMPT = """You are FinSight, an AI assistant specialized in analyzing Infosys financial reports.

RULES YOU MUST FOLLOW:
1. Answer ONLY using the context provided below. Do not use any outside knowledge.
2. Always cite your source: mention the document name and page number for every fact.
3. If the context does not contain enough information to answer, say: "I don't have enough information in the provided documents to answer this question."
4. Be precise with numbers — if you see revenue figures, quote them exactly as they appear.
5. If multiple years are mentioned in context, clearly distinguish between them.

FORMAT:
- Give a direct answer first
- Then provide supporting evidence with citations
- Citations format: [Source: <filename>, Page <number>]
"""
# ─────────────────────────────────────────────────────────────────────────────


def build_context_string(documents: list) -> str: # input is List of reranked documents. output is a single string.
    # we convert from list to string because the LLM expects a single string as context, not a list of documents.

    """
    Converts a list of Document objects into a single formatted context string.
    This string gets passed to the LLM as grounding context.

    Each chunk includes its source and page number so the LLM can cite them.
    """
    context_parts = []
    for i, doc in enumerate(documents):
        # Extract clean filename from full path
        source = doc.metadata.get("source", "unknown").split("\\")[-1].split("/")[-1]
        page = doc.metadata.get("page", "?")

        # Format each chunk with its citation info visible to the LLM
        chunk_text = f"[Chunk {i+1} | Source: {source} | Page {page}]\n{doc.page_content}" # here we keep metadata in the context so LLM can cite it. 
        #We use i+1 to make chunk numbering 1-based for human readability
        # here the f-string makes the document from a list to a string.


        context_parts.append(chunk_text) # this is a list of strings

    # Join all chunks with a separator so LLM can distinguish between them
    return "\n\n---\n\n".join(context_parts) # this creates one big string with all the chunks separated by "---" so LLM can see them as separate chunks.


def query_pipeline(query: str, retriever, llm: ChatGroq, top_n: int = 5) -> dict:

    # query is user question, retriever is hybrid retriever, llm is LLM object, top_n is number of top documents to return after reranking.


    """
    Full RAG pipeline for a single query.

    Flow:
    query → hybrid retriever (BM25+FAISS) → reranker → LLM → answer + citations

    Returns a dict with:
    - answer: the LLM's response
    - sources: list of (filename, page) tuples used
    - context: the raw chunks passed to LLM (useful for RAGAS evaluation later)
    """

    # Step 1: Retrieve top candidates using hybrid retriever
    # retriever.invoke() runs BM25 + FAISS in parallel and merges results
    candidates = retriever.invoke(query)

    # Step 2: Rerank candidates — cross-encoder picks best top_n
    top_docs = rerank(query, candidates, top_n=top_n)

    # Step 3: Build context string from reranked docs
    context = build_context_string(top_docs) # Convert document objects into LLM-readable text.

    # Step 4: Build messages for the LLM
    # We use a SystemMessage (behaviour rules) + HumanMessage (query + context)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        # Injecting context directly into the human message — standard RAG pattern
        # The LLM sees: system rules + retrieved context + user question
    ]

    # Step 5: Call LLM — LLaMA 3.3 70B via Groq API
    response = llm.invoke(messages)

    # Step 6: Extract source citations from top_docs metadata
    sources = []
    for doc in top_docs:
        source = doc.metadata.get("source", "unknown").split("\\")[-1].split("/")[-1]
        page = doc.metadata.get("page", "?")
        sources.append((source, page))  # list of (filename, page) tuples

    return {
        "answer": response.content,   # LLM's text response
        "sources": sources,            # list of (filename, page) for citations
        "context": top_docs            # raw Document objects — needed for RAGAS
    }


# ── MAIN: Full pipeline test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initialising FinSight RAG pipeline...")
    print("=" * 60)

    # Step 1: Load and chunk all PDFs
    print("\n[1/3] Loading and chunking documents...")
    chunks = load_and_chunk()

    # Step 2: Build hybrid retriever from chunks
    print("\n[2/3] Building hybrid retriever...")
    retriever = build_retriever(chunks)

    # Step 3: Initialise LLM — LLaMA 3.3 70B via Groq (fast, free tier)
    print("\n[3/3] Initialising LLM...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # Groq's hosted LLaMA 3.3 70B
        temperature=0,                     # temperature=0 -> deterministic, factual answers
                                           # higher temp = more creative but less accurate
        api_key=os.getenv("GROQ_API_KEY")
    )

    print("\n" + "=" * 60)
    print("Pipeline ready. Running test queries...")
    print("=" * 60)

    # Test queries — designed to test different retrieval scenarios
    test_queries = [
        "What was Infosys total revenue in FY2024?",
        "What is Infosys operating margin for FY2025?",
        "How many employees does Infosys have?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print("=" * 60)

        result = query_pipeline(query, retriever, llm)

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES USED:")
        for source, page in result['sources']:
            print(f"  - {source} | Page {page}")