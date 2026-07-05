import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Path to your PDFs — works regardless of where you run the script from
PDF_DIR = Path(__file__).parent.parent / "data" / "raw_pdfs"

# Keywords that identify TRUE junk pages — pages with NO financial data.
# Must be specific enough to NOT match the header that appears on every page.
# Strategy: match phrases that only appear in cover/intro pages as main content,
# not as a running header on data pages.
JUNK_PATTERNS = [
    "dear members,\nthe board of directors hereby submits",  # exact cover letter opening
    "dear shareholders,\nthe board of directors",
    "table of contents",
    "scan here to access the digital version",              # QR code pages
    "the cover and theme pages images have been created",   # cover page note
]

# Minimum content threshold — pages with very little text are likely
# image-only pages, dividers, or blank pages with just a header
MIN_PAGE_LENGTH = 200


def is_junk_page(text: str) -> bool:
    """
    Returns True only for genuine junk pages:
    1. Pages shorter than MIN_PAGE_LENGTH (image pages, dividers, blank pages)
    2. Pages whose content matches specific cover/intro patterns

    
    """
    # Filter 1: too short to contain real data
    if len(text.strip()) < MIN_PAGE_LENGTH:
        return True

    # Filter 2: specific junk patterns — check full page now, not just first 300 chars
    # because some patterns appear mid-page
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in JUNK_PATTERNS)
# ─────────────────────────────────────────────────────────────────────────────


def load_pdfs() -> list:  # This function is expected to return a list.
    """
    Loads all PDFs from the raw_pdfs directory.
    Returns a list of Document objects, one per page.
    Each Document has .page_content (text) and .metadata (source, page number).
    Now filters out junk pages before returning.
    """
    all_documents = []
    filtered_count = 0  # ── NEW: track how many junk pages were removed

    pdf_files = list(PDF_DIR.glob("*.pdf"))  # glob means: Search files matching pattern and the pattern is "*.pdf", where * = anything.
    # since glob returns a generator, we convert it to a list so we can check its length and iterate over it.

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}. Add your PDFs there.")  # raise means: Stop program and throw error

    print(f"Found {len(pdf_files)} PDF(s):")

    for pdf in pdf_files:  # pdf is Path object. Path objects -> A smarter string specifically designed for file/folder paths. .name is the filename with extension,
                           # .stem is filename without extension, .suffix is extension only.
        print(f"  -> {pdf.name}")

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))  # Because pdf_path is Path object, we convert it to string to pass to PyPDFLoader.
        documents = loader.load()            # One page = one Document object. This reads pdf.

        # ── NEW ──────────────────────────────────────────────────────────────
        # Filter junk pages BEFORE adding to all_documents.
        # Without this, cover letter pages pollute the FAISS + BM25 indexes
        # and the reranker scores them high because they mention "revenue", "FY2024" etc.
        clean_docs = []
        for doc in documents:
            if is_junk_page(doc.page_content):
                filtered_count += 1  # count it but don't add it
            else:
                clean_docs.append(doc)  # only clean pages go in

        all_documents.extend(clean_docs)  # extend: adds elements individually to the list
        print(f"Loaded {len(clean_docs)} clean pages from {pdf_path.name} "
              f"(filtered {len(documents) - len(clean_docs)} junk pages)")
        # ─────────────────────────────────────────────────────────────────────

    print(f"\nTotal clean pages loaded: {len(all_documents)}")
    print(f"Total junk pages filtered: {filtered_count}")  # ── NEW
    return all_documents


def chunk_documents(documents: list) -> list:  # This function expects a list of Document objects and then it returns a list.
    """
    Splits documents into smaller overlapping chunks.

    chunk_size=1000   — each chunk is ~1000 characters
    chunk_overlap=200 — chunks overlap by 200 chars so context isn't lost at boundaries

    Why these numbers for financial docs:
    - Financial reports have dense tables and paragraphs
    - 1000 chars captures enough context per chunk
    - 200 char overlap ensures a sentence split across chunk boundaries isn't lost
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],  # tries these in order to find split points
        length_function=len  # uses Python's built-in len() to measure chunk length in characters
    )

    chunks = splitter.split_documents(documents)  # each chunk is a Document object with .page_content and .metadata

    print(f"Total chunks created: {len(chunks)}")
    print(f"Sample chunk (first 200 chars):\n{chunks[0].page_content[:200]}")
    print(f"Sample metadata: {chunks[0].metadata}")

    return chunks


def load_and_chunk() -> list:
    """
    Master function — call this from other modules.
    Returns ready-to-use chunks.
    """
    # Master Function -> This function combines smaller functions so other files don't need to call them separately.
    documents = load_pdfs()       # After this line, documents is a list of clean Document objects, one per page.
    chunks = chunk_documents(documents)  # Now you pass those loaded documents into chunker.
    return chunks


# Quick test — run this file directly to verify it works
# This runs only if you execute this file directly, not if you import it as a module.
if __name__ == "__main__":
    chunks = load_and_chunk()
    print(f"\nIngestion complete. {len(chunks)} chunks ready for retrieval.")