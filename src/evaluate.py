# evaluate.py
# RAGAS evaluation pipeline for FinSight RAG
# Runs offline against our test dataset and produces benchmark scores
import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from ragas.run_config import RunConfig

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # did the answer come from the retrieved context?
    answer_relevancy,   # is the answer relevant to the question?
    context_precision,  # were the retrieved chunks actually useful?
    context_recall,     # did retrieval find all relevant information?
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Import our pipeline components
from ingestion import load_and_chunk
from retriever import build_retriever
from reranker import rerank
from eval_questions import EVAL_DATASET  # our 15 hand-crafted questions

load_dotenv()


def run_pipeline_on_question(question: str, retriever, llm) -> dict:
    """
    Runs a single question through the full RAG pipeline.
    Returns the answer + retrieved contexts — both needed for RAGAS.

    RAGAS needs 4 things per question:
    - question: the query
    - answer: what the LLM said
    - contexts: the raw chunks passed to the LLM (list of strings)
    - ground_truth: the correct answer (from our eval dataset)
    """
    from pipeline import SYSTEM_PROMPT, build_context_string
    from langchain.schema import HumanMessage, SystemMessage

    # Step 1: Retrieve candidates
    candidates = retriever.invoke(question)

    # Step 2: Rerank to top 5
    top_docs = rerank(question, candidates, top_n=5)

    # Step 3: Build context string for LLM
    context_string = build_context_string(top_docs)

    # Step 4: Get LLM answer
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context_string}\n\nQuestion: {question}")
    ]
    response = llm.invoke(messages)

    # Step 5: Extract raw text from each chunk for RAGAS
    # RAGAS expects contexts as a list of strings, not Document objects
    contexts = [doc.page_content for doc in top_docs] 

    return {
        "question": question,
        "answer": response.content,
        "contexts": contexts,  # list of strings — what the LLM saw
    }


def build_ragas_dataset(retriever, llm) -> Dataset:
    """
    Runs all 15 questions through the pipeline and builds a HuggingFace Dataset
    that RAGAS can evaluate.

    This is the most time-consuming step — 15 LLM calls + 15 retrieval passes.
    Expect 2-3 minutes to complete.
    """
    print(f"Running {len(EVAL_DATASET)} questions through pipeline...")
    print("This will take 2-3 minutes — one LLM call per question.\n")

    questions = []
    answers = []
    contexts = [] # context is a list of strings (the chunks passed to the LLM)
    ground_truths = []

    for i, item in enumerate(EVAL_DATASET):
        print(f"[{i+1}/{len(EVAL_DATASET)}] {item['question'][:60]}...")

        try:
            result = run_pipeline_on_question(item["question"], retriever, llm)

            questions.append(result["question"])
            answers.append(result["answer"])
            contexts.append(result["contexts"])          # list of list of contexts
            ground_truths.append(item["ground_truth"])   # from our eval dataset

        except Exception as e:
            # If one question fails (e.g. Groq rate limit), skip it and continue
            # Don't let one failure kill the entire evaluation
            print(f"  ⚠ Skipped (error: {e})")
            continue

    # HuggingFace Dataset format — what RAGAS expects
    ragas_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    return Dataset.from_dict(ragas_data)


def save_results(scores: dict, dataset: Dataset):
    """
    Saves evaluation results to eval_results/ folder.
    Creates two files:
    - scores.json: the benchmark numbers (go on resume + README)
    - detailed_results.json: per-question breakdown (useful for debugging)
    """
    output_dir = Path(__file__).parent.parent / "eval_results"
    output_dir.mkdir(exist_ok=True)  # create folder if it doesn't exist

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # File 1: Summary scores — these are your resume numbers
    scores_path = output_dir / "scores.json"
    scores_data = {
        "timestamp": timestamp,
        "num_questions": len(EVAL_DATASET),
        "metrics": {
            # round to 4 decimal places for clean display
            "faithfulness": round(float(scores["faithfulness"]), 4),
            "answer_relevancy": round(float(scores["answer_relevancy"]), 4),
            "context_precision": round(float(scores["context_precision"]), 4),
            "context_recall": round(float(scores["context_recall"]), 4),
        }
    }
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores_data, f, indent=2)
    print(f"\nScores saved to: {scores_path}")

    # File 2: Per-question breakdown — useful for understanding failures
    detailed_path = output_dir / f"detailed_{timestamp}.json"
    detailed_data = []
    for i in range(len(dataset["question"])):
        detailed_data.append({
            "question": dataset["question"][i],
            "answer": dataset["answer"][i],
            "ground_truth": dataset["ground_truth"][i],
            "contexts_used": len(dataset["contexts"][i]),
        })
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        # ensure_ascii=False preserves ₹ and other Unicode characters
    print(f"Detailed results saved to: {detailed_path}")


def main():
    print("=" * 60)
    print("FinSight RAG — RAGAS Evaluation Suite")
    print("=" * 60)

    # Step 1: Build pipeline (same as pipeline.py)
    print("\n[1/4] Loading and chunking documents...")
    chunks = load_and_chunk()

    print("\n[2/4] Building hybrid retriever...")
    retriever = build_retriever(chunks)

    print("\n[3/4] Initialising LLM...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,  # deterministic answers for consistent eval
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Step 4: Run all questions through pipeline
    print("\n[4/4] Running evaluation dataset through pipeline...")
    dataset = build_ragas_dataset(retriever, llm)

    # Step 5: Run RAGAS evaluation
    # RAGAS uses an LLM internally to judge faithfulness and relevancy
    # We use the same Groq LLM for this to keep costs at zero
    print("\n" + "=" * 60)
    print("Running RAGAS metrics...")
    print("=" * 60)

    # RAGAS needs its own LLM wrapper — different from LangChain's ChatGroq
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Wrap our existing LLM and embeddings for RAGAS compatibility
    ragas_llm = LangchainLLMWrapper(
    ChatGroq(
        model="llama-3.1-8b-instant",  # 8B model — 8x cheaper on tokens than 70B
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # Run evaluation — this makes additional LLM calls to judge each answer
    results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    raise_exceptions=False,  # don't crash on individual failures
    run_config=RunConfig(
        max_workers=1,        # run one job at a time — no parallel requests
        max_retries=3,        # retry failed jobs up to 3 times
        timeout=120,          # 2 minutes per job before giving up
    )
)

    # Step 6: Display and save results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Faithfulness:      {results['faithfulness']:.4f}  (did LLM answer from context?)")
    print(f"Answer Relevancy:  {results['answer_relevancy']:.4f}  (is answer on-topic?)")
    print(f"Context Precision: {results['context_precision']:.4f}  (were retrieved chunks useful?)")
    print(f"Context Recall:    {results['context_recall']:.4f}  (did retrieval find all relevant info?)")
    print("=" * 60)

    save_results(results, dataset)
    print("\nEvaluation complete.")


if __name__ == "__main__": # run only when executed directly, not when imported
    main()