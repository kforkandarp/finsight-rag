---
title: FinSight RAG
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# FinSight RAG — Financial Document Intelligence System

A production-grade RAG system for querying Infosys Annual Reports (FY2024, FY2025, FY2026).

## Architecture
- **Retrieval:** Hybrid BM25 + FAISS with cross-encoder reranking
- **Generation:** LLaMA 3.3 70B via Groq API
- **Evaluation:** RAGAS (Context Precision: 0.86, Faithfulness: 0.70)
- **API:** FastAPI with citation enforcement
- **Deployment:** Docker

## API Endpoints
- `GET /health` — service status
- `POST /query` — query the financial documents
- `GET /docs` — interactive API documentation