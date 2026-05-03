# Adaptive RAG Architecture

## Overview

This document describes the architecture of the Adaptive Retrieval-Augmented Generation (RAG) system with LangGraph and LangChain.

## System Flow

```
Question
   ↓
Retrieve Documents (from PDFs)
   ↓
Grade Documents (Relevance Check)
   ↓
   ├→ Relevant? YES → Generate Answer → Answer
   │
   └→ Relevant?  NO → Transform Query → Web Search → Generate Answer → Answer
```

## Components

### 1. Retrieval Module (`app/retrieval/`)
- **ingestion.py**: PDF loading, markdown conversion, and chunking
- **vectorstore.py**: Pinecone integration
- **retrievers.py**: BM25, dense, and hybrid retrieval

### 2. RAG Module (`app/rag/`)
- **rag.py**: Main RAG pipeline orchestration
- **graders.py**: Document relevance grading
- **rewriter.py**: Query transformation

### 3. API Module (`app/api/`)
- **main.py**: FastAPI application
- **routes.py**: API endpoints

## Configuration

All settings are managed in `config/settings.py`:
- API keys (Mistral, Tavily, Cohere, Pinecone)
- Model configurations
- Retrieval parameters

## Data Organization

- **data/pdfs/**: Source documents
- **data/chunks/**: Cached chunks
- **data/logs/**: Application logs

## Models

Pydantic models for request/response validation are in `app/models/schemas.py`.
