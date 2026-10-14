# Adaptive RAG with LangChain and FastAPI


This project implements an Adaptive Retrieval-Augmented Generation (RAG) system using LangChain and exposes it through a FastAPI backend.

## Features

- Adaptive question routing between vectorstore and web search
- Document relevance grading
- Hallucination detection
- Answer quality assessment
- Question rewriting for better retrieval

## Run
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8001 --reload
```

```bash
cd frontend
npm run dev
```

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

Use Python 3.10 or newer.

2. Set up environment variables:
```bash
export MISTRAL_API_KEY="your-mistral-api-key"
export COHERE_API_KEY="your-cohere-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
export PINECONE_CLOUD="aws"
export PINECONE_REGION="us-east-1"
```

Optional:
```bash
export PINECONE_NAMESPACE="default"
```

3. Run the ingestion pipeline to chunk PDFs and index them in Pinecone. It uses Mistral's `mistral-embed` embedding model and creates the `pdf-structured-rag-mistral` index. Re-run ingestion after changing embedding models:
```bash
python -m app.retrieval.chunking
```

4. Start the RAG pipeline. It uses Mistral's `mistral-large-latest` chat model:
```bash
python -m app.rag.rag
```

5. Run the FastAPI server:
```bash
uvicorn app.main:app --reload
```

Mistral inference and Pinecone are hosted services, so `MISTRAL_API_KEY`, `PINECONE_API_KEY`, and an internet connection are required.

## API Endpoints

- `GET /` - Welcome message
- `POST /ask` - Ask a question and get an answer

### Example Request
```json
{
  "question": "What are the types of agent memory?"
}
```

### Example Response
```json
{
  "answer": "The types of agent memory include...",
  "question": "What are the types of agent memory?"
}
```

## Technology Used

| Backend | Retrieval & Ranking | Frontend | Data & Storage |
|---------|-------------------|----------|----------------|
| Python  | Pinecone | React.js | PostgreSQL |
| FastAPI | Cohere | Vite | Pinecone |
| LangChain | Hybrid Retriever | Axios | Docker |
| LangGraph | Tavily Search | | Redis |
| Mistral AI | | | | 