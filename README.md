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

2. Set up environment variables:
```bash
export MISTRAL_API_KEY="your-mistral-api-key"
export COHERE_API_KEY="your-cohere-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
export PINECONE_INDEX_NAME="adaptive-rag"
export PINECONE_CLOUD="aws"
export PINECONE_REGION="us-east-1"
```

Optional:
```bash
export PINECONE_NAMESPACE="default"
```

3. Run the ingestion pipeline to chunk PDFs and index them in Pinecone:
```bash
python -m app.ingestion
```

4. Run the FastAPI server:
```bash
uvicorn app.main:app --reload
```

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

### Backend
- **Python** - Core programming language
- **FastAPI** - Modern, fast web framework for building APIs
- **LangChain** - Framework for developing applications with LLMs
- **LangGraph** - Orchestrating multi-step LLM workflows and agentic systems
- **Mistral AI** - Large Language Model for answer generation

### Retrieval & Ranking
- **Pinecone** - Vector database for semantic search and document retrieval
- **Cohere Reranker** - Re-ranking retrieved documents for improved relevance
- **Hybrid Retriever** - Combines dense and sparse retrieval strategies
- **Tavily Search** - Web search integration for real-time information

### Frontend
- **React.js** - UI library for building interactive user interface
- **Vite** - Fast frontend build tool and development server
- **Axios** - HTTP client for API communication

### Data & Storage
- **PostgreSQL** - Relational database for metadata and user data
- **Pinecone Vector Store** - Distributed vector database for embeddings

### Evaluation & Quality
- **Evaluation Framework** - Custom metrics for answer quality assessment
- **Grading System** - Document relevance and hallucination detection