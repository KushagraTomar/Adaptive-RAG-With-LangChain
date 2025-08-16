# Adaptive RAG with LangChain and FastAPI


This project implements an Adaptive Retrieval-Augmented Generation (RAG) system using LangChain and exposes it through a FastAPI backend.

## Features

- Adaptive question routing between vectorstore and web search
- Document relevance grading
- Hallucination detection
- Answer quality assessment
- Question rewriting for better retrieval

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
```

3. Run the FastAPI server:
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
