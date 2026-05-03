"""Configuration settings and environment variables"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHUNKS_DIR = DATA_DIR / "chunks"

# API Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone Settings
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "adaptive-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Model Settings
MISTRAL_MODEL = "mistral-large-latest"
COHERE_RERANK_MODEL = "rerank-english-v3.0"

# Retrieval Settings
DENSE_RETRIEVER_K = 3
BM25_RETRIEVER_K = 3
RERANK_TOP_N = 2

# Web Search
TAVILY_MAX_RESULTS = 2

# LLM Settings
LLM_TEMPERATURE = 0.1
