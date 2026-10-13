import os

from langchain_huggingface import HuggingFaceEndpointEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_INDEX_NAME = "pdf-structured-rag-hf"
PINECONE_INDEX_DIMENSION = 384
PINECONE_NAMESPACE = "default"


def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError(
            "Set the HUGGINGFACEHUB_API_TOKEN environment variable before using embeddings."
        )

    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=token,
    )
