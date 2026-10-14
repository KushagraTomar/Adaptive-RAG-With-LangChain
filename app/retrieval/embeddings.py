from langchain_mistralai import MistralAIEmbeddings

EMBEDDING_MODEL = "mistral-embed"
PINECONE_INDEX_NAME = "pdf-structured-rag-mistral"
PINECONE_INDEX_DIMENSION = 1024
PINECONE_NAMESPACE = "default"


def get_embeddings() -> MistralAIEmbeddings:
    return MistralAIEmbeddings(model=EMBEDDING_MODEL)
