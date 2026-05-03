import hashlib
from dataclasses import dataclass
from typing import List

import pymupdf4llm
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pinecone import Pinecone, ServerlessSpec

from config.settings import (
    PDF_DIR, PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE,
    PINECONE_CLOUD, PINECONE_REGION, DENSE_RETRIEVER_K, BM25_RETRIEVER_K,
    RERANK_TOP_N, COHERE_API_KEY, COHERE_RERANK_MODEL
)


@dataclass
class RetrievalResources:
    """Container for retrieval resources"""
    doc_splits: List[Document]
    vectorstore: PineconeVectorStore
    dense_retriever: object
    bm25_retriever: BM25Retriever
    hybrid_retriever: object
    compression_retriever: object


def get_doc_id(doc: Document) -> str:
    """Generate unique ID for document"""
    return hashlib.sha256(
        f"{doc.metadata.get('source', '')}:{doc.metadata.get('page', '')}:{doc.page_content}".encode("utf-8")
    ).hexdigest()


def load_and_chunk_documents(pdf_dir: str | None = None) -> List[Document]:
    """Load PDFs and chunk them"""
    pdf_dir = pdf_dir or str(PDF_DIR)
    doc_splits: List[Document] = []

    headers_to_split_on = [
        ("#", "title"),
        ("##", "title"),
        ("###", "title"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    for filename in __import__('os').listdir(pdf_dir):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = __import__('os').path.join(pdf_dir, filename)
        md_text = pymupdf4llm.to_markdown(pdf_path)
        chunks = md_splitter.split_text(md_text)

        for chunk in chunks:
            chunk.metadata.setdefault("source", pdf_path)
            chunk.metadata.setdefault("type", "local pdf")

        doc_splits.extend(chunks)

    print(f"Split into {len(doc_splits)} chunks from {pdf_dir}.")
    return doc_splits


def get_embeddings() -> MistralAIEmbeddings:
    """Get Mistral embeddings model"""
    return MistralAIEmbeddings()


def get_pinecone_vectorstore(embedding: MistralAIEmbeddings) -> PineconeVectorStore:
    """Initialize Pinecone vectorstore"""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable is required.")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            vector_type="dense",
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
        print(f"Created Pinecone index '{PINECONE_INDEX_NAME}'.")
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

    index = pc.Index(PINECONE_INDEX_NAME)
    return PineconeVectorStore(
        index=index,
        embedding=embedding,
        namespace=PINECONE_NAMESPACE,
    )


def index_documents(
    documents: List[Document],
    vectorstore: PineconeVectorStore,
) -> None:
    """Index documents into vectorstore"""
    doc_ids = [get_doc_id(doc) for doc in documents]
    vectorstore.add_documents(documents=documents, ids=doc_ids)
    print(f"Indexed {len(documents)} chunks into Pinecone.")


def build_retrieval_resources(
    dense_k: int = DENSE_RETRIEVER_K,
    bm25_k: int = BM25_RETRIEVER_K,
    rerank_top_n: int = RERANK_TOP_N,
) -> RetrievalResources:
    """Build retrieval resources with hybrid retriever"""
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    
    embedding = get_embeddings()
    doc_splits = load_and_chunk_documents()
    vectorstore = get_pinecone_vectorstore(embedding)

    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": dense_k})
    bm25_retriever = BM25Retriever.from_documents(doc_splits)
    bm25_retriever.k = bm25_k

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],
    )

    reranker = CohereRerank(
        model=COHERE_RERANK_MODEL,
        top_n=rerank_top_n,
        cohere_api_key=COHERE_API_KEY,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever,
    )

    return RetrievalResources(
        doc_splits=doc_splits,
        vectorstore=vectorstore,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        hybrid_retriever=hybrid_retriever,
        compression_retriever=compression_retriever,
    )


def run_ingestion_pipeline() -> None:
    """Run complete ingestion pipeline"""
    embedding = get_embeddings()
    doc_splits = load_and_chunk_documents()
    vectorstore = get_pinecone_vectorstore(embedding)
    index_documents(doc_splits, vectorstore)


if __name__ == "__main__":
    run_ingestion_pipeline()
