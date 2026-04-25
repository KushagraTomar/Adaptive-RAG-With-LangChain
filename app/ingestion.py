import hashlib
import os
from dataclasses import dataclass
from typing import List

import pymupdf4llm
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pinecone import Pinecone, ServerlessSpec


@dataclass
class RetrievalResources:
    doc_splits: List[Document]
    vectorstore: PineconeVectorStore
    dense_retriever: object
    bm25_retriever: BM25Retriever
    hybrid_retriever: EnsembleRetriever
    compression_retriever: ContextualCompressionRetriever


def get_doc_id(doc: Document) -> str:
    return hashlib.sha256(
        f"{doc.metadata.get('source', '')}:{doc.metadata.get('page', '')}:{doc.page_content}".encode("utf-8")
    ).hexdigest()


def load_and_chunk_documents(pdf_dir: str | None = None) -> List[Document]:
    pdf_dir = pdf_dir or os.getenv("PDF_DIR", "pdfs")
    doc_splits: List[Document] = []

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    for filename in os.listdir(pdf_dir):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, filename)
        md_text = pymupdf4llm.to_markdown(pdf_path)
        chunks = md_splitter.split_text(md_text)

        for chunk in chunks:
            chunk.metadata.setdefault("source", pdf_path)

        doc_splits.extend(chunks)

    print(f"Split into {len(doc_splits)} chunks from {pdf_dir}.")
    return doc_splits


def get_embeddings() -> MistralAIEmbeddings:
    return MistralAIEmbeddings()


def get_pinecone_vectorstore(embedding: MistralAIEmbeddings) -> PineconeVectorStore:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "adaptive-rag")
    pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "default")
    pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required.")

    pc = Pinecone(api_key=pinecone_api_key)

    if not pc.has_index(pinecone_index_name):
        pc.create_index(
            name=pinecone_index_name,
            vector_type="dense",
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=pinecone_cloud,
                region=pinecone_region,
            ),
        )
        print(f"Created Pinecone index '{pinecone_index_name}'.")
    else:
        print(f"Pinecone index '{pinecone_index_name}' already exists.")

    index = pc.Index(pinecone_index_name)
    return PineconeVectorStore(
        index=index,
        embedding=embedding,
        namespace=pinecone_namespace,
    )


def index_documents(
    documents: List[Document],
    vectorstore: PineconeVectorStore,
) -> None:
    doc_ids = [get_doc_id(doc) for doc in documents]
    vectorstore.add_documents(documents=documents, ids=doc_ids)
    print(f"Indexed {len(documents)} chunks into Pinecone.")


def build_retrieval_resources(
    dense_k: int = 3,
    bm25_k: int = 3,
    rerank_top_n: int = 2,
) -> RetrievalResources:
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
        model="rerank-english-v3.0",
        top_n=rerank_top_n,
        cohere_api_key=os.getenv("COHERE_API_KEY"),
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
    embedding = get_embeddings()
    doc_splits = load_and_chunk_documents()
    vectorstore = get_pinecone_vectorstore(embedding)
    index_documents(doc_splits, vectorstore)


if __name__ == "__main__":
    run_ingestion_pipeline()
