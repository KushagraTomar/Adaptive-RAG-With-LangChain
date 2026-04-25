import hashlib
import os
import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec

# Set up embeddings
embd = MistralAIEmbeddings()

# Load PDF documents from a local directory.
pdf_dir = os.getenv("PDF_DIR", "pdfs")
loader = PyPDFDirectoryLoader(pdf_dir, mode="single")
docs_list = loader.load()
print(f"Loaded {len(docs_list)} documents from {pdf_dir}")

# splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=400,
#     chunk_overlap=50,
#     separators=["\n## ","\n\n", "\n", ". "," ", ""]  # tries each in order
# )

# split_docs = splitter.split_documents(docs_list)
# print(f"Split into {len(split_docs)} chunks.")

# for i, doc in enumerate(split_docs[:20]):
#     print(f"\nChunk {i+1}:\n")
#     print(doc.page_content)

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import os

doc_splits = []
for filename in os.listdir("pdfs/"):
    if filename.endswith(".pdf"):
        # Convert PDF to Markdown preserving structure
        md_text = pymupdf4llm.to_markdown(f"pdfs/{filename}")
        
        # Split on Markdown headings — # ## ### are natural boundaries
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers = False,
        )
        chunks = splitter.split_text(
            md_text
        )
        doc_splits.extend(chunks)

print(f"Split into {len(doc_splits)} chunks.")

# Set up Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "adaptive-rag")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "default")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is required.")

pc = Pinecone(api_key=pinecone_api_key)

if not pc.has_index(pinecone_index_name):
    embedding_dimension = 1024  # MistralAI embedding dimension
    pc.create_index(
        name=pinecone_index_name,
        vector_type="dense",
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=pinecone_cloud, 
            region=pinecone_region
        ),
    )
else:
    print(f"Pinecone index '{pinecone_index_name}' already exists.")

index = pc.Index(pinecone_index_name)
vectorstore = PineconeVectorStore(
    index=index, 
    embedding=embd, 
    namespace=pinecone_namespace
)

# Upsert documents with stable IDs so repeated app starts do not create duplicates.
doc_ids = [
    hashlib.sha256(
        f"{doc.metadata.get('source', '')}:{doc.metadata.get('page', '')}:{doc.page_content}".encode("utf-8")
    ).hexdigest()
    for doc in doc_splits
]
vectorstore.add_documents(documents=doc_splits, ids=doc_ids)

# Make vectorstore as Retriever
# retriever = vectorstore.as_retriever()

# Build dense + BM25 retrievers for hybrid search.
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
bm25_retriever = BM25Retriever.from_documents(doc_splits)
bm25_retriever.k = 2

hybrid_retrieve = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6],
)

mistral_model = "mistral-large-latest"
llm = ChatMistralAI(model=mistral_model, temperature=0.1)

def format_docs(docs: List[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for question answering. "
            "Use only the retrieved context to answer the question. "
            "If the answer is not in the context, say you could not find it in the documents.",
        ),
        (
            "human",
            "Question: {question}\n\nRetrieved context:\n{context}",
        ),
    ]
)

rag_chain = generation_prompt | llm | StrOutputParser()

def answer_question(question: str) -> str:
    # The retriever embeds the user question internally before similarity search in Pinecone.
    # documents = retriever.invoke(question)
    documents = hybrid_retrieve.invoke(question)

    print(f"Retrieved {len(documents)} documents.")

    context = format_docs(documents)
    return rag_chain.invoke({"question": question, "context": context})


if __name__ == "__main__":
    user_question = "explain the encoder and decoder stacks"
    answer = answer_question(user_question)
    print("\nAnswer:\n")
    print(answer)
        
# #         # Grade documents
# #         filtered_docs = []
# #         for d in documents:
# #             score = retrieval_grader.invoke(
# #                 {"question": question, "document": d.page_content}
# #             )
# #             grade = score.binary_score
# #             if grade == "yes":
# #                 filtered_docs.append(d)
        
# #         if not filtered_docs:
# #             # No relevant documents, rephrase question
# #             better_question = question_rewriter.invoke({"question": question})
# #             documents = retriever.invoke(better_question)
            
# #             # Grade documents again
# #             filtered_docs = []
# #             for d in documents:
# #                 score = retrieval_grader.invoke(
# #                     {"question": better_question, "document": d.page_content}
# #                 )
# #                 grade = score.binary_score
# #                 if grade == "yes":
# #                     filtered_docs.append(d)
        
# #         if filtered_docs:
#             # Generate answer
#             # generation = rag_chain.invoke({"context": format_docs(filtered_docs), "question": question})
            
# #             # Check hallucinations
# #             hallucination_score = hallucination_grader.invoke(
# #                 {"documents": format_docs(filtered_docs), "generation": generation}
# #             )
            
# #             if hallucination_score.binary_score == "yes":
# #                 # Check answer quality
# #                 answer_score = answer_grader.invoke({"question": question, "generation": generation})
# #                 if answer_score.binary_score == "yes":
# #                     return {"answer": generation, "source": "vectorstore"}
# #                 else:
# #                     # Answer not useful, rephrase and try again
# #                     better_question = question_rewriter.invoke({"question": question})
# #                     documents = retriever.invoke(better_question)
# #                     filtered_docs = []
# #                     for d in documents:
# #                         score = retrieval_grader.invoke(
# #                             {"question": better_question, "document": d.page_content}
# #                         )
# #                         grade = score.binary_score
# #                         if grade == "yes":
# #                             filtered_docs.append(d)
                    
# #                     if filtered_docs:
# #                         generation = rag_chain.invoke({"context": format_docs(filtered_docs), "question": better_question})
# #                         return {"answer": generation, "source": "vectorstore"}
# #                     else:
# #                         return {"answer": "No relevant information found.", "source": "vectorstore"}
# #             else:
# #                 # Hallucination detected, try web search
# #                 docs = web_search_tool.invoke({"query": question})
# #                 web_results = "\n".join([d["content"] for d in docs])
# #                 return {"answer": web_results, "source": "web_search"}
# #         else:
# #             # No documents found, try web search
# #             docs = web_search_tool.invoke({"query": question})
# #             web_results = "\n".join([d["content"] for d in docs])
# #             return {"answer": web_results, "source": "web_search"}

# # # Function to get answer for a question
# # def get_answer(question: str) -> str:
# #     result = run_rag_query(question)
# #     return result["answer"]
