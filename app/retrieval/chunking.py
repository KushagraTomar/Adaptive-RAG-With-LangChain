import os
from pathlib import Path
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from utils import convert_section_headers

# 1. Load the PDF natively as structured Markdown
# Resolve the repo root relative to this file so it works regardless of where the script is launched from.
repo_root = Path(__file__).resolve().parents[2]
pdf_files = sorted((repo_root / "data").glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found in the directory.")

# 2. Load each PDF as structured Markdown
documents = []

for pdf_path in pdf_files:
    loader = PyMuPDF4LLMLoader(
        file_path=str(pdf_path),
        mode="single"
    )
    loaded_docs = loader.load()
    documents.extend(loaded_docs)

print(f"Total loaded documents: {len(documents)}")

# 4. Define the Markdown Header Splitter rules
# This extracts headers into structural metadata instead of breaking them mid-sentence
headers_to_split_on = [
    ("#", "Header_1"),
    ("##", "Header_2"),
    ("###", "Header_3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, 
    strip_headers=False  # Keeps the markdown tags (#) inside the chunk text for LLM context
)

final_chunks = []

# 3. Process each PDF's Markdown content
for doc in documents:
    markdown_text = doc.page_content
    markdown_text = convert_section_headers(markdown_text)

    # 3. Perform structure-aware splitting
    header_chunks = markdown_splitter.split_text(markdown_text)

    # 4. Sub-chunk massive sections (Sub-splitting)
    # If a single section is huge, recursively break it down safely by characters

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for chunk in header_chunks:
        # If the structural chunk fits our size limits, keep it whole
        if len(chunk.page_content) <= 1000:
            final_chunks.append(chunk)
        else:
            # Otherwise, split it further but pass down the header metadata inherited above
            sub_splits = text_splitter.split_documents([chunk])
            final_chunks.extend(sub_splits)

# 1. Initialize the Embedding Model
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError(
        "Set the MISTRAL_API_KEY environment variable before running this script."
    )

embeddings = MistralAIEmbeddings(api_key=mistral_api_key)

# 2. Setup Pinecone Client
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError(
        "Set the PINECONE_API_KEY environment variable before running this script."
    )

pc = Pinecone(api_key=pinecone_api_key)
index_name = 'pdf-structured-rag'

# Create the index if it doesn't exist yet
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1',
        ),
    )
    print(f"Created Pinecone index '{index_name}'.")
else:
    print(f"Pinecone index '{index_name}' already exists.")

# 3. Batch Upload Chunks to Pinecone
# This automatically converts documents, extracts text, builds vectors, and copies metadata
print("Embedding chunks and uploading to Pinecone...")
vector_store = PineconeVectorStore.from_documents(
    documents=final_chunks, # The chunks list from your PyMuPDF4LLM step
    embedding=embeddings,
    index_name=index_name
)
print("Ingestion complete!")
# # 4. Optional: Quick Test Retrieval (Similarity Search)
# query = "What is the use of Self Attention?"
# docs = vector_store.similarity_search(query, k=2)

# print("\n--- Top Retrieved Chunk Text ---")
# print(docs[0].page_content)
# print("\n--- Preserved Structural Metadata ---")
# print(docs[0].metadata)