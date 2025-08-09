# Adaptive Retrieval-Augmented Generation (RAG) Pipeline

## 📌 Introduction
This project implements an **Adaptive Retrieval-Augmented Generation (RAG)** system that dynamically adapts its retrieval and response generation based on the quality and relevance of retrieved documents. Unlike static RAG pipelines, this system includes **query routing, query transformation, and relevance grading**, allowing it to intelligently choose between **vectorstore retrieval** and **web search**.

The system indexes a curated set of domain-specific resources into a vectorstore and uses them whenever applicable. If the query is outside the scope of the indexed topics, it routes the request to a web search tool.

## 🛠️ Project Overview
The Adaptive RAG pipeline:
1. **Builds and indexes a vectorstore** from selected online resources.
2. **Routes queries** intelligently to the most relevant data source.
3. **Retrieves and grades documents** to ensure high-quality context.
4. **Rewrites queries** when relevant documents are lacking.
5. **Generates answers** that are both grounded in retrieved content and address the user’s question.

This adaptive flow ensures **accuracy, context relevance, and query efficiency**.

## ⚙️ Features
- **Domain-Specific Indexing** – Creates a Chroma vectorstore from selected web resources.
- **Intelligent Query Routing** – Routes to vectorstore or web search based on query topic.
- **Adaptive Query Refinement** – Reformulates queries when retrieval fails.
- **Document Relevance Grading** – Filters out irrelevant content before generation.
- **LLM Answer Validation** – Checks if the generated answer is both relevant and grounded in documents.
- **Graph-Based Workflow** – Orchestrated with `StateGraph` for flexible execution paths.

## 🧰 Tools & Technologies
- **Programming Language:** Python
- **Framework:** LangChain
- **LLM Model:** Mistral (`mistral-large-latest`)
- **Embeddings:** Mistral AI Embeddings (`langchain_mistralai`)
- **Vectorstore:** Chroma
- **Data Loading:** `WebBaseLoader` from `langchain_community`
- **Text Splitting:** RecursiveCharacterTextSplitter
- **Search Integration:** Tavily Search API
- **Prompt Source:** LangChain Hub
- **Workflow Management:** `langgraph`
- **Type Safety:** `TypedDict` and Pydantic models


## 📖 How It Works (Summary)
1. **Index Building**  
   - Loads predefined URLs.  
   - Splits content into chunks.  
   - Embeds and stores them in a Chroma vectorstore.
   
2. **Query Routing**  
   - Routes the question to **vectorstore** if related to indexed topics (agents, prompt engineering, adversarial attacks).  
   - Otherwise, sends it to **web search**.

3. **Document Retrieval & Grading**  
   - Retrieves relevant documents.  
   - Grades them for relevance to the query and checks generation for hallucination.

4. **Adaptive Query Transformation**  
   - If no relevant docs, rewrites the question for better results.

5. **Answer Generation & Validation**  
   - Generates an answer grounded in retrieved documents.  
   - Validates against hallucinations and question relevance.
