import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from typing import Literal
from langchain import hub

# Set up embeddings
embd = MistralAIEmbeddings()

# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load Documents from urls
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split Documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embd,
)

# Make vectorstore as Retriever
retriever = vectorstore.as_retriever()

# Data model for routing
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

# Data model for document grading
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Data model for hallucination grading
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# Data model for answer grading
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# Initialize LLMs
mistral_model = "mistral-large-latest"
llm = ChatMistralAI(model=mistral_model, temperature=0)

# Router setup
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

# Retrieval grader setup
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the user question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# Generate chain setup
prompt = hub.pull("rlm/rag-prompt")

# LLM for generation
llm = ChatMistralAI(model=mistral_model, temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Hallucination grader setup
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader

# Answer grader setup
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader

# Question re-writer
llm = ChatMistralAI(model=mistral_model, temperature=0)

system = """You a question re-writer that converts an input question to a better version that is optimized 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()

# Web search tool
web_search_tool = TavilySearchResults(k=3)

# Main RAG function
def run_rag_query(question: str) -> Dict[str, Any]:
    # Route question
    source = question_router.invoke({"question": question})
    
    if source.datasource == "web_search":
        # Web search
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        return {"answer": web_results, "source": "web_search"}
    else:
        # Vectorstore retrieval
        documents = retriever.invoke(question)
        
        # Grade documents
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                filtered_docs.append(d)
        
        if not filtered_docs:
            # No relevant documents, rephrase question
            better_question = question_rewriter.invoke({"question": question})
            documents = retriever.invoke(better_question)
            
            # Grade documents again
            filtered_docs = []
            for d in documents:
                score = retrieval_grader.invoke(
                    {"question": better_question, "document": d.page_content}
                )
                grade = score.binary_score
                if grade == "yes":
                    filtered_docs.append(d)
        
        if filtered_docs:
            # Generate answer
            generation = rag_chain.invoke({"context": format_docs(filtered_docs), "question": question})
            
            # Check hallucinations
            hallucination_score = hallucination_grader.invoke(
                {"documents": format_docs(filtered_docs), "generation": generation}
            )
            
            if hallucination_score.binary_score == "yes":
                # Check answer quality
                answer_score = answer_grader.invoke({"question": question, "generation": generation})
                if answer_score.binary_score == "yes":
                    return {"answer": generation, "source": "vectorstore"}
                else:
                    # Answer not useful, rephrase and try again
                    better_question = question_rewriter.invoke({"question": question})
                    documents = retriever.invoke(better_question)
                    filtered_docs = []
                    for d in documents:
                        score = retrieval_grader.invoke(
                            {"question": better_question, "document": d.page_content}
                        )
                        grade = score.binary_score
                        if grade == "yes":
                            filtered_docs.append(d)
                    
                    if filtered_docs:
                        generation = rag_chain.invoke({"context": format_docs(filtered_docs), "question": better_question})
                        return {"answer": generation, "source": "vectorstore"}
                    else:
                        return {"answer": "No relevant information found.", "source": "vectorstore"}
            else:
                # Hallucination detected, try web search
                docs = web_search_tool.invoke({"query": question})
                web_results = "\n".join([d["content"] for d in docs])
                return {"answer": web_results, "source": "web_search"}
        else:
            # No documents found, try web search
            docs = web_search_tool.invoke({"query": question})
            web_results = "\n".join([d["content"] for d in docs])
            return {"answer": web_results, "source": "web_search"}

# Function to get answer for a question
def get_answer(question: str) -> str:
    result = run_rag_query(question)
    return result["answer"]