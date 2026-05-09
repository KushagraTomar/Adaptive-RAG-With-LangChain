from typing import Any, List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

from app.models.schemas import GradeDocuments
from config.settings import MISTRAL_MODEL, LLM_TEMPERATURE, TAVILY_MAX_RESULTS
from app.retrieval.ingestion import build_retrieval_resources
from app.rag.graders import DocumentGrader
from app.rag.rewriter import QueryTransformer
from app.utils.logging import setup_logging


logger = setup_logging(__name__)

_llm = ChatMistralAI(model=MISTRAL_MODEL, temperature=LLM_TEMPERATURE)
_tavily_search_tool = TavilySearch(max_results=TAVILY_MAX_RESULTS)
_document_grader = DocumentGrader()
_query_transformer = QueryTransformer()
_retrieval_resources = build_retrieval_resources()
_compression_retriever = _retrieval_resources.compression_retriever


def get_llm():
    """Get LLM"""
    return _llm


def get_compression_retriever():
    """Get compression retriever"""
    return _compression_retriever


def get_tavily_search_tool():
    """Get Tavily search"""
    return _tavily_search_tool


def get_document_grader():
    """Get document grader"""
    return _document_grader


def get_query_transformer():
    """Get query transformer"""
    return _query_transformer


class RAGState(TypedDict, total=False):
    question: str              # User's question
    documents: List[Document]  # Retrieved documents
    context: str               # Formatted context from retrieved documents
    answer: str                # Generated answer to the question
    is_relevant: bool          # Whether documents are relevant to question
    transformed_query: str     # Query after transformation


grade_llm = _llm.with_structured_output(GradeDocuments)


def get_grade_llm():
    """Get grade LLM"""
    return grade_llm


def format_docs(docs: List[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def retrieve_documents(state: RAGState) -> RAGState:
    question = state["question"]
    compression_retriever = get_compression_retriever()
    documents = compression_retriever.invoke(question)
    logger.info(f"Retrieved {len(documents)} documents.")
    return {
        "documents": documents,
        "context": format_docs(documents),
    }


def grade_documents(state: RAGState) -> RAGState:
    """Grade whether retrieved documents are relevant to the question"""
    question = state["question"]
    documents = state.get("documents", [])
    context = state.get("context", "")
    
    if not documents:
        logger.info("No documents retrieved. Marking as not relevant.")
        return {"is_relevant": False}
    
    grader = get_document_grader()
    is_relevant = grader.grade(question, context)
    return {"is_relevant": is_relevant}


def decide_relevance(state: RAGState) -> str:
    """Decide whether to generate answer or transform query for web search"""
    if state.get("is_relevant", False):
        return "generate"
    else:
        return "transform_query"


def transform_query(state: RAGState) -> RAGState:
    """Transform the query to optimize for web search"""
    original_question = state["question"]
    transformer = get_query_transformer()
    transformed = transformer.transform(original_question)
    return {"transformed_query": transformed}


def web_search_node(state: RAGState) -> RAGState:
    """Perform web search using transformed query"""
    query = state.get("transformed_query", state["question"])
    logger.info(f"Searching web for: '{query}'")
    
    search_tool = get_tavily_search_tool()
    results = search_tool.invoke({"query": query})
    search_results = results.get("results", [])
    
    web_documents = [
        Document(
            page_content=(
                f"Title: {item.get('title', 'Untitled')}\n"
                f"URL: {item.get('url', '')}\n"
                f"Content: {item.get('content', '')}"
            ),
            metadata={
                "source": item.get("url", ""),
                "title": item.get("title", ""),
                "type": "web_search",
            },
        )
        for item in search_results
    ]
    
    logger.info(f"Web search returned {len(web_documents)} results.")
    
    return {
        "web_search_results": web_documents,
        "documents": web_documents,
        "context": format_docs(web_documents),
    }


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


def generate_answer(state: RAGState) -> RAGState:
    context = state.get("context", "")
    llm = ChatMistralAI(model=MISTRAL_MODEL, temperature=LLM_TEMPERATURE)
    rag_chain = generation_prompt | llm | StrOutputParser()
    answer = rag_chain.invoke(
        {
            "question": state["question"],
            "context": context,
        }
    )
    return {"answer": answer}


# Build graph following the architecture from the image
graph_builder = StateGraph(RAGState)
graph_builder.add_node("retrieve_documents", retrieve_documents)
graph_builder.add_node("grade_documents", grade_documents)
graph_builder.add_node("transform_query", transform_query)
graph_builder.add_node("web_search", web_search_node)
graph_builder.add_node("generate", generate_answer)

# Add edges following the diagram
graph_builder.add_edge(START, "retrieve_documents")
graph_builder.add_edge("retrieve_documents", "grade_documents")
graph_builder.add_conditional_edges(
    "grade_documents",
    decide_relevance,
    {
        "generate": "generate",
        "transform_query": "transform_query",
    },
)
graph_builder.add_edge("transform_query", "web_search")
graph_builder.add_edge("web_search", "generate")
graph_builder.add_edge("generate", END)
rag_workflow = graph_builder.compile()


def answer_question(question: str) -> str:
    state: RAGState = {"question": question}
    result = rag_workflow.invoke(state)
    return result["answer"]


if __name__ == "__main__":
    # user_question = "What is the transformer architecture?"
    # answer = answer_question(user_question)
    # print("\nAnswer:\n")
    # print(answer)
    pass