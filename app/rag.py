import sys
from pathlib import Path
from typing import Any, List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion import build_retrieval_resources


retrieval_resources = build_retrieval_resources()
compression_retriever = retrieval_resources.compression_retriever

mistral_model = "mistral-large-latest"
llm = ChatMistralAI(model=mistral_model, temperature=0.1)
tavily_search_tool = TavilySearch(max_results=2)


class RAGState(TypedDict, total=False):
    question: str                        # User's question
    documents: List[Document]            # Retrieved documents
    context: str                         # Formatted context from retrieved documents
    answer: str                          # Generated answer to the question
    is_relevant: bool                    # Whether documents are relevant to question  
    transformed_query: str               # Query after transformation

# pydantic model defining output schema for the routing decision LLM
class RouteDecision(BaseModel):
    use_web_search: bool = Field(
        description="Whether web search is needed because retrieved " \
                    "documents are missing or insufficient to answer the user query."
    )


route_decision_llm = llm.with_structured_output(RouteDecision)

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


grade_llm = llm.with_structured_output(GradeDocuments)

def format_docs(docs: List[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def grade_documents(state: RAGState) -> RAGState:
    """Grade whether retrieved documents are relevant to the question"""
    question = state["question"]
    documents = state.get("documents", [])
    context = state.get("context", "")
    
    if not documents:
        print("No documents retrieved. Marking as not relevant.")
        return {"is_relevant": False}
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are grading whether retrieved documents are relevant to the user's question. "
                "Consider documents relevant if they contain information that directly addresses the question.",
            ),
            (
                "human",
                "Question: {question}\n\nDocuments:\n{context}\n\nAre these documents relevant?",
            ),
        ]
    )
    
    grade_chain = grade_prompt | grade_llm
    result = grade_chain.invoke({"question": question, "context": context})
    
    is_relevant = result.binary_score.lower() == "yes"
    print(f"Grade documents: {'relevant' if is_relevant else 'not relevant'}")
    
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
    
    transform_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Transform the user's question into a concise search query optimized for web search. "
                "Return only the transformed query, no explanation.",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )
    
    transform_chain = transform_prompt | llm | StrOutputParser()
    transformed = transform_chain.invoke({"question": original_question}).strip()
    
    print(f"Transform query: '{original_question[:50]}...' → '{transformed[:50]}...'")
    
    return {"transformed_query": transformed}


def web_search_node(state: RAGState) -> RAGState:
    """Perform web search using transformed query"""
    query = state.get("transformed_query", state["question"])
    
    print(f"Searching web for: '{query}'")
    results = tavily_search_tool.invoke({"query": query})
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
    
    print(f"Web search returned {len(web_documents)} results.")
    
    return {
        "documents": web_documents,
        "context": format_docs(web_documents),
    }


def route_question(state: RAGState) -> RAGState:
    context = state.get("context", "")

    decision = route_decision_llm.invoke(
        (
            "You are deciding whether retrieved local documents are sufficient to answer a question. "
            "Choose web search only when the retrieved context does not contain the answer , "
            "or the question requires recent or live information not likely to exist in static PDFs. "
            "Prefer local documents when they appear relevant enough to answer.\n\n"
            f"Question: {state['question']}\n\n"
            f"Retrieved context:\n{context}"
        )
    )

    print(
        f"Routing question to {'web search' if decision.use_web_search else 'local retrieval'}: "
    )
    return {
        "use_web_search": decision.use_web_search,
    }


def route_after_decision(state: RAGState) -> str:
    return "use_web_search" if state.get("use_web_search", False) else "use_local_retrieval"


def retrieve_documents(state: RAGState) -> RAGState:
    question = state["question"]
    documents = compression_retriever.invoke(question)

    print("Retrieved documents:")
    print(documents[0])

    print(f"Retrieved {len(documents)} documents.")

    return {
        "documents": documents,
        "context": format_docs(documents),
    }


def web_search(state: RAGState) -> RAGState:
    question = state["question"]
    results = tavily_search_tool.invoke({"query": question})
    search_results = results.get("results", [])

    web_documents = [
        # create langchain Document object
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

    print(f"Tavily returned {len(web_documents)} web results.")
    return {
        "documents": web_documents,
        "context": format_docs(web_documents),
    }


def generate_answer(state: RAGState) -> RAGState:
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

    context = state.get("context", "")
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

# Add edges for self-corrective RAG
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
    user_question = "what is el nina?"
    answer = answer_question(user_question)
    print("\nAnswer:\n")
    print(answer)
