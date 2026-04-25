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
tavily_search_tool = TavilySearch(max_results=3)


class RAGState(TypedDict, total=False):
    question: str             # User's question
    documents: List[Document] # Retrieved documents
    context: str              # Formatted context from retrieved documents
    answer: str               # Generated answer to the question
    use_web_search: bool      # Whether Tavily web search should be used


class RouteDecision(BaseModel):
    use_web_search: bool = Field(
        description="Whether web search is needed because retrieved " \
                    "documents are missing or insufficient."
    )


route_decision_llm = llm.with_structured_output(RouteDecision)


def format_docs(docs: List[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def route_question(state: RAGState) -> RAGState:
    documents = state.get("documents", [])
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

    print(f"Retrieved {len(documents)} documents.")
    # for i, doc in enumerate(documents, start=1):
    #     print(f"\nDocument {i}:")
    #     print(doc.page_content[:150])

    return {
        "documents": documents,
        "context": format_docs(documents),
    }


def web_search(state: RAGState) -> RAGState:
    question = state["question"]
    results = tavily_search_tool.invoke({"query": question})
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

    print(f"Tavily returned {len(web_documents)} web results.")
    return {
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

rag_chain = generation_prompt | llm | StrOutputParser()


def generate_answer(state: RAGState) -> RAGState:
    context = state.get("context", "")
    answer = rag_chain.invoke(
        {
            "question": state["question"],
            "context": context,
        }
    )
    return {"answer": answer}


graph_builder = StateGraph(RAGState)
graph_builder.add_node("route_question", route_question)
graph_builder.add_node("retrieve_documents", retrieve_documents)
graph_builder.add_node("web_search", web_search)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_edge(START, "retrieve_documents")
graph_builder.add_edge("retrieve_documents", "route_question")
graph_builder.add_conditional_edges(
    "route_question",
    route_after_decision,
    {
        "use_local_retrieval": "generate_answer",
        "use_web_search": "web_search",
    },
)
graph_builder.add_edge("web_search", "generate_answer")
graph_builder.add_edge("generate_answer", END)
rag_workflow = graph_builder.compile()


def answer_question(question: str) -> str:
    state: RAGState = {"question": question}
    result = rag_workflow.invoke(state)
    return result["answer"]

if __name__ == "__main__":
    user_question = "where is india in world map?"
    answer = answer_question(user_question)
    print("\nAnswer:\n")
    print(answer)
