import sys
from pathlib import Path
from typing import Any, List, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, START, StateGraph

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion import build_retrieval_resources


retrieval_resources = build_retrieval_resources()
compression_retriever = retrieval_resources.compression_retriever

mistral_model = "mistral-large-latest"
llm = ChatMistralAI(model=mistral_model, temperature=0.1)
tavily_search_tool = TavilySearchResults(max_results=3)
router_llm = llm.bind_tools([tavily_search_tool])


class RAGState(TypedDict, total=False):
    question: str             # User's question
    documents: List[Document] # Retrieved documents
    context: str              # Formatted context from retrieved documents
    answer: str               # Generated answer to the question
    use_web_search: bool      # Whether Tavily web search should be used


def format_docs(docs: List[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def route_question(state: RAGState) -> RAGState:
    question = state["question"]
    response = router_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You decide whether a question needs Tavily web search. "
                    "Call the Tavily tool for current events, live facts, recent updates, or questions "
                    "that are likely not answerable from the local PDF knowledge base. "
                    "Do not call the tool if the question looks like it should be answered from the indexed documents."
                )
            ),
            HumanMessage(content=question),
        ]
    )

    use_web_search = bool(response.tool_calls)
    print(f"Routing question to {'web search' if use_web_search else 'local retrieval'}.")
    return {"use_web_search": use_web_search}


def route_after_decision(state: RAGState) -> str:
    return "web_search" if state.get("use_web_search", False) else "retrieve_documents"


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
        for item in results
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
graph_builder.add_edge(START, "route_question")
graph_builder.add_conditional_edges(
    "route_question",
    route_after_decision,
    {
        "retrieve_documents": "retrieve_documents",
        "web_search": "web_search",
    },
)
graph_builder.add_edge("retrieve_documents", "generate_answer")
graph_builder.add_edge("web_search", "generate_answer")
graph_builder.add_edge("generate_answer", END)
rag_workflow = graph_builder.compile()


def answer_question(question: str) -> str:
    state: RAGState = {"question": question}
    result = rag_workflow.invoke(state)
    return result["answer"]

if __name__ == "__main__":
    user_question = "explain the encoder and decoder stacks"
    answer = answer_question(user_question)
    print("\nAnswer:\n")
    print(answer)
