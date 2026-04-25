import sys
from pathlib import Path
from typing import Any, List, TypedDict

from langchain_core.documents import Document
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


class RAGState(TypedDict, total=False):
    question: str             # User's question
    documents: List[Document] # Retrieved documents
    context: str              # Formatted context from retrieved documents
    answer: str               # Generated answer to the question


def format_docs(docs: List[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


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
graph_builder.add_node("retrieve_documents", retrieve_documents)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_edge(START, "retrieve_documents")
graph_builder.add_edge("retrieve_documents", "generate_answer")
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
