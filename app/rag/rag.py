from typing import Any, List, TypedDict, Tuple

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
from app.cache.redis_cache import get_cache
from app.cache.question_similarity import get_similarity_matcher


logger = setup_logging(__name__)


class RAGState(TypedDict, total=False):
    question: str              # User's question
    documents: List[Document]  # Retrieved documents
    context: str               # Formatted context from retrieved documents
    answer: str                # Generated answer to the question
    is_relevant: bool          # Whether documents are relevant to question
    transformed_query: str     # Query after transformation


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


class RAGPipeline:
    """RAG Pipeline with dependency injection for flexible configuration and testing"""

    def __init__(self):
        """Initialize RAG pipeline with all dependencies"""
        self.llm = ChatMistralAI(model=MISTRAL_MODEL, temperature=LLM_TEMPERATURE)
        self.tavily_search_tool = TavilySearch(max_results=TAVILY_MAX_RESULTS)
        self.document_grader = DocumentGrader()
        self.query_transformer = QueryTransformer()
        self.retrieval_resources = build_retrieval_resources()
        self.rag_workflow = self.build_workflow()

    def build_workflow(self):
        """Build the RAG workflow graph"""
        graph_builder = StateGraph(RAGState)
        graph_builder.add_node("retrieve_documents", self.retrieve_documents)
        graph_builder.add_node("grade_documents", self.grade_documents)
        graph_builder.add_node("transform_query", self.transform_query)
        graph_builder.add_node("web_search", self.web_search_node)
        graph_builder.add_node("generate", self.generate_answer)

        # Add edges following the diagram
        graph_builder.add_edge(START, "retrieve_documents")
        graph_builder.add_edge("retrieve_documents", "grade_documents")
        graph_builder.add_conditional_edges(
            "grade_documents",
            self.decide_relevance,
            {
                "generate": "generate",
                "transform_query": "transform_query",
            },
        )
        graph_builder.add_edge("transform_query", "web_search")
        graph_builder.add_edge("web_search", "generate")
        graph_builder.add_edge("generate", END)
        
        return graph_builder.compile()

    def retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve documents from vector store"""
        question = state["question"]
        documents = self.retrieval_resources.compression_retriever.invoke(question)
        logger.info(f"Retrieved {len(documents)} documents.")
        return {
            "documents": documents,
            "context": format_docs(documents),
        }

    def grade_documents(self, state: RAGState) -> RAGState:
        """Grade whether retrieved documents are relevant to the question"""
        question = state["question"]
        documents = state.get("documents", [])
        context = state.get("context", "")
        
        if not documents:
            logger.info("No documents retrieved. Marking as not relevant.")
            return {"is_relevant": False}
        
        is_relevant = self.document_grader.grade(question, context)
        return {"is_relevant": is_relevant}

    def decide_relevance(self, state: RAGState) -> str:
        """Decide whether to generate answer or transform query for web search"""
        if state.get("is_relevant", False):
            return "generate"
        else:
            return "transform_query"

    def transform_query(self, state: RAGState) -> RAGState:
        """Transform the query to optimize for web search"""
        original_question = state["question"]
        transformed_query = self.query_transformer.transform(original_question)
        return {"transformed_query": transformed_query}

    def web_search_node(self, state: RAGState) -> RAGState:
        """Perform web search using transformed query"""
        query = state.get("transformed_query", state["question"])
        logger.info(f"Searching web for: '{query}'")
        
        results = self.tavily_search_tool.invoke({"query": query})
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

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer based on context"""
        context = state.get("context", "")
        rag_chain = generation_prompt | self.llm | StrOutputParser()
        answer = rag_chain.invoke(
            {
                "question": state["question"],
                "context": context,
            }
        )
        return {"answer": answer}

    def answer(self, question: str) -> str:
        """Answer a question using the RAG pipeline"""
        state: RAGState = {"question": question}
        result = self.rag_workflow.invoke(state)
        return result["answer"]


# Initialize singleton pipeline instance
rag_pipeline = RAGPipeline()


def answer_question(question: str) -> str:
    """Answer a question using the RAG system"""
    return rag_pipeline.answer(question)


def answer_question_with_cache(question: str) -> Tuple[str, bool]:
    """Answer a question with caching. Returns (answer, was_cached)"""
    cache = get_cache()
    matcher = get_similarity_matcher()
    
    # Generate cache key
    cache_key = matcher.get_cache_key(question)
    
    # Check cache first
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for question: {question}")
        return cached_result.get("answer", ""), True
    
    # Cache miss - generate answer
    logger.info(f"Cache miss for question: {question}")
    answer = rag_pipeline.answer(question)
    
    # Store in cache
    cache_data = {"answer": answer}
    cache.set(cache_key, cache_data)
    
    return answer, False