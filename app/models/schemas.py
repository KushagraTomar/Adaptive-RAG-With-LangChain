"""Pydantic models for request/response validation"""
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class GradeDocuments(BaseModel):
    """LLM output for grading document relevance"""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class RAGRequest(BaseModel):
    """Request model for RAG query"""
    question: str = Field(..., description="The user's question")


class RAGResponse(BaseModel):
    """Response model for RAG answer"""
    answer: str = Field(..., description="The generated answer")
    documents: Optional[List[dict]] = Field(None, description="Retrieved documents")
    source_type: Optional[str] = Field(None, description="Type of source: local_pdf or web_search")
