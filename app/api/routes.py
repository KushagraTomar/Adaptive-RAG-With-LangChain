"""Routes for RAG API"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import RAGRequest, RAGResponse
from app.rag.rag import answer_question
from app.utils.logging import setup_logging

logger = setup_logging(__name__)

router = APIRouter(prefix="", tags=["rag"])

@router.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the Adaptive RAG API", "version": "1.0.0"}


@router.post("/ask", response_model=RAGResponse)
async def ask_question_endpoint(request: RAGRequest):
    """Ask a question to the RAG system"""
    try:
        logger.info(f"Received question: {request.question}")
        answer = answer_question(request.question)
        logger.info(f"Generated answer for: {request.question}")
        return RAGResponse(answer=answer, source_type="hybrid")
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
