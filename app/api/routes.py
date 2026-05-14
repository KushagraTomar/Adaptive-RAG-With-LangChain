"""Routes for RAG API"""
import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import RAGRequest, RAGResponse
from app.rag.rag import answer_question, answer_question_with_cache
from app.utils.logging import setup_logging
from app.retrieval.ingestion import ingest_pdf
from app.cache.redis_cache import get_cache

logger = setup_logging(__name__)

router = APIRouter(prefix="", tags=["rag"])

# Temporary upload directory
UPLOAD_DIR = "data/temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the Adaptive RAG API", "version": "1.0.0"}


@router.post("/ask", response_model=RAGResponse)
async def ask_question_endpoint(request: RAGRequest):
    """Ask a question to the RAG system"""
    try:
        logger.info(f"Received question: {request.question}")
        answer, cached = answer_question_with_cache(request.question)
        logger.info(f"Generated answer for: {request.question} (cached={cached})")
        return RAGResponse(answer=answer, source_type="hybrid", cached=cached)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF file"""
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save temporary file
        temp_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"PDF uploaded: {file.filename}")
        
        # Ingest PDF into vector store
        ingest_pdf(temp_path)
        logger.info(f"PDF ingested successfully: {file.filename}")
        
        # Invalidate cache after new document ingestion
        cache = get_cache()
        cache.clear("rag:answer:*")
        logger.info("Cache invalidated after PDF ingestion")
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return {
            "status": "success",
            "message": f"PDF '{file.filename}' uploaded and indexed successfully!",
            "filename": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics and performance metrics"""
    cache = get_cache()
    stats = cache.get_stats()
    return {
        "status": "success",
        "cache_stats": stats
    }


@router.post("/cache/clear")
async def clear_cache():
    """Manually clear the entire cache"""
    cache = get_cache()
    cache.clear("rag:answer:*")
    logger.info("Cache manually cleared via API")
    return {
        "status": "success",
        "message": "Cache cleared successfully"
    }
