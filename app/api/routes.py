"""Routes for RAG API"""
import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import RAGRequest, RAGResponse
from app.rag.rag import answer_question
from app.utils.logging import setup_logging
from app.retrieval.ingestion import ingest_pdf

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
        answer = answer_question(request.question)
        logger.info(f"Generated answer for: {request.question}")
        return RAGResponse(answer=answer, source_type="hybrid")
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
