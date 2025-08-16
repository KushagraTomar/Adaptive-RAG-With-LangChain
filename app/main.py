from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Optional

# Import our RAG module
from app.rag import get_answer

app = FastAPI(title="Adaptive RAG API", description="An API for Adaptive Retrieval-Augmented Generation")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

# Pydantic model for response body
class AnswerResponse(BaseModel):
    answer: str
    question: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Adaptive RAG API"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        answer = get_answer(request.question)
        return AnswerResponse(answer=answer, question=request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)