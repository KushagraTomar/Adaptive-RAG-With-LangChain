"""Document grading logic"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from app.models.schemas import GradeDocuments
from config.settings import MISTRAL_MODEL, LLM_TEMPERATURE


class DocumentGrader:
    """Grades whether retrieved documents are relevant to the question"""
    
    def __init__(self):
        self.llm = ChatMistralAI(model=MISTRAL_MODEL, temperature=LLM_TEMPERATURE)
        self.grade_llm = self.llm.with_structured_output(GradeDocuments)
    
    def grade(self, question: str, context: str) -> bool:
        """
        Grade if documents are relevant to the question
        
        Args:
            question: User's question
            context: Formatted document context
            
        Returns:
            True if relevant, False otherwise
        """
        if not context or not question:
            return False
        
        grade_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are grading whether retrieved documents are relevant to the user's question. "
                "Consider documents relevant if they contain information that directly addresses the question.",
            ),
            (
                "human",
                "Question: {question}\n\nDocuments:\n{context}\n\nAre these documents relevant?",
            ),
        ])
        
        grade_chain = grade_prompt | self.grade_llm
        result = grade_chain.invoke({"question": question, "context": context})
        
        is_relevant = result.binary_score.lower() == "yes"
        print(f"Grade documents: {'relevant' if is_relevant else 'not relevant'}")
        
        return is_relevant
