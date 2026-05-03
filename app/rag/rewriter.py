"""Query rewriting and transformation logic"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from config.settings import MISTRAL_MODEL, LLM_TEMPERATURE


class QueryTransformer:
    """Transforms queries to optimize for web search"""
    
    def __init__(self):
        self.llm = ChatMistralAI(model=MISTRAL_MODEL, temperature=LLM_TEMPERATURE)
    
    def transform(self, question: str) -> str:
        """
        Transform a question into a web search optimized query
        
        Args:
            question: Original question
            
        Returns:
            Transformed query
        """
        transform_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Transform the user's question into a concise search query optimized for web search. "
                "Return only the transformed query, no explanation.",
            ),
            (
                "human",
                "{question}",
            ),
        ])
        
        transform_chain = transform_prompt | self.llm | StrOutputParser()
        transformed = transform_chain.invoke({"question": question}).strip()
        
        print(f"Transform query: '{question[:50]}...' → '{transformed[:50]}...'")
        
        return transformed
