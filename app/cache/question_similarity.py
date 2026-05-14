"""Question similarity and semantic matching for cache lookup"""
import hashlib
import logging
import re
from typing import Optional, Tuple
import numpy as np
from langchain_mistralai import MistralAIEmbeddings
from config.settings import SIMILARITY_THRESHOLD, MISTRAL_API_KEY

logger = logging.getLogger(__name__)


class QuestionSimilarityMatcher:
    """Semantic question matching using embeddings"""

    def __init__(self):
        """Initialize embeddings model"""
        try:
            self.embeddings = MistralAIEmbeddings(api_key=MISTRAL_API_KEY)
            logger.info("Question similarity matcher initialized with Mistral embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            self.embeddings = None

    def normalize_question(self, question: str) -> str:
        """Normalize question for consistent hashing"""
        # Lowercase
        text = question.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove punctuation (keep alphanumeric and spaces)
        text = re.sub(r'[^\w\s]', '', text)
        print(f"Normalized question: '{text}'")
        return text

    def get_cache_key(self, question: str) -> str:
        """Generate cache key from question"""
        normalized = self.normalize_question(question)
        # Hash the normalized question
        question_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"rag:answer:{question_hash}"

    def get_embedding(self, text: str) -> Optional[list]:
        """Get embedding vector for text"""
        if not self.embeddings:
            return None
        
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    def calculate_similarity(self, vec1: list, vec2: list) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        try:
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Cosine similarity
            similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def is_similar(self, question1: str, question2: str) -> bool:
        """Check if two questions are semantically similar"""
        # First check: normalized form similarity
        norm1 = self.normalize_question(question1)
        norm2 = self.normalize_question(question2)
        
        if norm1 == norm2:
            logger.debug(f"Exact normalized match: '{question1}' ≈ '{question2}'")
            return True
        
        # Second check: embedding similarity
        emb1 = self.get_embedding(question1)
        emb2 = self.get_embedding(question2)
        
        if not emb1 or not emb2:
            return False
        
        similarity = self.calculate_similarity(emb1, emb2)
        is_match = similarity >= SIMILARITY_THRESHOLD
        
        if is_match:
            logger.debug(f"Semantic match (similarity={similarity:.2f}): '{question1}' ≈ '{question2}'")
        else:
            logger.debug(f"No semantic match (similarity={similarity:.2f}): '{question1}' vs '{question2}'")
        
        return is_match


# Global instance
_matcher_instance = None


def get_similarity_matcher() -> QuestionSimilarityMatcher:
    """Get or create global similarity matcher instance"""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = QuestionSimilarityMatcher()
    return _matcher_instance
