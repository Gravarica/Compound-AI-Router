from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseOrchestrator(ABC):
    """
    Abstract base class defining the interface for a compound AI system orchestrator.
    """

    @abstractmethod
    def process_query(self,
                      query_id: str,
                      query: str,
                      choices: Dict,
                      correct_answer_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query and return the result.

        Args:
            query_id: Unique identifier for the query
            query: The query text
            choices: Dictionary of choices for multiple-choice questions
            correct_answer_key: Optional correct answer key for evaluation

        Returns:
            Dictionary with processing results
        """
        pass