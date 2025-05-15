# src/orchestration/compound_ai_system.py
from typing import Dict, Any, Optional

from src.models import LLMInterface
from src.routing import QueryRouter
from src.orchestration import CompoundAIOrchestrator
from src.utils import setup_logging

logger = setup_logging(name="compound_ai_system")


class CompoundAISystem:
    """
    Compound AI system that maintains backward compatibility.
    """

    def __init__(
            self,
            router: QueryRouter,
            small_llm: LLMInterface,
            large_llm: LLMInterface,
            router_confidence_threshold: float = 0.8
    ):
        """
        Initialize the compound AI system.

        Args:
            router: The router for query difficulty classification
            small_llm: The small LLM for easy queries
            large_llm: The large LLM for hard queries
            router_confidence_threshold: Confidence threshold for router
        """
        self.router = router
        self.small_llm = small_llm
        self.large_llm = large_llm
        self.router_confidence_threshold = router_confidence_threshold

        # Create the orchestrator
        self.orchestrator = CompoundAIOrchestrator(
            router=router,
            small_llm=small_llm,
            large_llm=large_llm,
            router_confidence_threshold=router_confidence_threshold
        )

        logger.info(f"Initialized Compound AI System with: ")
        logger.info(f"\t Router: {type(router).__name__}")
        logger.info(f"\t Small LLM: {small_llm.get_model_name()}")
        logger.info(f"\t Large LLM: {large_llm.get_model_name()}")
        logger.info(f"\t Router Confidence Threshold: {router_confidence_threshold}")

    def process_query(
            self,
            query_id: str,
            query: str,
            choices: Dict,
            correct_answer_key: Optional[str] = None
    ) -> Dict[str, Any]:
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
        return self.orchestrator.process_query(query_id, query, choices, correct_answer_key)

    def _select_llm(self, difficulty: str, router_confidence: float = None) -> tuple:
        """
        Select an LLM based on difficulty - for backward compatibility.

        Args:
            difficulty: Predicted difficulty ('easy' or 'hard')
            router_confidence: Optional confidence in the prediction

        Returns:
            Tuple of (selected_llm, llm_name, size, decision_reason)
        """
        return self.orchestrator.routing_strategy.select_llm(difficulty, router_confidence)