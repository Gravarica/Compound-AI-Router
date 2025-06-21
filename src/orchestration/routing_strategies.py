from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

from src.models import LLMInterface
from src.utils import setup_logging

logger = setup_logging(name="routing_strategies")


class RoutingStrategy(ABC):
    """
    Abstract base class for routing strategies.
    """

    @abstractmethod
    def select_llm(self, difficulty: str, confidence: float) -> Tuple[LLMInterface, str, str, str]:
        """
        Select an LLM based on the routing strategy.

        Args:
            difficulty: Predicted difficulty ('easy' or 'hard')
            confidence: Confidence in the prediction

        Returns:
            Tuple of (selected_llm, llm_name, size, decision_reason)
        """
        pass


class ThresholdBasedRoutingStrategy(RoutingStrategy):
    """
    Strategy that routes based on difficulty and confidence threshold.
    """

    def __init__(self, small_llm: LLMInterface, large_llm: LLMInterface, confidence_threshold: float = 0.8):
        """
        Initialize the strategy.

        Args:
            small_llm: The small LLM to use for easy queries
            large_llm: The large LLM to use for hard queries
            confidence_threshold: Confidence threshold for routing
        """
        self.small_llm = small_llm
        self.large_llm = large_llm
        self.confidence_threshold = confidence_threshold

    def select_llm(self, difficulty: str, confidence: float) -> Tuple[LLMInterface, str, str, str]:
        """
        Select an LLM based on confidence threshold.

        Args:
            difficulty: Predicted difficulty ('easy' or 'hard') - for logging only
            confidence: Confidence in the prediction (0.0 to 1.0)

        Returns:
            Tuple of (selected_llm, llm_name, size, decision_reason)
        """
        decision_reason = ""

        # Route based on confidence threshold, not difficulty prediction
        if confidence >= self.confidence_threshold:
            # High confidence: use small model (confident it's easy)
            decision_reason = f"High confidence ({confidence:.2%} >= {self.confidence_threshold:.2%}) - using small model"
            return self.small_llm, self.small_llm.get_model_name(), 'small', decision_reason
        else:
            # Low confidence: use large model (uncertain, err on safe side)
            decision_reason = f"Low confidence ({confidence:.2%} < {self.confidence_threshold:.2%}) - using large model"
            return self.large_llm, self.large_llm.get_model_name(), 'large', decision_reason