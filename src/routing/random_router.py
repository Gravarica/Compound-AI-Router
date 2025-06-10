# src/routing/random_router.py
import random
from typing import Tuple, Dict, Any
from src.routing.base_router import BaseRouter

class RandomRouter(BaseRouter):
    """
    A router that randomly assigns a difficulty to a query for baseline testing.
    """
    def __init__(self, seed: int = None):
        """
        Initialize the RandomRouter.
        
        Args:
            seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        self._model_info = {"type": "RandomRouter", "seed": seed}

    def predict_difficulty(self, query_text: str, confidence_threshold: float = 0.5) -> Tuple[str, float]:
        """
        Randomly predicts the difficulty of a query as 'easy' or 'hard'.

        Args:
            query_text: The query text (unused in this router).
            confidence_threshold: The threshold for prediction (unused in this router).

        Returns:
            A tuple containing the predicted difficulty ('easy' or 'hard') and a
            confidence score (always 1.0).
        """
        prediction = random.choice(['easy', 'hard'])
        return prediction, 1.0

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the router.
        """
        return self._model_info 