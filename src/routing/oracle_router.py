# src/routing/oracle_router.py
from typing import Tuple, Dict, Any, List, Optional

from src.routing.base_router import BaseRouter
from src.utils.logging import setup_logging

logger = setup_logging(name="oracle_router")


class OracleRouter(BaseRouter):
    """
    A router that "predicts" the difficulty of a query by looking up its
    ground truth label. This is used for establishing a theoretical
    upper-bound on performance for ablation studies.
    """

    def __init__(self, evaluation_set: List[Dict[str, Any]]):
        """
        Initialize the OracleRouter.

        Args:
            evaluation_set: The full evaluation dataset, containing 'id' and 'difficulty'.
        """
        self._model_info = {"type": "OracleRouter"}
        self.label_map = {0: 'easy', 1: 'hard'}

        # Create a lookup table from query ID to ground truth difficulty
        self.ground_truth_map = {}
        for item in evaluation_set:
            if 'id' not in item or 'label' not in item:
                logger.warning(f"Skipping item in evaluation set due to missing 'id' or 'label': {item}")
                continue
            self.ground_truth_map[item['id']] = self.label_map[item['label']]

        if not self.ground_truth_map:
            raise ValueError("Ground truth map could not be created. Ensure the evaluation set is not empty and contains 'id' and 'label' keys.")

        logger.info(f"OracleRouter initialized with {len(self.ground_truth_map)} ground truth labels.")

    def predict_difficulty(self, query_text: Optional[str] = None, query_id: Optional[str] = None, **kwargs) -> Tuple[str, float]:
        """
        Looks up the ground truth difficulty for the given query ID.

        Args:
            query_text: The query text (unused).
            query_id: The ID of the query to "predict".
            **kwargs: Additional arguments to match the interface (ignored).

        Returns:
            A tuple containing the predicted difficulty ('easy' or 'hard') and
            a confidence score (always 1.0).
        """
        if query_id is None:
            raise ValueError("OracleRouter requires query_id for prediction.")

        if query_id not in self.ground_truth_map:
            logger.error(f"Query ID '{query_id}' not found in ground truth map. Defaulting to 'hard'.")
            return 'hard', 1.0

        difficulty = self.ground_truth_map[query_id]
        return difficulty, 1.0

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the router.
        """
        return self._model_info 