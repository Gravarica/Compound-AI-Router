# src/routing/query_router.py
from typing import Dict, List, Optional, Tuple, Union, Any

from src.routing.base_router import BaseRouter
from src.routing import RouterFactory
from src.routing.training import RouterTrainer
from src.routing.evaluation import RouterEvaluator
from src.utils.logging import setup_logging

logger = setup_logging(name="query_router")


class QueryRouter:
    """
    Main router class that combines router, trainer, and evaluator.
    This class maintains backward compatibility with the original implementation.
    """

    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int = 2,
            device: Optional[str] = None,
            max_length: int = 512
    ):
        """
        Initialize the router.

        Args:
            model_name_or_path: Pretrained model name or path
            num_labels: Number of classes (default: 2 for binary classification)
            device: Device to use (default: None, auto-detect)
            max_length: Maximum sequence length for tokenization
        """
        config = {
            'model_name_or_path': model_name_or_path,
            'num_labels': num_labels,
            'device': device,
            'max_length': max_length
        }

        self.router = RouterFactory.create_router('transformer', config)
        self.trainer = RouterTrainer(self.router)
        self.evaluator = RouterEvaluator(self.router)

        # For backward compatibility
        self.model = self.router.model
        self.tokenizer = self.router.tokenizer
        self.device = self.router.device
        self.label_map = self.router.label_map
        self.max_length = self.router.max_length

    def predict_difficulty(self, query_text: Optional[str] = None, query_id: Optional[str] = None, **kwargs) -> Tuple[str, float]:
        """
        Predict the difficulty of a query.

        Args:
            query_text: The query text to analyze
            query_id: The ID of the query (for oracle router)
            **kwargs: Additional arguments for the underlying router (e.g., confidence_threshold)

        Returns:
            Tuple of (difficulty, confidence)
        """
        return self.router.predict_difficulty(query_text=query_text, query_id=query_id, **kwargs)

    def fine_tune(self, train_data, val_data, output_dir, **kwargs):
        """
        Fine-tune the router on the provided data.

        Args:
            train_data: Training data
            val_data: Validation data
            output_dir: Directory to save the fine-tuned model
            **kwargs: Additional arguments for fine-tuning
        """
        return self.trainer.fine_tune(train_data, val_data, output_dir, **kwargs)

    def evaluate_router(self, test_data):
        """
        Evaluate the router on test data.

        Args:
            test_data: Test data

        Returns:
            Dictionary with evaluation results
        """
        return self.evaluator.evaluate_router(test_data)

    def load_fine_tuned_model(self, model_path: str) -> None:
        """
        Load a fine-tuned model.

        Args:
            model_path: Path to the fine-tuned model
        """
        # Re-initialize the router with the fine-tuned model
        config = {
            'model_name_or_path': model_path,
            'num_labels': self.router.num_labels,
            'device': self.router.device,
            'max_length': self.router.max_length
        }

        self.router = RouterFactory.create_router('transformer', config)
        self.trainer = RouterTrainer(self.router)
        self.evaluator = RouterEvaluator(self.router)

        # Update references for backward compatibility
        self.model = self.router.model
        self.tokenizer = self.router.tokenizer
        self.device = self.router.device
        self.label_map = self.router.label_map

        logger.info(f"Fine-tuned model loaded from {model_path}")

    def analyze_confidence_distribution(self, test_data):
        """
        Analyze confidence distribution.

        Args:
            test_data: Test data

        Returns:
            Dictionary with analysis results
        """
        return self.evaluator.analyze_confidence_distribution(test_data)