import torch
from typing import Dict, List, Optional, Tuple, Union, Any

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.utils.logging import setup_logging
from src.routing.base_router import BaseRouter

logger = setup_logging(name="transformer_router")


class TransformerRouter(BaseRouter):

    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int = 2,
            device: Optional[str] = None,
            max_length: int = 512
    ):

        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.max_length = max_length
        self.label_map = {0: 'easy', 1: 'hard'}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        # Determine device
        self.mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        self.cuda_available = torch.cuda.is_available()

        if device is not None:
            self.device = torch.device(device)
        elif self.cuda_available:
            self.device = torch.device("cuda")
        elif self.mps_available:
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")

        self._load_model()

    def _load_model(self) -> None:

        try:
            logger.info(f"Loading tokenizer from {self.model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

            logger.info(f"Loading model from {self.model_name_or_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
                trust_remote_code=True
            )

            # Convert model to float32 to avoid issues with mixed precision
            for param in self.model.parameters():
                if param.data.dtype == torch.float16:
                    param.data = param.data.to(torch.float32)

            self.model = self.model.to(self.device)
            logger.info(f"Model and tokenizer loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {e}")
            raise

    def predict_difficulty(self, query_text: Optional[str] = None, query_id: Optional[str] = None, confidence_threshold: float = 0.7, **kwargs) -> Tuple[str, float]:

        if query_text is None:
            raise ValueError("TransformerRouter requires query_text for prediction.")

        try:
            inputs = self.tokenizer(
                query_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            self.model.eval()

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
                easy_prob = float(probs[0])
                hard_prob = float(probs[1])

                logger.debug(f'Easy prob: {easy_prob}, Hard prob: {hard_prob}, Threshold: {confidence_threshold}')

                if easy_prob >= confidence_threshold:
                    difficulty = 'easy'
                    confidence = easy_prob
                else:
                    difficulty = 'hard'
                    confidence = hard_prob if hard_prob > easy_prob else easy_prob

            return difficulty, confidence

        except Exception as e:
            logger.error(f"Error predicting difficulty: {e}")
            # Default to 'hard' on error for safety
            return 'hard', 0.0

    def get_model_info(self) -> Dict[str, Any]:

        return {
            "model_name": self.model_name_or_path,
            "num_labels": self.num_labels,
            "device": str(self.device),
            "max_length": self.max_length,
            "label_map": self.label_map
        }