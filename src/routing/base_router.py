from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

class BaseRouter(ABC):

    @abstractmethod
    def predict_difficulty(self, query_text: str, confidence_threshold: float = 0.7) -> Tuple[str, float]:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass