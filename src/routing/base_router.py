from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

class BaseRouter(ABC):

    @abstractmethod
    def predict_difficulty(self, query_text: Optional[str] = None, query_id: Optional[str] = None, **kwargs) -> Tuple[str, float]:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass