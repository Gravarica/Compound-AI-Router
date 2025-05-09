from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LLMInterface(ABC):

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_resource_usage(self) -> Dict[str, Any]:

        return {}