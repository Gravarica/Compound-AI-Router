from typing import Optional, Dict, Any

from src.models import LLMInterface
from src.models import OllamaLLM
from src.models import ClaudeLLM

class LLMFactory:

    @staticmethod
    def create_llm(llm_type: str, config: Optional[Dict[str, Any]] = None) -> LLMInterface:

        config = config or {}

        if llm_type.lower() == 'ollama':
            model_name = config.get("model_name", 'llama3')
            host = config.get("host", "http://localhost:11434")
            return OllamaLLM(model_name, host)

        elif llm_type.lower() == 'claude':
            api_key = config.get("api_key")
            model_name = config.get("model_name", 'claude-3-opus-20240229')
            return ClaudeLLM(api_key, model_name)

        else:
            raise ValueError(f"Invalid LLM type: {llm_type}")