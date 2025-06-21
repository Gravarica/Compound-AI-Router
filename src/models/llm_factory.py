from typing import Optional, Dict, Any

from src.models import LLMInterface
from src.models import OllamaLLM
from src.models import ClaudeLLM
from src.models.openai_llm import OpenAILLM

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

        elif llm_type.lower() == 'openai':
            api_key = config.get("api_key")
            model_name = config.get("model_name", 'gpt-4o-mini')
            max_tokens = config.get("max_tokens", 1000)
            return OpenAILLM(api_key, model_name, max_tokens)

        else:
            raise ValueError(f"Invalid LLM type: {llm_type}")