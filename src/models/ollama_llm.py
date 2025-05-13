import time
import requests
import ollama
from typing import Dict, Any, Optional

from src.models.llm_interface import LLMInterface

class OllamaLLM(LLMInterface):

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self._last_resource_usage = {}

        try:
            if host != "http://localhost:11434":
                ollama.BASE_URL = host

            ollama.list()

        except Exception as e:
            print(f"Warning: Could not connect to Ollama at {host}: {e}")
            print("Make sure Ollama is running and the model is pulled.")

    def generate(self, prompt: str, **kwargs) -> str:
        start_time = time.time()

        try:
            params = {
                'stream': False
            }

            params.update(kwargs)

            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **params
            )

            result = response['message']['content']

            end_time = time.time()

            prompt_tokens = response.get('prompt_eval_count', 0)
            completion_tokens = response.get('completion_eval_count', 0)

            self._last_resource_usage = {
                'latency_ms': round((end_time - start_time) * 1000, 2),
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            }

            return result

        except Exception as e:
            end_time = time.time()
            self._last_resource_usage = {
                'latency_ms': round((end_time - start_time) * 1000, 2),
                'error': str(e)
            }
            raise Exception(f"Error generating response: ({self.model_name}): {e}")

    def get_model_name(self) -> str:
        return f'ollama/{self.model_name}'

    def get_resource_usage(self) -> Dict[str, Any]:
        return self._last_resource_usage