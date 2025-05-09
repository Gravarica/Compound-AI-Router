import time
import os
from typing import Dict, Any, Optional
import anthropic
import requests
from dotenv import load_dotenv

from llm_interface import LLMInterface

load_dotenv()

class ClaudeLLM(LLMInterface):

    def __init__(self, api_key: Optional[str] = None, model_name: str = 'claude-3-haiku-20240307'):

        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Please set the ANTHROPIC_API_KEY environment variable.")

        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self._last_resource_usage = {}

    def generate(self, prompt: str, **kwargs) -> str:

        start_time = time.time()

        try:
            params = {
                'model': self.model_name,
                'max_tokens': 1024,
                'temperature': 0.1,
            }

            params.update(kwargs)

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=params.get('max_tokens', 1024),
                temperature=params.get('temperature', 0.1),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            result = response.content[0].text

            end_time = time.time()
            self._last_resource_usage = {
                'latency_ms': round((end_time - start_time) * 1000),
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
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
        return f'claude/{self.model_name}'

    def get_resource_usage(self) -> Dict[str, Any]:
        return self._last_resource_usage