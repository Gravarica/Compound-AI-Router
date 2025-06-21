# src/models/openai_llm.py
import time
import openai
from typing import Dict, Any, Optional

from src.models.llm_interface import LLMInterface

class OpenAILLM(LLMInterface):
    """
    Interface for OpenAI GPT models via API.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", max_tokens: int = 1000):
        """
        Initialize the OpenAI LLM.
        
        Args:
            api_key: OpenAI API key
            model_name: Model name (e.g., 'gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo')
            max_tokens: Maximum tokens to generate
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._last_resource_usage = {}

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using OpenAI API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the API call
            
        Returns:
            Generated text response
        """
        start_time = time.time()
        
        try:
            # Prepare API parameters
            api_params = {
                'model': self.model_name,
                'messages': [{"role": "user", "content": prompt}],
                'max_tokens': self.max_tokens,
                'temperature': 0.1  # Low temperature for consistent responses
            }
            
            # Override with any provided kwargs
            api_params.update(kwargs)
            
            # Make API call
            response = self.client.chat.completions.create(**api_params)
            
            end_time = time.time()
            
            # Extract response text
            result = response.choices[0].message.content
            
            # Track resource usage
            usage = response.usage
            self._last_resource_usage = {
                'latency_ms': round((end_time - start_time) * 1000, 2),
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
                'rate_limit_wait_time': 0  # Could track this if needed
            }
            
            return result
            
        except Exception as e:
            end_time = time.time()
            self._last_resource_usage = {
                'latency_ms': round((end_time - start_time) * 1000, 2),
                'error': str(e)
            }
            raise Exception(f"Error generating response with {self.model_name}: {e}")

    def get_model_name(self) -> str:
        """
        Return the model name.
        """
        return f'openai/{self.model_name}'

    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Return resource usage from the last API call.
        """
        return self._last_resource_usage