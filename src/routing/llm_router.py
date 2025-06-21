# src/routing/llm_router.py
import time
from typing import Tuple, Dict, Any, Optional
from src.routing.base_router import BaseRouter
from src.models.llm_interface import LLMInterface

class LLMRouter(BaseRouter):
    """
    A router that uses an LLM to predict query difficulty via prompting.
    """
    
    def __init__(self, llm: LLMInterface, confidence_threshold: float = 0.7):
        """
        Initialize the LLM Router.
        
        Args:
            llm: The LLM to use for difficulty prediction
            confidence_threshold: Threshold for high-confidence predictions
        """
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self._model_info = {
            "type": "LLMRouter", 
            "base_model": llm.get_model_name(),
            "confidence_threshold": confidence_threshold
        }
        
        # Create the difficulty classification prompt
        self.prompt_template = """You are a router that decides whether a small 3B language model can answer a question correctly.

CRITICAL: A small model (Llama3.2-3B) can handle:
- Basic factual recall (capitals, simple science facts)
- Elementary math (addition, basic fractions) 
- Common vocabulary and definitions
- Simple logical reasoning (1-2 steps)

A small model STRUGGLES with:
- Multi-step reasoning requiring 3+ logical steps
- Advanced scientific concepts requiring deep understanding
- Complex mathematical calculations or formulas
- Questions requiring specialized domain expertise
- Abstract reasoning or inference chains

EXAMPLES:
Q: "What is the capital of France?" → EASY (simple recall)
Q: "Which process converts sugar to energy in cells?" → EASY (basic biology fact)
Q: "If photosynthesis requires chlorophyll and sunlight, and a plant lacks chlorophyll, what happens to its energy production in relation to cellular respiration rates?" → HARD (multi-step reasoning)

Question: {question}
Choices: {choices}

Analyze: Does this require multi-step reasoning or specialized knowledge?
Answer with exactly one word: "easy" or "hard"."""

    def predict_difficulty(self, query_text: Optional[str] = None, query_id: Optional[str] = None, **kwargs) -> Tuple[str, float]:
        """
        Predict query difficulty using the LLM.
        
        Args:
            query_text: The query text to classify
            query_id: Optional query identifier
            **kwargs: Additional arguments (may contain 'choices')
            
        Returns:
            Tuple of (difficulty, confidence)
        """
        if not query_text:
            return "hard", 0.5  # Default to hard if no query provided
            
        start_time = time.time()
        
        try:
            # Extract choices if provided
            choices_text = ""
            if 'choices' in kwargs and kwargs['choices']:
                choices = kwargs['choices']
                if isinstance(choices, dict) and 'text' in choices and 'label' in choices:
                    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choices['label'], choices['text'])])
                else:
                    choices_text = str(choices)
            
            # Format the prompt
            prompt = self.prompt_template.format(
                question=query_text,
                choices=choices_text
            )
            
            # Get LLM response
            response = self.llm.generate(prompt)
            
            # Parse response
            response_lower = response.lower().strip()
            
            if "easy" in response_lower and "hard" not in response_lower:
                difficulty = "easy"
                confidence = self.confidence_threshold + 0.2  # High confidence for clear answers
            elif "hard" in response_lower and "easy" not in response_lower:
                difficulty = "hard" 
                confidence = self.confidence_threshold + 0.2
            else:
                # Ambiguous response - default to hard with low confidence
                difficulty = "hard"
                confidence = 0.5
                
            # Ensure confidence is in valid range
            confidence = min(confidence, 1.0)
            
            end_time = time.time()
            
            return difficulty, confidence
            
        except Exception as e:
            # If LLM fails, default to hard with low confidence
            return "hard", 0.3

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the router.
        """
        return self._model_info