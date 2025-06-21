# src/routing/cot_llm_router.py
import time
import re
from typing import Tuple, Dict, Any, Optional
from src.routing.base_router import BaseRouter
from src.models.llm_interface import LLMInterface

class ChainOfThoughtLLMRouter(BaseRouter):
    """
    A router that uses chain-of-thought reasoning for better difficulty classification.
    """
    
    def __init__(self, llm: LLMInterface, confidence_threshold: float = 0.7):
        """
        Initialize the Chain-of-Thought LLM Router.
        
        Args:
            llm: The LLM to use for difficulty prediction
            confidence_threshold: Threshold for high-confidence predictions
        """
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self._model_info = {
            "type": "ChainOfThoughtLLMRouter", 
            "base_model": llm.get_model_name(),
            "confidence_threshold": confidence_threshold
        }
        
        # Create the chain-of-thought difficulty classification prompt
        self.prompt_template = """You need to decide if a 3B parameter language model (Llama3.2-3B) can answer this question correctly.

Think through this step by step:

1. QUESTION ANALYSIS: What type of question is this?
2. KNOWLEDGE REQUIRED: What knowledge/skills does it need?
3. REASONING COMPLEXITY: How many logical steps are required?
4. SMALL MODEL CAPABILITY: Can a 3B model handle this?

DECISION CRITERIA:
- EASY: Simple recall, basic facts, 1-2 step reasoning, elementary concepts
- HARD: Multi-step reasoning (3+ steps), specialized knowledge, complex inference, advanced concepts

Question: {question}
Choices: {choices}

Let me analyze this step by step:

1. QUESTION ANALYSIS:
[Analyze what type of question this is]

2. KNOWLEDGE REQUIRED:
[What knowledge is needed?]

3. REASONING COMPLEXITY:
[How many logical steps?]

4. SMALL MODEL CAPABILITY:
[Can a 3B model handle this?]

FINAL CLASSIFICATION: [easy/hard]"""

    def predict_difficulty(self, query_text: Optional[str] = None, query_id: Optional[str] = None, **kwargs) -> Tuple[str, float]:
        """
        Predict query difficulty using chain-of-thought reasoning.
        
        Args:
            query_text: The query text to classify
            query_id: Optional query identifier
            **kwargs: Additional arguments (may contain 'choices')
            
        Returns:
            Tuple of (difficulty, confidence)
        """
        if not query_text:
            return "hard", 0.5  # Default to hard if no query provided
            
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
            
            # Parse the final classification from the response
            response_lower = response.lower()
            
            # Look for "FINAL CLASSIFICATION:" or final easy/hard decision
            final_classification_match = re.search(r'final classification:\s*(easy|hard)', response_lower)
            if final_classification_match:
                difficulty = final_classification_match.group(1)
                confidence = 0.9  # High confidence when following the format
            else:
                # Fallback: look for last occurrence of easy/hard
                easy_matches = [(m.start(), 'easy') for m in re.finditer(r'\beasily?\b', response_lower)]
                hard_matches = [(m.start(), 'hard') for m in re.finditer(r'\bhard\b', response_lower)]
                
                all_matches = easy_matches + hard_matches
                if all_matches:
                    # Get the last match
                    last_match = max(all_matches, key=lambda x: x[0])
                    difficulty = last_match[1]
                    confidence = 0.7
                else:
                    # Default to hard with low confidence
                    difficulty = "hard"
                    confidence = 0.5
                    
            # Adjust confidence based on reasoning quality
            if 'step' in response_lower and 'analysis' in response_lower:
                confidence = min(confidence + 0.1, 1.0)  # Bonus for structured reasoning
            
            return difficulty, confidence
            
        except Exception as e:
            # If LLM fails, default to hard with low confidence
            return "hard", 0.3

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the router.
        """
        return self._model_info