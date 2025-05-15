from typing import Dict, Any, Optional

from src.utils import setup_logging

logger = setup_logging(name="response_parser")


class ResponseParser:
    """
    Handles parsing and processing of LLM responses.
    """

    def __init__(self):
        """Initialize the response parser."""
        pass

    def parse_answer(self, response: str, choices: Dict) -> str:
        """
        Parse the answer from the LLM response.

        Args:
            response: The raw LLM response
            choices: Dictionary of choices for multiple-choice questions

        Returns:
            Parsed answer
        """
        from src.routing import parse_answer
        return parse_answer(response, choices)

    def evaluate_correctness(self, parsed_answer: str, correct_answer_key: Optional[str]) -> bool:
        """
        Evaluate if the parsed answer is correct.

        Args:
            parsed_answer: The parsed answer
            correct_answer_key: The correct answer key

        Returns:
            True if the answer is correct, False otherwise or if correct_answer_key is None
        """
        if correct_answer_key is None:
            return None
        return parsed_answer == correct_answer_key