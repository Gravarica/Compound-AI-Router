from typing import Dict, Any, Tuple

from src.utils import setup_logging

logger = setup_logging(name="query_processor")


class QueryProcessor:
    """
    Handles query preprocessing and formatting.
    """

    def __init__(self):
        """Initialize the query processor."""
        pass

    def prepare_router_input(self, query: str, choices: Dict) -> str:
        """
        Prepare input for the router.

        Args:
            query: The query text
            choices: Dictionary of choices for multiple-choice questions

        Returns:
            Formatted input for the router
        """
        from src.routing import create_bert_router_prompt
        return create_bert_router_prompt(query, choices)

    def prepare_llm_input(self, query: str, choices: Dict) -> str:
        """
        Prepare input for the LLM.

        Args:
            query: The query text
            choices: Dictionary of choices for multiple-choice questions

        Returns:
            Formatted input for the LLM
        """
        from src.routing import create_llm_prompt
        return create_llm_prompt(query, choices)