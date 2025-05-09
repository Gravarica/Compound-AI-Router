import time
from tokenize import cookie_re
from typing import List, Dict, Any, Optional, Tuple

from llm_interface import LLMInterface

class CompoundAISystem:

    def __init__(self, router: LLMInterface, small_llm: LLMInterface, large_llm: LLMInterface):
        self.router = router
        self.small_llm = small_llm
        self.large_llm = large_llm

    def process_query(self,
                      query_id: str,
                      query: str,
                      choices: Dict,
                      correct_answer_key: Optional[str] = None) -> Dict[str, Any]:

        start_time = time.time()

        print(choices)

        router_prompt = self._create_router_prompt(query, choices)

        router_start_time = time.time()
        difficulty = self._determine_difficulty(router_prompt)
        router_end_time = time.time()
        router_latency = round((router_end_time - router_start_time) * 1000, 2)

        llm_prompt = self._create_llm_prompt(query, choices)

        chosen_llm, chosen_llm_name, size = self._select_llm(difficulty)

        llm_start_time = time.time()
        response = chosen_llm.generate(llm_prompt)
        llm_end_time = time.time()
        llm_latency = round((llm_end_time - llm_start_time) * 1000, 2)

        parsed_answer = self._parse_answer(response, choices)

        end_time = time.time()
        total_latency = round((end_time - start_time) * 1000, 2)

        resource_usage = {
            'router': self.router.get_resource_usage(),
            'llm': chosen_llm.get_resource_usage()
        }

        is_correct = None
        if correct_answer_key is not None:
            is_correct = (parsed_answer == correct_answer_key)

        result = {
            'query_id': query_id,
            'query': query,
            'choices': choices,
            'prompt': llm_prompt,
            'predicted_difficulty': difficulty,
            'chosen_llm': size,
            'chosen_llm_name': chosen_llm_name,
            'response': response,
            'parsed_answer': parsed_answer,
            'correct': is_correct,
            'correct_answer_key': correct_answer_key,
            'total_time_ms': total_latency,
            'routing_time_ms': router_latency,
            'llm_latency_ms': llm_latency,
            'resource_usage': resource_usage,
        }

        return result

    def _create_router_prompt(self, query: str, choices: Dict) -> str:

        formatted_choices = ""
        for i, choice in enumerate(choices['text']):
            label = choices['label'][i]
            formatted_choices += f"{label}. {choice}\n"

        prompt = f"""You are an expert at analyzing questions.
        Your task is to determine if the following question is EASY or HARD.

        Question: {query}

        Choices:
        {formatted_choices}

        First, analyze the question step by step:
        1. What knowledge domains does this question involve?
        2. Does it require specialized knowledge?
        3. Does it involve multi-step reasoning?
        4. How much context or background information is needed?

        Based on your analysis, classify this question as either "EASY" or "HARD".
        Only respond with the single word "EASY" or "HARD".
        """
        return prompt

    def _determine_difficulty(self, router_prompt: str) -> str:

        router_response = self.router.generate(router_prompt)

        difficulty = 'hard'

        if 'EASY' in router_response.upper():
            difficulty = 'easy'

        return difficulty

    def _create_llm_prompt(self, query: str, choices: Dict) -> str:

        formatted_choices = ""
        for i, choice in enumerate(choices['text']):
            label = choices['label'][i]
            formatted_choices += f"{label}. {choice}\n"

        prompt = f"""Question: {query}

        Choices:
        {formatted_choices}

        Please select the correct answer. Respond with only the letter of the correct choice (A, B, C, or D).
        """

        return prompt

    def _select_llm(self, difficulty: str) -> Tuple[LLMInterface, str, str]:
        if difficulty.lower() == "easy":
            return self.small_llm, self.small_llm.get_model_name(), 'small'
        else:
            return self.large_llm, self.large_llm.get_model_name(), 'large'

    def _parse_answer(self, response: str, choices: List[str]) -> Optional[str]:
        import re

        # Get available labels from choices
        available_labels = choices['label']

        # Look for an exact match of an available label
        for label in available_labels:
            if re.search(fr'\b{label}\b', response):
                return label

        # Look for "answer is X" pattern with available labels
        for label in available_labels:
            answer_match = re.search(fr'[Aa]nswer(?:\s+is)?(?:\s*:)?\s*{label}', response)
            if answer_match:
                return label

        # Last resort: look for any of the available labels
        for label in available_labels:
            if label in response:
                return label

        # If we can't find a label, return the first label as a fallback
        return available_labels[0] if available_labels else "A"