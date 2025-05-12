import time
from tokenize import cookie_re
from typing import List, Dict, Any, Optional, Tuple

from llm_interface import LLMInterface
from query_router import QueryRouter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompoundAISystem:

    def __init__(self,
                 router: QueryRouter,
                 small_llm: LLMInterface,
                 large_llm: LLMInterface,
                 router_confidence_threshold: float = 0.7):
        self.router = router
        self.small_llm = small_llm
        self.large_llm = large_llm
        self.router_confidence_threshold = router_confidence_threshold
        logger.info(f"Initialized Compound AI System with: ")
        logger.info(f"\t Router: {type(router).__name__}")
        logger.info(f"\t Small LLM: {small_llm.get_model_name()}")
        logger.info(f"\t Large LLM: {large_llm.get_model_name()}")
        logger.info(f"\t Router Confidence Threshold: {router_confidence_threshold}")

    def process_query(self,
                      query_id: str,
                      query: str,
                      choices: Dict,
                      correct_answer_key: Optional[str] = None) -> Dict[str, Any]:

        start_time = time.time()

        print(choices)

        router_prompt = self._create_router_prompt(query, choices)

        router_start_time = time.time()
        difficulty, confidence = self._determine_difficulty(router_prompt)
        router_end_time = time.time()
        router_latency = round((router_end_time - router_start_time) * 1000, 2)

        llm_prompt = self._create_llm_prompt(query, choices)

        chosen_llm, chosen_llm_name, size, decision_reason = self._select_llm(difficulty)

        llm_start_time = time.time()
        response = chosen_llm.generate(llm_prompt)
        llm_end_time = time.time()
        llm_latency = round((llm_end_time - llm_start_time) * 1000, 2)

        parsed_answer = self._parse_answer(response, choices)

        end_time = time.time()
        total_latency = round((end_time - start_time) * 1000, 2)

        resource_usage = {
            'router': {
                'type': 'transformer',
                'latency_ms': router_latency
            },
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
            'decision_reason': decision_reason,
            'response': response,
            'parsed_answer': parsed_answer,
            'correct': is_correct,
            'correct_answer_key': correct_answer_key,
            'total_time_ms': total_latency,
            'routing_time_ms': router_latency,
            'llm_latency_ms': llm_latency,
            'resource_usage': resource_usage,
        }

        logger.debug(f"Processed query {query_id}: difficulty={difficulty}, confidence={confidence:.4f}, chosen={size}")

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

    def _determine_difficulty(self, router_prompt: str) -> Tuple[str, float]:

        import torch

        inputs = self.router.tokenizer(
            router_prompt,
            truncation=True,
            max_length=self.router.max_length,
            return_tensors="pt",
            padding="max_length"
        ).to(self.router.device)

        self.router.model.eval()
        with torch.no_grad():
            outputs = self.router.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
            predicted_class = torch.argmax(logits, dim=1).item()

        confidence = float(probs[predicted_class])

        difficulty = self.router.label_map.get(predicted_class, 'hard')

        return difficulty, confidence

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

    def _select_llm(self, difficulty: str, confidence: float) -> Tuple[LLMInterface, str, str, str]:

        # Routing algorithm:
        # 1. Use small LLM for confident easy questions
        # 2. Use large LLM for easy questions with low confidence
        # 3. Use large LLM for hard questions


        decision_reason = ""

        if difficulty.lower() == "easy" and confidence >= self.router_confidence_threshold:
            decision_reason = f"Question classified as EASY with high confidence ({confidence:.4f})"
            return self.small_llm, self.small_llm.get_model_name(), 'small', decision_reason
        elif difficulty.lower() == "easy" and confidence < self.router_confidence_threshold:
            decision_reason = f"Question classified as EASY but with low confidence ({confidence:.4f})"
            return self.large_llm, self.large_llm.get_model_name(), 'large', decision_reason
        else:
            decision_reason = f"Question classified as HARD with confidence ({confidence:.4f})"
            return self.large_llm, self.large_llm.get_model_name(), 'large', decision_reason


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