from typing import Dict, Any, Optional, Tuple

from src.models import LLMInterface
from src.routing import BaseRouter
from src.orchestration.base_orchestrator import BaseOrchestrator
from src.orchestration.query_processor import QueryProcessor
from src.orchestration.response_parser import ResponseParser
from src.orchestration.metrics_collector import MetricsCollector
from src.orchestration.routing_strategies import RoutingStrategy, ThresholdBasedRoutingStrategy
from src.utils import setup_logging

logger = setup_logging(name="compound_ai_orchestrator")


class CompoundAIOrchestrator(BaseOrchestrator):


    def __init__(
            self,
            router: BaseRouter,
            small_llm: LLMInterface,
            large_llm: LLMInterface,
            router_confidence_threshold: float = 0.8,
            routing_strategy: Optional[RoutingStrategy] = None
    ):
        self.router = router
        self.small_llm = small_llm
        self.large_llm = large_llm

        self.query_processor = QueryProcessor()
        self.response_parser = ResponseParser()
        self.metrics_collector = MetricsCollector()

        if routing_strategy is None:
            self.routing_strategy = ThresholdBasedRoutingStrategy(
                small_llm=small_llm,
                large_llm=large_llm,
                confidence_threshold=router_confidence_threshold
            )
        else:
            self.routing_strategy = routing_strategy

        logger.info(f"Initialized Compound AI Orchestrator with: ")
        logger.info(f"\t Router: {type(router).__name__}")
        logger.info(f"\t Small LLM: {small_llm.get_model_name()}")
        logger.info(f"\t Large LLM: {large_llm.get_model_name()}")
        logger.info(f"\t Routing Strategy: {type(self.routing_strategy).__name__}")

    def process_query(
            self,
            query_id: str,
            query: str,
            choices: Dict,
            correct_answer_key: Optional[str] = None
    ) -> Dict[str, Any]:

        self.metrics_collector.start_query()

        try:
            # Step 1: Prepare router input
            router_prompt = self.query_processor.prepare_router_input(query, choices)

            # Step 2: Predict query difficulty
            self.metrics_collector.start_routing()
            difficulty, confidence = self.router.predict_difficulty(query_text=router_prompt, query_id=query_id)
            self.metrics_collector.end_routing()
            logger.debug(f"Query {query_id} classified as {difficulty} with confidence {confidence:.4f}")

            # Step 3: Select LLM based on difficulty
            chosen_llm, chosen_llm_name, size, decision_reason = self.routing_strategy.select_llm(difficulty,
                                                                                                  confidence)
            logger.debug(f"Selected {chosen_llm_name} for query {query_id}")

            # Step 4: Prepare LLM input
            llm_prompt = self.query_processor.prepare_llm_input(query, choices)

            # Step 5: Generate response
            self.metrics_collector.start_llm()
            response = chosen_llm.generate(llm_prompt)
            self.metrics_collector.end_llm()

            # Step 6: Get LLM resource usage metrics
            llm_resource_usage = chosen_llm.get_resource_usage()
            self.metrics_collector.set_llm_metrics(llm_resource_usage)

            # Step 7: Parse response
            parsed_answer = self.response_parser.parse_answer(response, choices)

            # Step 8: Evaluate correctness if correct answer provided
            is_correct = self.response_parser.evaluate_correctness(parsed_answer, correct_answer_key)

        except Exception as e:
            logger.error(f"Error processing query {query_id}: {e}")
            self.metrics_collector.end_query()

            return {
                'query_id': query_id,
                'query': query,
                'choices': choices,
                'error': str(e),
                'success': False
            }

        self.metrics_collector.end_query()
        metrics = self.metrics_collector.get_metrics()

        result = {
            'query_id': query_id,
            'query': query,
            'choices': choices,
            'prompt': llm_prompt,
            'predicted_difficulty': difficulty,
            'router_confidence': confidence,
            'chosen_llm': size,
            'chosen_llm_name': chosen_llm_name,
            'decision_reason': decision_reason,
            'response': response,
            'parsed_answer': parsed_answer,
            'correct': is_correct,
            'correct_answer_key': correct_answer_key,
            'success': True,
            **metrics,
            'resource_usage': {
                'router': {
                    'type': 'transformer',
                    'latency_ms': metrics['routing_time_ms']
                },
                'llm': llm_resource_usage
            }
        }

        return result