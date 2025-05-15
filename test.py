import os
import argparse
import json
import random
import time
from typing import Dict, List, Any, Optional
import logging

from src.data import ARCDataManager
from src.models.llm_factory import LLMFactory
from src.routing import QueryRouter
from src import CompoundAISystem
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.result_utils import save_results, save_baseline_results

logger = setup_logging(name="compound_ai_system_test")


class TestConfiguration:
    """
    Manages test configuration and parameters.
    """

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize test configuration.

        Args:
            config_path: Path to configuration file
            **kwargs: Override configuration parameters
        """
        self.config = load_config(config_path)

        for key, value in kwargs.items():
            if key == 'router_model_path' and value:
                self.config['router']['model_path'] = value
            elif key == 'small_model' and value:
                self.config['small_llm']['model_name'] = value
            elif key == 'large_model_type' and value:
                self.config['large_llm']['type'] = value
            elif key == 'large_model' and value:
                self.config['large_llm']['model_name'] = value
            elif key == 'num_samples' and value:
                self.config['evaluation']['num_samples'] = value
            elif key == 'confidence_threshold' and value:
                self.config['router']['confidence_threshold'] = value
            elif key == 'output_file' and value:
                self.config['evaluation']['output_file'] = value

        output_dir = os.path.dirname(self.config['evaluation']['output_file'])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def get_router_config(self) -> Dict[str, Any]:
        """
        Get router configuration.

        Returns:
            Dictionary with router configuration
        """
        return self.config['router']

    def get_small_llm_config(self) -> Dict[str, Any]:
        """
        Get small LLM configuration.

        Returns:
            Dictionary with small LLM configuration
        """
        return self.config['small_llm']

    def get_large_llm_config(self) -> Dict[str, Any]:
        """
        Get large LLM configuration.

        Returns:
            Dictionary with large LLM configuration
        """
        return self.config['large_llm']

    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        Get evaluation configuration.

        Returns:
            Dictionary with evaluation configuration
        """
        return self.config['evaluation']


class DatasetManager:
    """
    Handles dataset loading and processing for tests.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize dataset manager.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        self.arc_manager = ARCDataManager()

    def load_evaluation_dataset(self, num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load evaluation dataset.

        Args:
            num_samples: Optional number of samples to use

        Returns:
            List of evaluation samples
        """
        logger.info("Loading ARC dataset...")
        self.arc_manager.load_data()
        eval_set = self.arc_manager.get_arc_evaluation_set()

        if num_samples and num_samples < len(eval_set):
            logger.info(f"Sampling {num_samples} examples from {len(eval_set)} total examples")
            samples = random.sample(eval_set, num_samples)
        else:
            logger.info(f"Using all {len(eval_set)} examples")
            samples = eval_set

        return samples


class CompoundAISystemEvaluator:
    """
    Handles evaluation of the Compound AI System.
    """

    def __init__(self, config: TestConfiguration):
        """
        Initialize the evaluator.

        Args:
            config: Test configuration
        """
        self.config = config

    def setup_system(self) -> CompoundAISystem:
        """
        Setup the Compound AI System.

        Returns:
            Initialized CompoundAISystem
        """
        logger.info("Setting up Compound AI System...")

        # Setup router
        router_config = self.config.get_router_config()
        router = QueryRouter(
            model_name_or_path=router_config['model_path'],
            max_length=router_config.get('max_length', 512)
        )

        # Setup small LLM
        small_llm_config = self.config.get_small_llm_config()
        small_llm = LLMFactory.create_llm(
            small_llm_config['type'],
            small_llm_config
        )

        # Setup large LLM
        large_llm_config = self.config.get_large_llm_config()
        large_llm = LLMFactory.create_llm(
            large_llm_config['type'],
            large_llm_config
        )

        # Create compound system
        compound_system = CompoundAISystem(
            router=router,
            small_llm=small_llm,
            large_llm=large_llm,
            router_confidence_threshold=router_config.get('confidence_threshold', 0.8)
        )

        return compound_system

    def evaluate_system(self, system: CompoundAISystem, eval_set: List[Dict]) -> List[Dict]:
        """
        Evaluate the Compound AI System on the evaluation set.

        Args:
            system: The Compound AI System
            eval_set: Evaluation dataset

        Returns:
            List of evaluation results
        """
        logger.info(f"Evaluating Compound AI System on {len(eval_set)} examples...")
        results = []

        for i, question_data in enumerate(eval_set):
            logger.info(f"Processing question {i + 1}/{len(eval_set)}: {question_data['id']}")

            try:
                result = system.process_query(
                    query_id=question_data['id'],
                    query=question_data['question'],
                    choices=question_data['choices'],
                    correct_answer_key=question_data['correct_answer']
                )

                # Add true difficulty for analysis
                result['true_difficulty'] = question_data['difficulty']

                # Log progress
                logger.info(
                    f"Routed to: {result['chosen_llm_name']} (predicted difficulty: {result['predicted_difficulty']})")
                logger.info(f"Correct: {result['correct']}")

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing question {question_data['id']}: {e}")

                # Add error result
                error_result = {
                    'query_id': question_data['id'],
                    'query': question_data['question'],
                    'choices': question_data['choices'],
                    'correct_answer_key': question_data['correct_answer'],
                    'true_difficulty': question_data['difficulty'],
                    'error': str(e),
                    'success': False
                }

                results.append(error_result)

        return results

    def evaluate_baseline(self, eval_set: List[Dict]) -> List[Dict]:
        """
        Evaluate baseline performance using only the large LLM.

        Args:
            eval_set: Evaluation dataset

        Returns:
            List of baseline results
        """
        logger.info(f"Evaluating baseline (large LLM only) on {len(eval_set)} examples...")

        # Setup large LLM
        large_llm_config = self.config.get_large_llm_config()
        large_llm = LLMFactory.create_llm(
            large_llm_config['type'],
            large_llm_config
        )

        results = []

        for i, question_data in enumerate(eval_set):
            logger.info(f"Processing baseline question {i + 1}/{len(eval_set)}: {question_data['id']}")

            try:
                # Prepare prompt
                from src.routing.prompt_utils import create_llm_prompt
                prompt = create_llm_prompt(question_data['question'], question_data['choices'])

                # Generate response
                start_time = time.time()
                response = large_llm.generate(prompt)
                end_time = time.time()

                # Get resource usage
                usage = large_llm.get_resource_usage()
                if 'latency_ms' in usage:
                    total_time = usage['latency_ms']
                else:
                    total_time = round((end_time - start_time) * 1000, 2)

                # Adjust for rate limit wait time
                total_time -= usage.get('rate_limit_wait_time', 0)

                # Parse answer
                from src.routing.prompt_utils import parse_answer
                parsed_answer = parse_answer(response, question_data['choices'])
                is_correct = (parsed_answer == question_data['correct_answer'])

                # Log progress
                logger.info(f"Correct: {is_correct}")

                # Create result
                result = {
                    'query_id': question_data['id'],
                    'query': question_data['question'],
                    'true_difficulty': question_data['difficulty'],
                    'model': large_llm.get_model_name(),
                    'response': response,
                    'parsed_answer': parsed_answer,
                    'correct': is_correct,
                    'correct_answer_key': question_data['correct_answer'],
                    'total_time_ms': total_time,
                    'resource_usage': large_llm.get_resource_usage()
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing baseline question {question_data['id']}: {e}")

                # Add error result
                error_result = {
                    'query_id': question_data['id'],
                    'query': question_data['question'],
                    'true_difficulty': question_data['difficulty'],
                    'model': large_llm.get_model_name(),
                    'error': str(e),
                    'success': False
                }

                results.append(error_result)

        return results


class ResultAnalyzer:
    """
    Analyzes and compares evaluation results.
    """

    def __init__(self):
        """Initialize the result analyzer."""
        pass

    def calculate_price_metrics(self, results: List[Dict], model_type: str) -> Dict[str, Any]:
        """
        Calculate pricing metrics for evaluation results.

        Args:
            results: Evaluation results
            model_type: Model type for pricing calculations

        Returns:
            Dictionary with pricing metrics
        """
        if model_type.lower() == 'claude':
            input_price_per_1m = 0.80
            output_price_per_1m = 4.00
        else:
            input_price_per_1m = 0.50
            output_price_per_1m = 1.50

        total_input_tokens = 0
        total_output_tokens = 0

        for r in results:
            if model_type.lower() == 'compound':
                if r.get('chosen_llm') == 'large' and 'resource_usage' in r and 'llm' in r['resource_usage']:
                    usage = r['resource_usage']['llm']
                    if 'prompt_tokens' in usage:
                        total_input_tokens += usage['prompt_tokens']
                    if 'completion_tokens' in usage:
                        total_output_tokens += usage['completion_tokens']
            else:
                if 'resource_usage' in r:
                    usage = r['resource_usage']
                    if 'prompt_tokens' in usage:
                        total_input_tokens += usage['prompt_tokens']
                    if 'completion_tokens' in usage:
                        total_output_tokens += usage['completion_tokens']

        input_cost = (total_input_tokens / 1000000) * input_price_per_1m
        output_cost = (total_output_tokens / 1000000) * output_price_per_1m
        total_cost = input_cost + output_cost

        return {
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        }

    def compare_results(self, compound_results: List[Dict], baseline_results: List[Dict]) -> Dict[str, Any]:
        """
        Compare compound system results with baseline results.

        Args:
            compound_results: Compound system evaluation results
            baseline_results: Baseline evaluation results

        Returns:
            Dictionary with comparison metrics
        """
        logger.info("Comparing compound system with baseline...")

        compound_accuracy = sum(1 for r in compound_results if r.get('correct')) / len(compound_results)
        baseline_accuracy = sum(1 for r in baseline_results if r.get('correct')) / len(baseline_results)

        compound_avg_time = sum(r.get('total_time_ms', 0) for r in compound_results) / len(compound_results)
        baseline_avg_time = sum(r.get('total_time_ms', 0) for r in baseline_results) / len(baseline_results)

        easy_compound = [r for r in compound_results if r.get('true_difficulty') == 'easy']
        hard_compound = [r for r in compound_results if r.get('true_difficulty') == 'hard']
        easy_baseline = [r for r in baseline_results if r.get('true_difficulty') == 'easy']
        hard_baseline = [r for r in baseline_results if r.get('true_difficulty') == 'hard']

        compound_easy_acc = sum(1 for r in easy_compound if r.get('correct')) / len(
            easy_compound) if easy_compound else 0
        compound_hard_acc = sum(1 for r in hard_compound if r.get('correct')) / len(
            hard_compound) if hard_compound else 0
        baseline_easy_acc = sum(1 for r in easy_baseline if r.get('correct')) / len(
            easy_baseline) if easy_baseline else 0
        baseline_hard_acc = sum(1 for r in hard_baseline if r.get('correct')) / len(
            hard_baseline) if hard_baseline else 0

        small_llm_usage = sum(1 for r in compound_results if r.get('chosen_llm') == 'small') / len(compound_results)

        compound_costs = self.calculate_price_metrics(compound_results, 'compound')
        baseline_costs = self.calculate_price_metrics(baseline_results, 'claude')

        router_correct = sum(1 for r in compound_results if r.get('predicted_difficulty') == r.get('true_difficulty'))
        router_accuracy = router_correct / len(compound_results) if compound_results else 0

        false_neg = sum(1 for r in compound_results
                        if r.get('true_difficulty') == 'easy' and r.get('predicted_difficulty') == 'hard')
        false_pos = sum(1 for r in compound_results
                        if r.get('true_difficulty') == 'hard' and r.get('predicted_difficulty') == 'easy')

        comparison = {
            'accuracy': {
                'compound': compound_accuracy,
                'baseline': baseline_accuracy,
                'difference': compound_accuracy - baseline_accuracy
            },
            'average_time_ms': {
                'compound': compound_avg_time,
                'baseline': baseline_avg_time,
                'speedup': baseline_avg_time / compound_avg_time if compound_avg_time > 0 else 0
            },
            'accuracy_by_difficulty': {
                'easy': {
                    'compound': compound_easy_acc,
                    'baseline': baseline_easy_acc,
                },
                'hard': {
                    'compound': compound_hard_acc,
                    'baseline': baseline_hard_acc,
                }
            },
            'resource_utilization': {
                'small_llm_usage': small_llm_usage,
            },
            'router_performance': {
                'accuracy': router_accuracy,
                'false_positives': false_pos,
                'false_negatives': false_neg
            },
            'api_costs': {
                'compound': compound_costs,
                'baseline': baseline_costs,
                'savings': {
                    'amount': baseline_costs['total_cost'] - compound_costs['total_cost'],
                    'percentage': (baseline_costs['total_cost'] - compound_costs['total_cost']) / baseline_costs[
                        'total_cost'] * 100 if baseline_costs['total_cost'] > 0 else 0
                }
            }
        }

        return comparison

    def print_comparison_summary(self, comparison: Dict[str, Any]):
        """
        Print a summary of the comparison results.

        Args:
            comparison: Comparison metrics
        """
        logger.info("\n======== PERFORMANCE COMPARISON ========")
        logger.info(f"Compound AI accuracy: {comparison['accuracy']['compound']:.2%}")
        logger.info(f"Baseline accuracy: {comparison['accuracy']['baseline']:.2%}")
        logger.info(f"Accuracy difference: {comparison['accuracy']['difference']:.2%}")
        logger.info(f"Compound AI avg time: {comparison['average_time_ms']['compound']:.2f}ms")
        logger.info(f"Baseline avg time: {comparison['average_time_ms']['baseline']:.2f}ms")
        logger.info(f"Speedup factor: {comparison['average_time_ms']['speedup']:.2f}x")
        logger.info(f"Small LLM usage: {comparison['resource_utilization']['small_llm_usage']:.2%}")

        logger.info("\nRouter Performance:")
        logger.info(f"Accuracy: {comparison['router_performance']['accuracy']:.2%}")
        logger.info(f"False positives: {comparison['router_performance']['false_positives']}")
        logger.info(f"False negatives: {comparison['router_performance']['false_negatives']}")

        logger.info("\nAPI Costs:")
        logger.info(f"Compound API cost: ${comparison['api_costs']['compound']['total_cost']:.4f}")
        logger.info(f"Baseline API cost: ${comparison['api_costs']['baseline']['total_cost']:.4f}")
        logger.info(
            f"Cost savings: ${comparison['api_costs']['savings']['amount']:.4f} ({comparison['api_costs']['savings']['percentage']:.2f}%)")


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Test compound AI system")

    parser.add_argument('--config', type=str, help='Path to configuration file')

    parser.add_argument('--router-model-path', type=str, required=False,
                        help='Path to the fine-tuned router model')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length for router')

    parser.add_argument('--small-model', default='llama3')
    parser.add_argument('--large-model-type', default='claude', choices=['ollama', 'claude'])
    parser.add_argument('--large-model', default='claude-3-haiku-20240307')

    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--baseline', action='store_true', help='Run baseline test with large LLM only')
    parser.add_argument('--output-file', default='results/cai_results.json')

    parser.add_argument('--confidence-threshold', type=float, default=0.8,
                        help='Confidence threshold for routing to the small LLM (default: 0.8)')

    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    return parser.parse_args()


def main():
    args = parse_arguments()

    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    config = TestConfiguration(
        config_path=args.config,
        router_model_path=args.router_model_path,
        small_model=args.small_model,
        large_model_type=args.large_model_type,
        large_model=args.large_model,
        num_samples=args.num_samples,
        confidence_threshold=args.confidence_threshold,
        output_file=args.output_file
    )

    dataset_manager = DatasetManager()

    print("NUM SAMPLES: ", config.get_evaluation_config().get('num_samples'))

    eval_dataset = dataset_manager.load_evaluation_dataset(
        num_samples=config.get_evaluation_config().get('num_samples')
    )

    try:
        evaluator = CompoundAISystemEvaluator(config)

        compound_system = evaluator.setup_system()

        compound_results = evaluator.evaluate_system(compound_system, eval_dataset)

        save_results(compound_results, config.get_evaluation_config().get('output_file'))

        baseline_results = None
        if args.baseline:
            random.seed(42)

            baseline_results = evaluator.evaluate_baseline(eval_dataset)

            save_baseline_results(baseline_results,
                                  config.get_evaluation_config().get('output_file').replace('.json', '_baseline.json'))

        if baseline_results:
            analyzer = ResultAnalyzer()

            comparison = analyzer.compare_results(compound_results, baseline_results)

            analyzer.print_comparison_summary(comparison)

            output = {
                'compound_results': compound_results,
                'baseline_results': baseline_results,
                'comparison': comparison,
                'test_config': config.config
            }

            with open(config.get_evaluation_config().get('output_file').replace('.json', '_full.json'), 'w') as f:
                json.dump(output, f, indent=2)

    except Exception as e:
        logger.error(f"Error testing compound AI system: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())