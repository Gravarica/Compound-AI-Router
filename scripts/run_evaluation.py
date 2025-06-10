import os
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import json
from typing import Dict, Any, List

from src.data.dataloader import ARCDataManager
from src.models import LLMFactory
from src.routing import RouterFactory
from src.orchestration import CompoundAIOrchestrator, routing_strategies
from src.utils import setup_logging, save_results, save_baseline_results, result_utils
from src.orchestration.query_processor import QueryProcessor
from src.orchestration.response_parser import ResponseParser
from src.orchestration.metrics_collector import MetricsCollector

logger = setup_logging(name="evaluation_script")

def run_baseline_evaluation(cfg: DictConfig, llm, eval_set: List[Dict[str, Any]]):
    """Runs evaluation on a single baseline LLM."""
    logger.info(f"Running baseline evaluation for: {llm.get_model_name()}")

    results = []
    query_processor = QueryProcessor()
    response_parser = ResponseParser()
    metrics_collector = MetricsCollector()

    for item in tqdm(eval_set, desc=f"Evaluating Baseline {llm.get_model_name()}"):
        metrics_collector.start_query()
        prompt = query_processor.prepare_llm_input(item['question'], item['choices'])

        try:
            metrics_collector.start_llm()
            response = llm.generate(prompt)
            metrics_collector.end_llm()

            llm_resource_usage = llm.get_resource_usage()
            metrics_collector.set_llm_metrics(llm_resource_usage)

            parsed_answer = response_parser.parse_answer(response, item['choices'])
            is_correct = response_parser.evaluate_correctness(parsed_answer, item['correct_answer'])

            metrics_collector.end_query()
            metrics = metrics_collector.get_metrics()

            result = {
                'query_id': item['id'],
                'true_difficulty': item['difficulty'],
                'response': response,
                'parsed_answer': parsed_answer,
                'correct': is_correct,
                'correct_answer_key': item['correct_answer'],
                'success': True,
                **metrics,
                'resource_usage': llm_resource_usage
            }
            results.append(result)

        except Exception as e:
            logger.error(f"Error processing query {item['id']} for baseline: {e}")
            results.append({'query_id': item['id'], 'success': False, 'error': str(e)})

    return results


def run_compound_evaluation(cfg: DictConfig, orchestrator, eval_set: List[Dict[str, Any]]):
    """Runs evaluation on the Compound AI system."""
    logger.info("Running compound system evaluation...")
    results = []
    for item in tqdm(eval_set, desc="Evaluating Compound System"):
        result = orchestrator.process_query(
            query_id=item['id'],
            query=item['question'],
            choices=item['choices'],
            correct_answer_key=item['correct_answer']
        )
        # Add true difficulty for analysis
        result['true_difficulty'] = item['difficulty']
        results.append(result)
    return results


@hydra.main(config_path="../configs/evaluation", config_name="default", version_base=None)
def main(cfg: DictConfig):
    logger.info("Starting evaluation run...")
    logger.info(f"Hydra CWD: {os.getcwd()}")
    logger.info(f"Original CWD: {hydra.utils.get_original_cwd()}")
    os.chdir(hydra.utils.get_original_cwd())
    logger.info(f"Changed CWD to: {os.getcwd()}")


    logger.info("Loading ARC data...")
    manager = ARCDataManager()
    # We need the full test set for the Oracle Router and for consistent evaluation
    eval_set = manager.get_arc_evaluation_set()

    # Sub-sample if configured
    if cfg.evaluation.num_samples and cfg.evaluation.num_samples < len(eval_set):
        random.seed(cfg.evaluation.seed)
        eval_set = random.sample(eval_set, cfg.evaluation.num_samples)
    logger.info(f"Evaluation set size: {len(eval_set)}")

    # --- Run evaluation based on mode ---
    results = []
    if cfg.run_mode == 'baseline':
        logger.info("Initializing baseline LLM...")
        baseline_llm = LLMFactory.create_llm(cfg.baseline_llm.type, OmegaConf.to_container(cfg.baseline_llm, resolve=True))
        results = run_baseline_evaluation(cfg, baseline_llm, eval_set)
        
        output_file = cfg.evaluation.output_file.replace('.json', '_full.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result_utils.save_baseline_results(results, output_file)

    elif cfg.run_mode == 'compound':
        logger.info("Initializing Compound AI System components...")
        small_llm = LLMFactory.create_llm(cfg.small_llm.type, OmegaConf.to_container(cfg.small_llm, resolve=True))
        large_llm = LLMFactory.create_llm(cfg.large_llm.type, OmegaConf.to_container(cfg.large_llm, resolve=True))

        router_config = OmegaConf.to_container(cfg.router, resolve=True)
        if cfg.router.type == 'oracle':
            # The OracleRouter needs the evaluation set to know the ground truth
            router_config['evaluation_set'] = eval_set
        
        router = RouterFactory.create_router(cfg.router.type, router_config)
        
        routing_strategy = routing_strategies.ThresholdBasedRoutingStrategy(
            small_llm=small_llm,
            large_llm=large_llm,
            confidence_threshold=cfg.router.confidence_threshold
        )

        orchestrator = CompoundAIOrchestrator(
            router=router,
            small_llm=small_llm,
            large_llm=large_llm,
            routing_strategy=routing_strategy
        )

        results = run_compound_evaluation(cfg, orchestrator, eval_set)

        output_file = cfg.evaluation.output_file.replace('.json', '_full.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        result_utils.save_results(results, output_file)

    else:
        raise ValueError(f"Invalid run_mode: {cfg.run_mode}")
    
    logger.info(f"Evaluation complete. Results saved to {cfg.evaluation.output_file.replace('.json', '_full.json')}")


if __name__ == "__main__":
    main() 