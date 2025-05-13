import os
import argparse
import json
from typing import Dict, Any, List
import time
import random

from query_router import QueryRouter
from util import save_results, save_baseline_results

from dotenv import load_dotenv

from llm_factory import LLMFactory
from compound_ai_system import CompoundAISystem
from dataloader import ARCDataManager

from query_util import parse_answer, create_llm_prompt

load_dotenv()

CLAUDE_HAIKU_INPUT_PRICE = 0.80
CLAUDE_HAIKU_OUTPUT_PRICE = 4.00

def setup_compound_ai_system(args):

    #router_config = {
    #    'model_name': args.router_model,
    #    'host': os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    #}

    #router_llm = LLMFactory.create_llm('ollama', router_config)

    router = QueryRouter(
        model_name_or_path=args.router_model_path,
        max_length=args.max_length
    )

    small_llm_config = {
        'model_name': args.small_model,
        'host': os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    }

    small_llm = LLMFactory.create_llm('ollama', small_llm_config)

    if args.large_model_type == 'claude':
        claude_api_key = os.getenv('ANTHROPIC_API_KEY')
        if not claude_api_key:
            raise ValueError("Anthropic API key not provided. Please set the ANTHROPIC_API_KEY environment variable.")
        large_llm_config = {
            "api_key": claude_api_key,
            "model_name": args.large_model
        }

        large_llm = LLMFactory.create_llm('claude', large_llm_config)

    else:
        large_llm_config = {
            'model_name': args.large_model,
            'host': os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        }

        large_llm = LLMFactory.create_llm('ollama', large_llm_config)

    return CompoundAISystem(
        router = router,
        small_llm = small_llm,
        large_llm = large_llm
    )

def test_with_arc_samples(compound_ai: CompoundAISystem, num_samples = 5):

    print("Loading ARC dataset...")

    manager = ARCDataManager()
    manager.load_data()

    eval_set = manager.get_arc_evaluation_set()

    samples = random.sample(eval_set, min(num_samples, len(eval_set)))

    results = []

    print(f"\n Testing with {len(samples)} ARC samples...")
    for i, question_data in enumerate(samples):

        print(f"\n------- Question {i+1}/{len(samples)} -------")

        question = question_data['question']
        choices = question_data['choices']
        correct_answer = question_data['correct_answer']
        qid = question_data['id']
        true_difficulty = question_data['difficulty']

        print(f"Question ID: {qid}")
        print(f"Question: {question}")
        print(f"Choices: ")
        for i in range(len(choices['text'])):
            print(f"\t {choices['label'][i]}. {choices['text'][i]}")
        print(f"Difficulty: {true_difficulty}")
        print(f"Correct answer: {correct_answer}")

        print("PROCESSING QUERY")

        result = compound_ai.process_query(
            query_id = qid,
            query = question,
            choices=choices,
            correct_answer_key=correct_answer
        )

        print("PROCESSED QUERY")

        result['true_difficulty'] = true_difficulty

        print(f"Routed to: {result['chosen_llm_name']} (predicted difficulty: {result['predicted_difficulty']})")
        print(f"Response: {result['response']}")
        print(f"Parsed answer: {result['parsed_answer']}")
        print(f"Correct: {result['correct']}")
        print(f"Routing time: {result['routing_time_ms']}ms")
        print(f"LLM inference time: {result['llm_latency_ms']}ms")
        print(f"Total time: {result['total_time_ms']}ms")

        results.append(result)

    return results

def run_baseline_test(args, num_samples = 5):
    print("\n Running baseline test with Large LLM only")

    if args.large_model_type == 'claude':
        claude_api_key = os.getenv('ANTHROPIC_API_KEY')
        if not claude_api_key:
            raise ValueError("Anthropic API key not provided. Please set the ANTHROPIC_API_KEY environment variable.")
        large_llm_config = {
            "api_key": claude_api_key,
            "model_name": args.large_model
        }
        large_llm = LLMFactory.create_llm('claude', large_llm_config)
    else:
        large_llm_config = {
            'model_name': args.large_model,
            'host': os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        }
        large_llm = LLMFactory.create_llm('ollama', large_llm_config)

    manager = ARCDataManager()
    manager.load_data()
    eval_set = manager.get_arc_evaluation_set()

    random.seed(42)
    samples = random.sample(eval_set, min(num_samples, len(eval_set)))

    results = []

    print(f"\n Testing baseline with {len(samples)} ARC samples...")
    for i, question_data in enumerate(samples):

        print(f"\n ------- Question {i+1}/{len(samples)} -------")

        question = question_data['question']
        choices = question_data['choices']
        correct_answer = question_data['correct_answer']
        qid = question_data['id']
        true_difficulty = question_data['difficulty']

        print(f"Question ID: {qid}")
        print(f"Question: {question}")
        print(f"True difficulty: {true_difficulty}")

        prompt = create_llm_prompt(question, choices)
        start_time = time.time()
        response = large_llm.generate(prompt)
        end_time = time.time()

        usage = large_llm.get_resource_usage()
        if 'latency_ms' in usage:
            total_time = usage['latency_ms']
        else:
            total_time = round((end_time - start_time) * 1000, 2)

        total_time -= usage['rate_limit_wait_time']

        parsed_answer = parse_answer(response, choices)
        is_correct = (parsed_answer == correct_answer)

        print(f"Response: {response}")
        print(f"Parsed answer: {parsed_answer}")
        print(f"Correct: {is_correct}")
        print(f"Total time: {total_time}ms")

        result = {
            'query_id': qid,
            'query': question,
            'true_difficulty': true_difficulty,
            'model': large_llm.get_model_name(),
            'response': response,
            'parsed_answer': parsed_answer,
            'correct': is_correct,
            'correct_answer_key': correct_answer,
            'total_time_ms': total_time,
            'resource_usage': large_llm.get_resource_usage()
        }

        results.append(result)

    return results


def calculate_costs_compound(results, price_per_1m_input_tokens, price_per_1m_output_tokens):
    total_input_tokens = 0
    total_output_tokens = 0

    for r in results:
        if 'resource_usage' in r and 'llm' in r['resource_usage']:
            usage = r['resource_usage']['llm']
            if 'prompt_tokens' in usage:
                total_input_tokens += usage['prompt_tokens']
            if 'completion_tokens' in usage:
                total_output_tokens += usage['completion_tokens']

    input_cost = (total_input_tokens / 1000000) * price_per_1m_input_tokens
    output_cost = (total_output_tokens / 1000000) * price_per_1m_output_tokens
    total_cost = input_cost + output_cost

    return {
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost
    }

def calculate_costs_baseline(results, price_per_1m_input_tokens, price_per_1m_output_tokens):
    total_input_tokens = 0
    total_output_tokens = 0

    for r in results:
        if 'resource_usage' in r:
            usage = r['resource_usage']
            if 'prompt_tokens' in usage:
                total_input_tokens += usage['prompt_tokens']
            if 'completion_tokens' in usage:
                total_output_tokens += usage['completion_tokens']

    input_cost = (total_input_tokens / 1000000) * price_per_1m_input_tokens
    output_cost = (total_output_tokens / 1000000) * price_per_1m_output_tokens
    total_cost = input_cost + output_cost

    return {
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost
    }

def compare_results(compound_ai_results, baseline_results):

    compound_accuracy = sum(1 for r in compound_ai_results if r['correct']) / len(compound_ai_results)
    baseline_accuracy = sum(1 for r in baseline_results if r['correct']) / len(baseline_results)

    compound_avg_time = sum(r['total_time_ms'] for r in compound_ai_results) / len(compound_ai_results)
    baseline_avg_time = sum(r['total_time_ms'] for r in baseline_results) / len(baseline_results)

    easy_compound = [r for r in compound_ai_results if r['true_difficulty'] == 'easy']
    hard_compound = [r for r in compound_ai_results if r['true_difficulty'] == 'hard']
    easy_baseline = [r for r in baseline_results if r['true_difficulty'] == 'easy']
    hard_baseline = [r for r in baseline_results if r['true_difficulty'] == 'hard']

    compound_easy_acc = sum(1 for r in easy_compound if r['correct']) / len(easy_compound) if easy_compound else 0
    compound_hard_acc = sum(1 for r in hard_compound if r['correct']) / len(hard_compound) if hard_compound else 0
    baseline_easy_acc = sum(1 for r in easy_baseline if r['correct']) / len(easy_baseline) if easy_baseline else 0
    baseline_hard_acc = sum(1 for r in hard_baseline if r['correct']) / len(hard_baseline) if hard_baseline else 0

    small_llm_usage = sum(1 for r in compound_ai_results if r.get('chosen_llm') == 'small') / len(compound_ai_results)

    compound_costs = calculate_costs_compound([r for r in compound_ai_results if r.get('chosen_llm') == 'large'],
                                     CLAUDE_HAIKU_INPUT_PRICE, CLAUDE_HAIKU_OUTPUT_PRICE)
    print(compound_costs)
    baseline_costs = calculate_costs_baseline(baseline_results,
                                     CLAUDE_HAIKU_INPUT_PRICE, CLAUDE_HAIKU_OUTPUT_PRICE)
    print(baseline_costs)
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
        'api_costs': {
            'compound': compound_costs,
            'baseline': baseline_costs,
            'savings': {
                'amount': baseline_costs['total_cost'] - compound_costs['total_cost'],
                'percentage': (baseline_costs['total_cost'] - compound_costs['total_cost']) / baseline_costs['total_cost'] * 100 if baseline_costs['total_cost'] > 0 else 0
            }
        }
    }

    return comparison


def analyze_router_predictions(compound_results):

    correct_predictions = sum(1 for r in compound_results
                              if r['predicted_difficulty'] == r['true_difficulty'])

    router_accuracy = correct_predictions / len(compound_results)

    false_positives = sum(1 for r in compound_results
                          if r['true_difficulty'] == 'hard' and r['predicted_difficulty'] == 'easy')
    false_negatives = sum(1 for r in compound_results
                          if r['true_difficulty'] == 'easy' and r['predicted_difficulty'] == 'hard')

    true_positives = sum(1 for r in compound_results
                         if r['true_difficulty'] == 'hard' and r['predicted_difficulty'] == 'hard')
    true_negatives = sum(1 for r in compound_results
                         if r['true_difficulty'] == 'easy' and r['predicted_difficulty'] == 'easy')

    hard_count = sum(1 for r in compound_results if r['true_difficulty'] == 'hard')

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / hard_count if hard_count > 0 else 0

    correct_routing_easy = sum(1 for r in compound_results
                               if r['true_difficulty'] == 'easy' and
                               r['predicted_difficulty'] == 'easy' and
                               r['correct'])
    correct_routing_hard = sum(1 for r in compound_results
                               if r['true_difficulty'] == 'hard' and
                               r['predicted_difficulty'] == 'hard' and
                               r['correct'])

    router_analysis = {
        'accuracy': router_accuracy,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'correct_routing': {
            'easy': correct_routing_easy,
            'hard': correct_routing_hard
        }
    }

    return router_analysis

def main():
    parser = argparse.ArgumentParser(description="Test compound AI system")

    parser.add_argument('--router-model-path', type=str, required=True,
                        help='Path to the fine-tuned router model')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length for router')

    parser.add_argument('--small-model', default='llama3')
    parser.add_argument('--large-model-type', default='claude', choices=['ollama', 'claude'])
    parser.add_argument('--large-model', default='claude-3-haiku-20240307')

    parser.add_argument('--num-samples', default=5, type=int)
    parser.add_argument('--baseline', action='store_true', help='Run baseline test with large LLM only')
    parser.add_argument('--output-file', default='cai_results.json')

    parser.add_argument('--confidence-threshold', type=float, default=0.8,
                        help='Confidence threshold for routing to the small LLM (default: 0.7)')

    args = parser.parse_args()

    try:
        print("Setting up compound AI system...")
        compound_ai = setup_compound_ai_system(args)

        results = test_with_arc_samples(compound_ai, args.num_samples)

        baseline_results = None
        if args.baseline:
            random.seed(42)
            baseline_results = run_baseline_test(args, args.num_samples)

        save_results(results, args.output_file)
        save_baseline_results(baseline_results)
        router_analysis = analyze_router_predictions(results)
        print("\n ====== ROUTER PERFORMANCE ======")
        print(f"Router accuracy: {router_analysis['accuracy']}")
        print(f"False positives: {router_analysis['false_positives']}")
        print(f"False negatives: {router_analysis['false_negatives']}")

        comparison = None
        if baseline_results:
            comparison = compare_results(results, baseline_results)

            print("\n ======== PERFORMANCE COMPARISON ========")
            print(f"Compound AI accuracy: {comparison['accuracy']['compound']:.2%}")
            print(f"Baseline accuracy: {comparison['accuracy']['baseline']:.2%}")
            print(f"Accuracy difference: {comparison['accuracy']['difference']:.2%}")
            print(f"Compound AI avg time: {comparison['average_time_ms']['compound']:.2f}ms")
            print(f"Baseline avg time: {comparison['average_time_ms']['baseline']:.2f}ms")
            print(f"Speedup factor: {comparison['average_time_ms']['speedup']:.2f}x")
            print(f"Small LLM usage: {comparison['resource_utilization']['small_llm_usage']:.2%}")

        output = {
            'compound_results': results,
            'baseline_results': baseline_results,
            'router_analysis': router_analysis,
            'comparison': comparison,
            'test_config': vars(args)
        }

        with open(args.output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {args.output_file}")

        # Save compound results in the format expected by save_results
        if results:
            save_results(results, args.output_file.replace('.json', '_summary.json'))

    except Exception as e:
        print(f"Error testing compound AI system: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())


