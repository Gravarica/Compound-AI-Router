import os
import argparse
import json
from typing import Dict, Any, List
import time
import random
from util import save_results

from dotenv import load_dotenv

from llm_factory import LLMFactory
from compound_ai_system import CompoundAISystem
from dataloader import ARCDataManager

load_dotenv()

def setup_compound_ai_system(args):

    router_config = {
        'model_name': args.router_model,
        'host': os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    }

    router_llm = LLMFactory.create_llm('ollama', router_config)

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
        router = router_llm,
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

        result = compound_ai.process_query(
            query_id = qid,
            query = question,
            choices=choices,
            correct_answer_key=correct_answer
        )

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

def main():
    parser = argparse.ArgumentParser(description="Test compound AI system")

    parser.add_argument('--router-model', default='phi')

    parser.add_argument('--small-model', default='llama3')
    parser.add_argument('--large-model-type', default='claude', choices=['ollama', 'claude'])
    parser.add_argument('--large-model', default='claude-3-haiku-20240307')

    parser.add_argument('--num-samples', default=5, type=int)
    parser.add_argument('--output-file', default='cai_results.json')

    args = parser.parse_args()

    try:
        print("Setting up compound AI system...")
        compound_ai = setup_compound_ai_system(args)

        results = test_with_arc_samples(compound_ai, args.num_samples)

        save_results(results, args.output_file)

    except Exception as e:
        print(f"Error testing compound AI system: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())


