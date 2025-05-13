import os
import json
import sys
import argparse
from typing import Dict, Any

from dotenv import load_dotenv

from llm_interface import LLMInterface
from llm_factory import LLMFactory

load_dotenv()

def test_llm(llm: LLMInterface, test_prompt: str) -> Dict[str, Any]:

    print(f"Testing {llm.get_model_name()}...")
    print(f"Prompt: {test_prompt}")

    try:
        response = llm.generate(test_prompt)

        print(f"Response: {response}")

        resource_usage = llm.get_resource_usage()
        print(f"Resource usage: {resource_usage}")

        return {
            'model': llm.get_model_name(),
            'prompt': test_prompt,
            'response': response,
            'resource_usage': resource_usage,
            'success': True
        }

    except Exception as e:
        print(f"Error generating response: {e}")
        return {
            'model': llm.get_model_name(),
            'prompt': test_prompt,
            'error': str(e),
            'success': False
        }

def main():
    parser = argparse.ArgumentParser(description="Test LLM interfaces")
    parser.add_argument('--model', choices=['ollama', 'claude', 'all'], default = 'all', help="Model to test")
    parser.add_argument('--prompt', default = "What is the capital of France?", help="Prompt to test")
    parser.add_argument('--ollama-model', default='llama3', help='Ollama model name')

    args = parser.parse_args()

    results = []

    if args.model in ['ollama', 'all']:
        try:
            ollama_config = {
                'model_name': args.ollama_model,
                'host': os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            }
            ollama_llm = LLMFactory.create_llm('ollama', ollama_config)
            ollama_result = test_llm(ollama_llm, args.prompt)
            results.append(ollama_result)
        except Exception as e:
            print(f"Error initializing Ollama LLM: {e}")

    if args.model in ['claude', 'all']:
        try:
            claude_api_key = os.getenv('ANTHROPIC_API_KEY')
            if not claude_api_key:
                print("Warning: Anthropic API key not provided. Skipping Claude LLM tests.")
            else:
                claude_config = {
                    "api_key": claude_api_key,
                    "model_name": 'claude-3-haiku-20240307'
                }
                claude_llm = LLMFactory.create_llm('claude', claude_config)
                claude_result = test_llm(claude_llm, args.prompt)
                results.append(claude_result)
        except:
            print(f"Error initializing Claude LLM: {e}")

    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\nTest results saved to test_results.json')

if __name__ == "__main__":
    main()

