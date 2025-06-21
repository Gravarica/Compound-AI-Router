#!/usr/bin/env python3
"""
Analyze router performance and create insights for improvement.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_router_effectiveness(results_file):
    """Analyze how much router decisions actually matter."""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Calculate different scenarios
    scenarios = {
        'correct_route_correct_answer': 0,  # Router right, answer right
        'correct_route_wrong_answer': 0,    # Router right, answer wrong  
        'wrong_route_correct_answer': 0,    # Router wrong, answer right
        'wrong_route_wrong_answer': 0,      # Router wrong, answer wrong
    }
    
    routing_impact = {
        'router_saved_cost': 0,       # Easy→Small, got it right
        'router_wasted_cost': 0,      # Easy→Large, but Small would have worked
        'router_saved_accuracy': 0,   # Hard→Large, Small would have failed
        'router_hurt_accuracy': 0,    # Hard→Small, failed when Large would work
    }
    
    for result in results:
        router_correct = (result['predicted_difficulty'] == result['true_difficulty'])
        answer_correct = result['correct']
        predicted_diff = result['predicted_difficulty']
        true_diff = result['true_difficulty']
        chosen_llm = result['chosen_llm']
        
        # Categorize outcomes
        if router_correct and answer_correct:
            scenarios['correct_route_correct_answer'] += 1
        elif router_correct and not answer_correct:
            scenarios['correct_route_wrong_answer'] += 1
        elif not router_correct and answer_correct:
            scenarios['wrong_route_correct_answer'] += 1
        else:
            scenarios['wrong_route_wrong_answer'] += 1
            
        # Analyze routing impact
        if true_diff == 'easy' and chosen_llm == 'small' and answer_correct:
            routing_impact['router_saved_cost'] += 1
        elif true_diff == 'easy' and chosen_llm == 'large':
            routing_impact['router_wasted_cost'] += 1
        elif true_diff == 'hard' and chosen_llm == 'large' and answer_correct:
            routing_impact['router_saved_accuracy'] += 1
        elif true_diff == 'hard' and chosen_llm == 'small' and not answer_correct:
            routing_impact['router_hurt_accuracy'] += 1
    
    return scenarios, routing_impact, results

def calculate_router_value(results_file):
    """Calculate actual value provided by the router."""
    
    scenarios, routing_impact, results = analyze_router_effectiveness(results_file)
    total = len(results)
    
    print("🔍 ROUTER EFFECTIVENESS ANALYSIS")
    print("=" * 50)
    
    print(f"\n📊 Scenario Breakdown (out of {total} queries):")
    for scenario, count in scenarios.items():
        print(f"  {scenario}: {count} ({count/total:.1%})")
    
    print(f"\n💰 Routing Impact Analysis:")
    for impact, count in routing_impact.items():
        print(f"  {impact}: {count} ({count/total:.1%})")
    
    # Calculate potential improvements
    print(f"\n🎯 IMPROVEMENT OPPORTUNITIES:")
    
    # How much could perfect routing help?
    wrong_route_scenarios = scenarios['wrong_route_correct_answer'] + scenarios['wrong_route_wrong_answer']
    print(f"  Queries hurt by wrong routing: {wrong_route_scenarios} ({wrong_route_scenarios/total:.1%})")
    
    # Cost savings potential
    cost_waste = routing_impact['router_wasted_cost']
    print(f"  Unnecessary expensive calls: {cost_waste} ({cost_waste/total:.1%})")
    
    # Accuracy loss potential  
    accuracy_loss = routing_impact['router_hurt_accuracy']
    print(f"  Accuracy lost to wrong routing: {accuracy_loss} ({accuracy_loss/total:.1%})")
    
    return scenarios, routing_impact

def create_router_comparison_framework():
    """Create framework for comparing different router approaches."""
    
    framework = {
        'datasets_to_test': [
            'ARC-Challenge only (harder questions)',
            'MMLU subset (specialized knowledge)',
            'GSM8K math problems',
            'Mixed difficulty with bigger LLM gaps'
        ],
        'router_improvements': [
            'Better prompts (more specific examples)',
            'Chain-of-thought reasoning',
            'Few-shot examples in prompt',
            'Fine-tuned router on actual LLM performance data'
        ],
        'evaluation_metrics': [
            'Router accuracy (current: 48%)',
            'Cost efficiency (tokens saved vs accuracy lost)',
            'Latency improvement',
            'Actual business value (accuracy×speed×cost)'
        ]
    }
    
    print("\n🚀 ROUTER IMPROVEMENT FRAMEWORK")
    print("=" * 50)
    
    for category, items in framework.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for item in items:
            print(f"  • {item}")
    
    return framework

def suggest_next_experiments():
    """Suggest specific next experiments to run."""
    
    experiments = [
        {
            'name': 'Improved Prompt Router',
            'description': 'Test updated prompt with better examples',
            'command': 'python scripts/run_evaluation.py --config-name=phi2_router_claude',
            'expected_improvement': 'Router accuracy: 48% → 65%+'
        },
        {
            'name': 'Chain-of-Thought Router', 
            'description': 'Test CoT reasoning for difficulty classification',
            'command': 'python scripts/run_evaluation.py --config-name=cot_router_test',
            'expected_improvement': 'Router accuracy: 48% → 70%+'
        },
        {
            'name': 'Stress Test (1B vs Claude)',
            'description': 'Bigger LLM gap to see router impact',
            'command': 'python scripts/run_evaluation.py --config-name=stress_test',
            'expected_improvement': 'Show router value more clearly'
        },
        {
            'name': 'ARC-Challenge Only',
            'description': 'Test on harder subset of questions',
            'command': 'Create config with ARC-Challenge filter',
            'expected_improvement': 'Better differentiation of difficulty'
        }
    ]
    
    print("\n🧪 RECOMMENDED NEXT EXPERIMENTS")
    print("=" * 50)
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{i}. {exp['name']}")
        print(f"   📝 {exp['description']}")
        print(f"   💻 {exp['command']}")
        print(f"   🎯 {exp['expected_improvement']}")
    
    return experiments

if __name__ == "__main__":
    # Analyze current results
    results_file = "results/experiments/router_types/llm/phi2_router_llama3.2_claude_results_full.json"
    
    if Path(results_file).exists():
        scenarios, routing_impact = calculate_router_value(results_file)
        framework = create_router_comparison_framework()
        experiments = suggest_next_experiments()
    else:
        print(f"Results file not found: {results_file}")
        print("Run an experiment first!")