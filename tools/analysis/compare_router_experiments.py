#!/usr/bin/env python3
"""
Router Effectiveness Experimental Comparison
Compares Transformer vs Random vs Oracle routers using GPT-4o-mini + Gemma2 2B
against GPT-4o-mini baseline to determine if routers add value.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_results():
    """Load all experiment results."""
    
    # Load Transformer Router results
    with open('results/experiments/compound/transformer_router/gpt_gemma2/evaluation_results_full.json', 'r') as f:
        transformer_data = json.load(f)
    
    # Load Random Router results
    with open('results/compound_random_gpt_gemma2/compound_random_gpt_gemma2_results_full.json', 'r') as f:
        random_data = json.load(f)
    
    # Load Oracle Router results
    with open('results/compound_oracle_gpt_gemma2/compound_oracle_gpt_gemma2_results_full.json', 'r') as f:
        oracle_data = json.load(f)
    
    # Load GPT-4o-mini baseline
    with open('results/baselines/openai_gpt4o_mini/baseline_openai_gpt4o_mini_results_full.json', 'r') as f:
        baseline_data = json.load(f)
    
    return transformer_data, random_data, oracle_data, baseline_data

def calculate_router_accuracy(data):
    """Calculate router decision accuracy."""
    results = data.get('results', [])
    router_correct = 0
    router_decisions = 0
    
    for result in results:
        true_difficulty = result.get('true_difficulty', '')
        chosen_llm = result.get('chosen_llm', '')
        
        if true_difficulty and chosen_llm:
            router_decisions += 1
            if (true_difficulty == 'easy' and chosen_llm == 'small') or (true_difficulty == 'hard' and chosen_llm == 'large'):
                router_correct += 1
    
    return router_correct / router_decisions if router_decisions > 0 else 0

def create_comparison_table():
    """Create comprehensive comparison table."""
    
    transformer_data, random_data, oracle_data, baseline_data = load_results()
    
    # Extract metrics
    experiments = {
        'GPT-4o-mini Baseline': {
            'system_accuracy': baseline_data['cost_summary']['summary_metrics']['accuracy'],
            'avg_latency_ms': baseline_data['cost_summary']['summary_metrics']['avg_latency_ms'],
            'total_cost': baseline_data['cost_summary']['summary_metrics']['total_cost'],
            'cost_per_query': baseline_data['cost_summary']['summary_metrics']['cost_per_query'],
            'router_accuracy': 'N/A',
            'router_type': 'No Router'
        },
        'Random Router': {
            'system_accuracy': random_data['cost_summary']['summary_metrics']['accuracy'],
            'avg_latency_ms': random_data['cost_summary']['summary_metrics']['avg_latency_ms'],
            'total_cost': random_data['cost_summary']['summary_metrics']['total_cost'],
            'cost_per_query': random_data['cost_summary']['summary_metrics']['cost_per_query'],
            'router_accuracy': calculate_router_accuracy(random_data),
            'router_type': 'Random'
        },
        'Transformer Router (DistilBERT)': {
            'system_accuracy': transformer_data['cost_summary']['summary_metrics']['accuracy'],
            'avg_latency_ms': transformer_data['cost_summary']['summary_metrics']['avg_latency_ms'],
            'total_cost': transformer_data['cost_summary']['summary_metrics']['total_cost'],
            'cost_per_query': transformer_data['cost_summary']['summary_metrics']['cost_per_query'],
            'router_accuracy': calculate_router_accuracy(transformer_data),
            'router_type': 'DistilBERT'
        },
        'Oracle Router (Perfect)': {
            'system_accuracy': oracle_data['cost_summary']['summary_metrics']['accuracy'],
            'avg_latency_ms': oracle_data['cost_summary']['summary_metrics']['avg_latency_ms'],
            'total_cost': oracle_data['cost_summary']['summary_metrics']['total_cost'],
            'cost_per_query': oracle_data['cost_summary']['summary_metrics']['cost_per_query'],
            'router_accuracy': calculate_router_accuracy(oracle_data),
            'router_type': 'Oracle'
        }
    }
    
    return experiments

def create_comparison_visualization():
    """Create comprehensive visualization comparing all approaches."""
    
    experiments = create_comparison_table()
    
    # Prepare data for plotting
    names = list(experiments.keys())
    system_accuracies = [exp['system_accuracy'] * 100 for exp in experiments.values()]
    latencies = [exp['avg_latency_ms'] for exp in experiments.values()]
    costs = [exp['cost_per_query'] * 1000000 for exp in experiments.values()]  # Convert to $/Million
    router_accuracies = [exp['router_accuracy'] * 100 if isinstance(exp['router_accuracy'], float) else 0 for exp in experiments.values()]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Router Effectiveness Analysis: Experimental Results\nGPT-4o-mini + Gemma2 2B vs GPT-4o-mini Baseline', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. System Accuracy Comparison (Top Left)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(names, system_accuracies, color=colors, alpha=0.8)
    ax1.set_title('System Accuracy Comparison', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(80, 95)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, system_accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add baseline reference line
    baseline_acc = system_accuracies[0]
    ax1.axhline(y=baseline_acc, color='blue', linestyle='--', alpha=0.7, label='GPT-4o-mini Baseline')
    ax1.legend()
    
    # 2. Latency Comparison (Top Right)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(names, latencies, color=colors, alpha=0.8)
    ax2.set_title('Average Latency Comparison', fontweight='bold')
    ax2.set_ylabel('Latency (ms)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, latencies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}ms', ha='center', va='bottom', fontweight='bold')
    
    # 3. Cost Comparison (Bottom Left)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(names, costs, color=colors, alpha=0.8)
    ax3.set_title('Cost per Query Comparison', fontweight='bold')
    ax3.set_ylabel('Cost per Query ($/Million)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars3, costs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'${val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Router Accuracy (Bottom Right)
    ax4 = axes[1, 1]
    # Only show router experiments (exclude baseline)
    router_names = names[1:]  # Skip baseline
    router_accs = router_accuracies[1:]  # Skip baseline
    router_colors = colors[1:]
    
    bars4 = ax4.bar(router_names, router_accs, color=router_colors, alpha=0.8)
    ax4.set_title('Router Decision Accuracy', fontweight='bold')
    ax4.set_ylabel('Router Accuracy (%)')
    ax4.set_ylim(0, 105)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars4, router_accs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add random baseline line
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random Baseline (50%)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('router_effectiveness_experimental_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_analysis():
    """Print detailed numerical analysis."""
    
    experiments = create_comparison_table()
    
    print("=" * 80)
    print("🎯 ROUTER EFFECTIVENESS EXPERIMENTAL ANALYSIS")
    print("=" * 80)
    
    print("\n📊 EXPERIMENTAL SETUP:")
    print("• Models: GPT-4o-mini (Large) + Gemma2 2B (Small)")
    print("• Dataset: ARC (500 questions)")
    print("• Routers: Random, DistilBERT Transformer, Oracle (Perfect)")
    print("• Baseline: GPT-4o-mini only")
    
    print("\n📈 DETAILED RESULTS:")
    print("-" * 60)
    
    baseline_exp = experiments['GPT-4o-mini Baseline']
    
    for name, exp in experiments.items():
        print(f"\n{name}:")
        print(f"  • System Accuracy: {exp['system_accuracy']:.1%}")
        if exp['router_accuracy'] != 'N/A':
            print(f"  • Router Accuracy: {exp['router_accuracy']:.1%}")
        print(f"  • Avg Latency: {exp['avg_latency_ms']:.0f}ms")
        print(f"  • Cost per Query: ${exp['cost_per_query'] * 1000000:.1f}/Million")
        
        # Compare to baseline
        if name != 'GPT-4o-mini Baseline':
            acc_diff = (exp['system_accuracy'] - baseline_exp['system_accuracy']) * 100
            latency_diff = exp['avg_latency_ms'] - baseline_exp['avg_latency_ms']
            cost_diff = (exp['cost_per_query'] - baseline_exp['cost_per_query']) * 1000000
            
            print(f"  • vs Baseline: {acc_diff:+.1f}% accuracy, {latency_diff:+.0f}ms latency, ${cost_diff:+.1f}/M cost")
    
    print("\n🔍 KEY INSIGHTS:")
    print("-" * 60)
    
    # Router effectiveness analysis
    random_acc = experiments['Random Router']['system_accuracy']
    transformer_acc = experiments['Transformer Router (DistilBERT)']['system_accuracy']
    oracle_acc = experiments['Oracle Router (Perfect)']['system_accuracy']
    baseline_acc = baseline_exp['system_accuracy']
    
    insights = []
    
    # 1. Router vs Baseline
    if transformer_acc > baseline_acc:
        insights.append(f"✅ DistilBERT router outperforms baseline (+{(transformer_acc - baseline_acc)*100:.1f}%)")
    else:
        insights.append(f"❌ DistilBERT router underperforms baseline ({(transformer_acc - baseline_acc)*100:.1f}%)")
    
    # 2. Router vs Random
    if transformer_acc > random_acc:
        insights.append(f"✅ DistilBERT router beats random routing (+{(transformer_acc - random_acc)*100:.1f}%)")
    else:
        insights.append(f"❌ DistilBERT router barely better than random (+{(transformer_acc - random_acc)*100:.1f}%)")
    
    # 3. Router potential
    oracle_gap = (oracle_acc - transformer_acc) * 100
    insights.append(f"🎯 Gap to perfect routing: {oracle_gap:.1f}% (significant improvement potential)")
    
    # 4. Cost effectiveness
    transformer_cost = experiments['Transformer Router (DistilBERT)']['cost_per_query']
    baseline_cost = baseline_exp['cost_per_query']
    if transformer_cost < baseline_cost:
        cost_savings = ((baseline_cost - transformer_cost) / baseline_cost) * 100
        insights.append(f"💰 Router reduces cost by {cost_savings:.1f}% vs baseline")
    else:
        cost_increase = ((transformer_cost - baseline_cost) / baseline_cost) * 100
        insights.append(f"💸 Router increases cost by {cost_increase:.1f}% vs baseline")
    
    # 5. Latency impact
    transformer_latency = experiments['Transformer Router (DistilBERT)']['avg_latency_ms']
    baseline_latency = baseline_exp['avg_latency_ms']
    latency_improvement = ((baseline_latency - transformer_latency) / baseline_latency) * 100
    if latency_improvement > 0:
        insights.append(f"⚡ Router improves latency by {latency_improvement:.1f}% vs baseline")
    else:
        insights.append(f"🐌 Router increases latency by {-latency_improvement:.1f}% vs baseline")
    
    for i, insight in enumerate(insights, 1):
        print(f"{i:2d}. {insight}")
    
    print("\n🎯 FINAL VERDICT:")
    print("-" * 60)
    
    # Calculate overall router value
    router_value_score = 0
    
    # Accuracy vs baseline (40% weight)
    if transformer_acc > baseline_acc:
        router_value_score += 40
    elif transformer_acc > baseline_acc * 0.98:  # Within 2%
        router_value_score += 20
    
    # Cost effectiveness (30% weight) 
    if transformer_cost < baseline_cost:
        router_value_score += 30
    elif transformer_cost < baseline_cost * 1.1:  # Within 10%
        router_value_score += 15
    
    # Latency improvement (20% weight)
    if transformer_latency < baseline_latency:
        router_value_score += 20
    elif transformer_latency < baseline_latency * 1.1:  # Within 10%
        router_value_score += 10
    
    # Router accuracy (10% weight)
    transformer_router_acc = experiments['Transformer Router (DistilBERT)']['router_accuracy']
    if transformer_router_acc > 0.7:
        router_value_score += 10
    elif transformer_router_acc > 0.6:
        router_value_score += 5
    
    print(f"Router Value Score: {router_value_score}/100")
    
    if router_value_score >= 70:
        verdict = "✅ STRONG VALUE - Router significantly improves the system"
    elif router_value_score >= 50:
        verdict = "⚠️  MODERATE VALUE - Router provides some benefits but needs improvement"
    elif router_value_score >= 30:
        verdict = "🟡 LIMITED VALUE - Router shows promise but current implementation questionable"
    else:
        verdict = "❌ NO VALUE - Router does not improve over baseline"
    
    print(f"\n{verdict}")
    
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 60)
    
    recommendations = []
    
    if transformer_acc <= baseline_acc:
        recommendations.append("🔧 IMMEDIATE: Router underperforms baseline - try DeBERTa or RoBERTa architectures")
    
    if oracle_gap > 20:
        recommendations.append("📈 HIGH PRIORITY: Large gap to oracle suggests router can be substantially improved")
    
    if transformer_router_acc < 0.7:
        recommendations.append("🎯 FOCUS: Router accuracy is below 70% - improve training data or model architecture")
    
    recommendations.extend([
        "📊 VALIDATE: Test on additional datasets (MMLU, GSM8K) to confirm routing effectiveness",
        "⚡ OPTIMIZE: Consider confidence-based routing thresholds for better performance",
        "💰 ANALYZE: Current cost reduction suggests routing concept is economically viable"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print("🚀 Running Router Effectiveness Experimental Analysis...")
    create_comparison_visualization()
    print_detailed_analysis()
    print("✅ Analysis complete! Check 'router_effectiveness_experimental_results.png'")