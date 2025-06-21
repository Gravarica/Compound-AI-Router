#!/usr/bin/env python3
"""
Compound AI vs Monolithic Models Comparison Charts
Creates separate bar charts for accuracy, latency, and cost comparisons.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_experiment_results():
    """Load all experimental results for comparison."""
    
    results = {}
    
    # Load baseline models (monolithic)
    baseline_files = {
        'GPT-4o-mini': 'results/baselines/openai_gpt4o_mini/baseline_openai_gpt4o_mini_results_full.json',
        'Claude Haiku': 'results/baselines/claude_haiku/baseline_claude_haiku_results_full.json',
        'Llama3.2 3B': 'results/baselines/llama3_2_3b/baseline_llama3_2_3b_results_full.json',
        'Qwen2.5 1.5B': 'results/baselines/qwen2_5_1_5b/baseline_qwen2_5_1_5b_results_full.json'
    }
    
    for name, file_path in baseline_files.items():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                metrics = data['cost_summary']['summary_metrics']
                results[name] = {
                    'type': 'monolithic',
                    'accuracy': metrics['accuracy'] * 100,
                    'latency': metrics['avg_latency_ms'],
                    'cost': data['cost_summary']['total_cost'],  # Convert to $/Million
                    'model_size': 'Large' if name in ['GPT-4o-mini', 'Claude Haiku'] else 'Small'
                }
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
    
    # Load compound AI results (transformer router only)
    compound_files = {
        'Claude Haiku + Llama3.2 3B': 'results/experiments/compound/transformer_router/claude_3b/evaluation_results_full.json',
        'GPT-4o-mini + Llama3.2 3B': 'results/experiments/compound/transformer_router/gpt_3b/evaluation_results_full.json',
        'GPT-4o-mini + Qwen2.5 1.5B': 'results/experiments/compound/transformer_router/gpt_qwen/evaluation_results_full.json',
        'Claude Haiku + Qwen2.5 1.5B': 'results/experiments/compound/transformer_router/claude_qwen/evaluation_results_full.json'
    }
    
    for name, file_path in compound_files.items():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                metrics = data['cost_summary']['summary_metrics']
                results[name] = {
                    'type': 'compound',
                    'accuracy': metrics['accuracy'] * 100,
                    'latency': metrics['avg_latency_ms'],
                    'cost': data['cost_summary']['total_cost'],  # Convert to $/Million
                    'router_type': 'transformer'
                }
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
    
    return results

def create_accuracy_comparison(results):
    """Create accuracy comparison bar chart."""
    
    # Separate baselines and compound systems
    baselines = {k: v for k, v in results.items() if v['type'] == 'monolithic'}
    compounds = {k: v for k, v in results.items() if v['type'] == 'compound'}
    
    # Prepare data
    all_items = [(name, results[name]['accuracy']) for name in results]
    sorted_items = sorted(all_items, key=lambda x: x[1], reverse=True)
    all_names = [name for name, _ in sorted_items]
    all_accuracies = [acc for _, acc in sorted_items]

    # Create colors
    colors = []
    for name in all_names:
        if results[name]['type'] == 'monolithic':
            if results[name]['model_size'] == 'Large':
                colors.append('#2E86AB')  # Blue for large models
            else:
                colors.append('#A23B72')  # Purple for small models
        else:
            colors.append('#F18F01')  # Orange for compound systems
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(range(len(all_names)), all_accuracies, color=colors, alpha=0.8)
    
    # Customize plot
    ax.set_title('Accuracy Comparison: Compound AI vs Monolithic Models', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, all_accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#2E86AB', alpha=0.8, label='Large Models'),
        plt.Rectangle((0,0),1,1, facecolor='#A23B72', alpha=0.8, label='Small Models'),
        plt.Rectangle((0,0),1,1, facecolor='#F18F01', alpha=0.8, label='Compound AI')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_latency_comparison(results):
    """Create latency comparison bar chart."""
    
    # Prepare data
    all_items = [(name, results[name]['latency']) for name in results]
    sorted_items = sorted(all_items, key=lambda x: x[1])  # Ascending
    all_names = [name for name, _ in sorted_items]
    all_latencies = [lat for _, lat in sorted_items]

    # Create colors (same scheme as accuracy)
    colors = []
    for name in all_names:
        if results[name]['type'] == 'monolithic':
            if results[name]['model_size'] == 'Large':
                colors.append('#2E86AB')  # Blue for large models
            else:
                colors.append('#A23B72')  # Purple for small models
        else:
            colors.append('#F18F01')  # Orange for compound systems
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(range(len(all_names)), all_latencies, color=colors, alpha=0.8)
    
    # Customize plot
    ax.set_title('Latency Comparison: Compound AI vs Monolithic Models', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Average Latency (ms)', fontsize=12)
    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, latency in zip(bars, all_latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{latency:.0f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#2E86AB', alpha=0.8, label='Large Models'),
        plt.Rectangle((0,0),1,1, facecolor='#A23B72', alpha=0.8, label='Small Models'),
        plt.Rectangle((0,0),1,1, facecolor='#F18F01', alpha=0.8, label='Compound AI')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt

def create_cost_comparison(results):
    """Create cost comparison bar chart."""

    # Prepare data
    all_items = [(name, results[name]['cost']) for name in results]
    sorted_items = sorted(all_items, key=lambda x: x[1])  # Ascending
    all_names = [name for name, _ in sorted_items]
    all_costs = [cost for _, cost in sorted_items]

    # Create colors (same scheme as accuracy)
    colors = []
    for name in all_names:
        if results[name]['type'] == 'monolithic':
            if results[name]['model_size'] == 'Large':
                colors.append('#2E86AB')  # Blue for large models
            else:
                colors.append('#A23B72')  # Purple for small models
        else:
            colors.append('#F18F01')  # Orange for compound systems

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(range(len(all_names)), all_costs, color=colors, alpha=0.8)

    # Customize plot
    ax.set_title('Cost Comparison: Compound AI vs Monolithic Models',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=60, ha='right')

    # Extend y-axis to avoid clipping bar labels
    ax.set_ylim(0, max(all_costs) * 1.2)

    # Add gridlines
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, cost in zip(bars, all_costs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(all_costs) * 0.02),  # dynamic offset
            f'${cost:.4f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#2E86AB', alpha=0.8, label='Large Models'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#A23B72', alpha=0.8, label='Small Models'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#F18F01', alpha=0.8, label='Compound AI')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    # Adjust layout to prevent label cut-off
    plt.subplots_adjust(bottom=0.25)  # Add space for rotated x-labels
    plt.tight_layout()
    plt.savefig('cost_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_table(results):
    """Print a summary table of all results."""
    
    print("\n" + "="*80)
    print("📊 COMPOUND AI vs MONOLITHIC MODELS COMPARISON")
    print("="*80)
    
    print(f"\n{'Model Configuration':<35} {'Type':<10} {'Accuracy':<10} {'Latency':<12} {'Cost':<12}")
    print("-" * 80)
    
    # Sort by accuracy (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for name, metrics in sorted_results:
        model_type = "Baseline" if metrics['type'] == 'monolithic' else "Compound"
        print(f"{name:<35} {model_type:<10} {metrics['accuracy']:<9.1f}% {metrics['latency']:<11.0f}ms ${metrics['cost']:<10.1f}/M")
    
    print("\n💡 KEY INSIGHTS:")
    print("-" * 40)
    
    # Find best performers
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_latency = min(results.items(), key=lambda x: x[1]['latency'])
    best_cost = min(results.items(), key=lambda x: x[1]['cost'])
    
    print(f"🎯 Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.1f}%)")
    print(f"⚡ Best Latency: {best_latency[0]} ({best_latency[1]['latency']:.0f}ms)")
    print(f"💰 Best Cost: {best_cost[0]} (${best_cost[1]['cost']:.1f}/M)")
    
    # Compound AI analysis
    compound_results = {k: v for k, v in results.items() if v['type'] == 'compound'}
    if compound_results:
        avg_compound_acc = np.mean([v['accuracy'] for v in compound_results.values()])
        avg_compound_latency = np.mean([v['latency'] for v in compound_results.values()])
        avg_compound_cost = np.mean([v['cost'] for v in compound_results.values()])
        
        baseline_results = {k: v for k, v in results.items() if v['type'] == 'monolithic'}
        large_baselines = {k: v for k, v in baseline_results.items() if v['model_size'] == 'Large'}
        
        if large_baselines:
            avg_large_acc = np.mean([v['accuracy'] for v in large_baselines.values()])
            avg_large_latency = np.mean([v['latency'] for v in large_baselines.values()])
            avg_large_cost = np.mean([v['cost'] for v in large_baselines.values()])
            
            print(f"\n📈 Compound AI vs Large Models Average:")
            print(f"   Accuracy: {avg_compound_acc:.1f}% vs {avg_large_acc:.1f}% ({avg_compound_acc - avg_large_acc:+.1f}%)")
            print(f"   Latency: {avg_compound_latency:.0f}ms vs {avg_large_latency:.0f}ms ({avg_compound_latency - avg_large_latency:+.0f}ms)")
            print(f"   Cost: ${avg_compound_cost:.1f}/M vs ${avg_large_cost:.1f}/M (${avg_compound_cost - avg_large_cost:+.1f}/M)")

def main():
    """Main function to create all comparison charts."""
    
    print("🚀 Creating Compound AI vs Monolithic Models Comparison Charts...")
    
    # Load results
    results = load_experiment_results()
    
    if not results:
        print("❌ No results found! Make sure experiment files exist.")
        return
    
    print(f"📊 Loaded {len(results)} configurations")
    
    # Create charts
    print("\n📈 Creating accuracy comparison chart...")
    create_accuracy_comparison(results)
    
    print("⏱️ Creating latency comparison chart...")
    create_latency_comparison(results)
    
    print("💰 Creating cost comparison chart...")
    create_cost_comparison(results)
    
    # Print summary
    print_summary_table(results)
    
    print("\n✅ All comparison charts created!")
    print("Generated files:")
    print("  - accuracy_comparison.png")
    print("  - latency_comparison.png") 
    print("  - cost_comparison.png")

if __name__ == "__main__":
    main()