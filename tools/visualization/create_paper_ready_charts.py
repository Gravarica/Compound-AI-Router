#!/usr/bin/env python3
"""
Paper-Ready Compound AI vs Monolithic Models Comparison Charts

Creates publication-quality charts with:
- Larger, readable fonts
- PDF and PNG outputs  
- Academic paper styling
- Better contrast and spacing
- Proper font sizes for 2-column layouts
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Configure matplotlib for paper-quality plots
plt.style.use('default')
plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 16,     # Title font size
    'axes.labelsize': 14,     # Axis label font size
    'xtick.labelsize': 12,    # X-tick label font size
    'ytick.labelsize': 12,    # Y-tick label font size
    'legend.fontsize': 12,    # Legend font size
    'figure.titlesize': 18,   # Figure title font size
    'font.family': 'serif',   # Use serif fonts for academic look
    'text.usetex': False,     # Set to True if you have LaTeX installed
    'figure.dpi': 300,        # High DPI for crisp plots
    'savefig.dpi': 300,       # High DPI for saved figures
    'axes.linewidth': 1.2,    # Thicker axis lines
    'grid.linewidth': 0.8,    # Grid line width
    'lines.linewidth': 2.0,   # Line width
})

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
                    'cost': data['cost_summary']['total_cost'],
                    'model_size': 'Large' if name in ['GPT-4o-mini', 'Claude Haiku'] else 'Small'
                }
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
    
    # Load compound AI results
    compound_files = {
        'Claude+Llama3B': 'results/experiments/compound/transformer_router/claude_3b/evaluation_results_full.json',
        'GPT+Llama3B': 'results/experiments/compound/transformer_router/gpt_3b/evaluation_results_full.json',
        'GPT+Qwen1.5B': 'results/experiments/compound/transformer_router/gpt_qwen/evaluation_results_full.json',
        'Claude+Qwen1.5B': 'results/experiments/compound/transformer_router/claude_qwen/evaluation_results_full.json'
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
                    'cost': data['cost_summary']['total_cost'],
                    'router_type': 'transformer'
                }
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
    
    return results

def create_accuracy_comparison(results, output_prefix="accuracy_comparison"):
    """Create paper-ready accuracy comparison chart."""
    
    # Separate and sort data
    all_items = [(name, results[name]['accuracy']) for name in results]
    sorted_items = sorted(all_items, key=lambda x: x[1], reverse=True)
    names = [name for name, _ in sorted_items]
    accuracies = [acc for _, acc in sorted_items]

    # Define colors with better contrast
    colors = []
    for name in names:
        if results[name]['type'] == 'monolithic':
            if results[name]['model_size'] == 'Large':
                colors.append('#1f77b4')  # Blue for large models
            else:
                colors.append('#d62728')  # Red for small models
        else:
            colors.append('#ff7f0e')  # Orange for compound systems
    
    # Create figure with appropriate size for papers
    fig, ax = plt.subplots(figsize=(10, 6))  # Good for 2-column layout
    
    # Create bars with better spacing
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, accuracies, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=0.8)
    
    # Customize plot
    ax.set_title('Task Accuracy: Compound AI vs Monolithic Models', 
                fontweight='bold', pad=20)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('System Configuration')
    
    # Set x-axis labels with better rotation
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=35, ha='right')
    
    # Set y-axis limits with some padding
    ax.set_ylim(0, max(accuracies) * 1.15)
    
    # Add horizontal grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add value labels on bars with better positioning
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + max(accuracies)*0.01,
                f'{acc:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    # Create legend with proper academic styling
    legend_elements = [
        mpatches.Patch(color='#1f77b4', alpha=0.8, label='Large Models'),
        mpatches.Patch(color='#d62728', alpha=0.8, label='Small Models'),
        mpatches.Patch(color='#ff7f0e', alpha=0.8, label='Compound AI')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
             fancybox=True, shadow=True)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"✅ Accuracy chart saved as {output_prefix}.png and {output_prefix}.pdf")

def create_latency_comparison(results, output_prefix="latency_comparison"):
    """Create paper-ready latency comparison chart."""
    
    # Prepare and sort data
    all_items = [(name, results[name]['latency']) for name in results]
    sorted_items = sorted(all_items, key=lambda x: x[1])  # Ascending order
    names = [name for name, _ in sorted_items]
    latencies = [lat for _, lat in sorted_items]

    # Colors (same scheme as accuracy)
    colors = []
    for name in names:
        if results[name]['type'] == 'monolithic':
            if results[name]['model_size'] == 'Large':
                colors.append('#1f77b4')  # Blue
            else:
                colors.append('#d62728')  # Red
        else:
            colors.append('#ff7f0e')  # Orange
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, latencies, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.8)
    
    # Customize plot
    ax.set_title('Response Latency: Compound AI vs Monolithic Models', 
                fontweight='bold', pad=20)
    ax.set_ylabel('Average Latency (ms)')
    ax.set_xlabel('System Configuration')
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=35, ha='right')
    
    # Set y-axis with padding
    ax.set_ylim(0, max(latencies) * 1.15)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Value labels
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + max(latencies)*0.01,
                f'{latency:.0f}ms', ha='center', va='bottom',
                fontweight='bold', fontsize=11)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#1f77b4', alpha=0.8, label='Large Models'),
        mpatches.Patch(color='#d62728', alpha=0.8, label='Small Models'),
        mpatches.Patch(color='#ff7f0e', alpha=0.8, label='Compound AI')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True,
             fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"✅ Latency chart saved as {output_prefix}.png and {output_prefix}.pdf")

def create_cost_comparison(results, output_prefix="cost_comparison"):
    """Create paper-ready cost comparison chart."""
    
    # Prepare and sort data
    all_items = [(name, results[name]['cost']) for name in results]
    sorted_items = sorted(all_items, key=lambda x: x[1])  # Ascending order
    names = [name for name, _ in sorted_items]
    costs = [cost for _, cost in sorted_items]

    # Colors
    colors = []
    for name in names:
        if results[name]['type'] == 'monolithic':
            if results[name]['model_size'] == 'Large':
                colors.append('#1f77b4')  # Blue
            else:
                colors.append('#d62728')  # Red
        else:
            colors.append('#ff7f0e')  # Orange

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, costs, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.8)

    # Customize plot
    ax.set_title('Computational Cost: Compound AI vs Monolithic Models',
                fontweight='bold', pad=20)
    ax.set_ylabel('Cost per 1M Queries ($)')
    ax.set_xlabel('System Configuration')
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=35, ha='right')

    # Set y-axis with padding
    ax.set_ylim(0, max(costs) * 1.2)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Value labels with better formatting
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        # Format cost nicely
        if cost >= 1:
            cost_str = f'${cost:.2f}'
        elif cost >= 0.1:
            cost_str = f'${cost:.3f}'
        else:
            cost_str = f'${cost:.4f}'
            
        ax.text(bar.get_x() + bar.get_width()/2, height + max(costs)*0.01,
                cost_str, ha='center', va='bottom',
                fontweight='bold', fontsize=11)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#1f77b4', alpha=0.8, label='Large Models'),
        mpatches.Patch(color='#d62728', alpha=0.8, label='Small Models'),
        mpatches.Patch(color='#ff7f0e', alpha=0.8, label='Compound AI')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True,
             fancybox=True, shadow=True)

    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"✅ Cost chart saved as {output_prefix}.png and {output_prefix}.pdf")

def create_combined_summary_chart(results, output_prefix="combined_summary"):
    """Create a combined chart showing all three metrics normalized."""
    
    # Prepare data
    models = list(results.keys())
    n_models = len(models)
    
    # Extract metrics
    accuracies = [results[model]['accuracy'] for model in models]
    latencies = [results[model]['latency'] for model in models]
    costs = [results[model]['cost'] for model in models]
    
    # Normalize metrics (0-1 scale)
    # For accuracy: higher is better, so normalize directly
    norm_acc = np.array(accuracies) / max(accuracies)
    
    # For latency and cost: lower is better, so invert
    norm_lat = 1 - (np.array(latencies) - min(latencies)) / (max(latencies) - min(latencies))
    norm_cost = 1 - (np.array(costs) - min(costs)) / (max(costs) - min(costs))
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set positions
    x = np.arange(n_models)
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, norm_acc, width, label='Accuracy (normalized)', 
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x, norm_lat, width, label='Latency (inverted & normalized)', 
                   color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + width, norm_cost, width, label='Cost (inverted & normalized)', 
                   color='#2ca02c', alpha=0.8)
    
    # Customize
    ax.set_title('Normalized Performance Comparison\n(Higher is Better for All Metrics)', 
                fontweight='bold', pad=20)
    ax.set_ylabel('Normalized Score (0-1)')
    ax.set_xlabel('System Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    
    # Grid
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"✅ Combined chart saved as {output_prefix}.png and {output_prefix}.pdf")

def main():
    """Create all paper-ready charts."""
    
    print("🚀 Creating Paper-Ready Comparison Charts...")
    print("📝 Using academic styling with larger fonts and PDF output")
    
    # Load results
    results = load_experiment_results()
    
    if not results:
        print("❌ No results found! Make sure experiment files exist.")
        return
    
    print(f"📊 Loaded {len(results)} configurations")
    
    # Create all charts
    print("\n📈 Creating accuracy comparison...")
    create_accuracy_comparison(results, "paper_accuracy_comparison")
    
    print("\n⏱️ Creating latency comparison...")
    create_latency_comparison(results, "paper_latency_comparison")
    
    print("\n💰 Creating cost comparison...")
    create_cost_comparison(results, "paper_cost_comparison")
    
    print("\n📊 Creating combined summary...")
    create_combined_summary_chart(results, "paper_combined_summary")
    
    print("\n✅ All paper-ready charts created!")
    print("\nGenerated files (PNG + PDF):")
    print("  📈 paper_accuracy_comparison.png/.pdf")
    print("  ⏱️  paper_latency_comparison.png/.pdf") 
    print("  💰 paper_cost_comparison.png/.pdf")
    print("  📊 paper_combined_summary.png/.pdf")
    print("\n💡 Use PDF versions for best quality in LaTeX documents!")

if __name__ == "__main__":
    main()