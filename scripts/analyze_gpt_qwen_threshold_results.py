#!/usr/bin/env python3
"""
Analyze GPT + Qwen Threshold Sensitivity Results
==============================================
Creates threshold sensitivity plots for GPT-4o-mini + Qwen2.5-1.5B system
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_threshold_results():
    """Load results from all threshold experiments"""
    
    base_dir = "results/experiments/threshold_sensitivity_gpt_qwen"
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    
    print("📊 Loading GPT + Qwen threshold experiment results...")
    print(f"📊 System: GPT-4o-mini + Qwen2.5-1.5B")
    print("=" * 50)
    
    for threshold in thresholds:
        result_file = f"{base_dir}/threshold_{threshold:.2f}/evaluation_results_full.json"
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            # Extract key metrics from evaluation_metadata
            eval_metadata = data['evaluation_metadata']
            accuracy = eval_metadata['accuracy']
            total_cost = eval_metadata['total_cost']
            total_queries = eval_metadata['total_queries']
            cost_per_1k = (total_cost / total_queries) * 1000  # Convert to cost per 1K queries
            
            # Calculate routing statistics from small_llm_usage_ratio
            small_llm_ratio = eval_metadata['small_llm_usage_ratio']
            small_llm_count = int(total_queries * small_llm_ratio)
            large_llm_count = total_queries - small_llm_count
            
            if total_queries > 0:
                pct_to_large = (large_llm_count / total_queries) * 100
                pct_to_small = (small_llm_count / total_queries) * 100
            else:
                pct_to_large = 0
                pct_to_small = 100
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'cost_per_1k': cost_per_1k,
                'pct_to_large': pct_to_large,
                'pct_to_small': pct_to_small,
                'small_count': small_llm_count,
                'large_count': large_llm_count
            })
            
            print(f"✅ Loaded threshold {threshold:.2f}: Acc={accuracy:.3f}, Cost=${cost_per_1k:.4f}, Large={pct_to_large:.1f}%")
        else:
            print(f"❌ Missing: {result_file}")
    
    print(f"\n✅ Successfully loaded {len(results)} threshold experiments")
    return results

def create_threshold_sensitivity_plot(results):
    """Create the threshold sensitivity analysis plot"""
    
    # Extract data
    thresholds = [r['threshold'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    costs = [r['cost_per_1k'] for r in results]
    pct_large = [r['pct_to_large'] for r in results]
    
    print(f"📊 Creating plot with {len(results)} data points")
    print(f"Threshold range: {min(thresholds):.2f} - {max(thresholds):.2f}")
    print(f"Accuracy range: {min(accuracies):.1%} - {max(accuracies):.1%}")
    print(f"Cost range: ${min(costs):.2f} - ${max(costs):.2f} per 1K queries")
    
    # Create the plot with academic styling
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Configure for academic papers
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 3,
        'axes.linewidth': 1.5,
        'grid.linewidth': 1,
        'grid.alpha': 0.3
    })
    
    # Colors
    color1 = '#1f77b4'  # Blue for accuracy
    color2 = '#d62728'  # Red for cost  
    color3 = '#2ca02c'  # Green for large model usage
    
    # Plot 1: System Accuracy
    ax1.set_xlabel('Router Confidence Threshold', fontweight='bold')
    ax1.set_ylabel('System Accuracy', color=color1, fontweight='bold')
    line1 = ax1.plot(thresholds, accuracies, 'o-', color=color1, linewidth=3, 
                    markersize=8, label='System Accuracy', markerfacecolor='white',
                    markeredgewidth=2, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(min(accuracies) - 0.05, max(accuracies) + 0.05)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cost per 1K queries
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cost per 1K Queries ($)', color=color2, fontweight='bold')
    line2 = ax2.plot(thresholds, costs, 's-', color=color2, linewidth=3,
                    markersize=8, label='Total Cost', markerfacecolor='white',
                    markeredgewidth=2, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(costs) * 1.1)
    
    accuracy_cost_ratios = []
    for acc, cost in zip(accuracies, costs):
        if cost > 0:
            accuracy_cost_ratios.append(acc/(cost/1000))  # Accuracy per dollar
        else:
            accuracy_cost_ratios.append(float('inf'))  # Infinite ratio for zero cost
    
    # Title and legend
    plt.title('Router Threshold Sensitivity Analysis',
              fontsize=16, fontweight='bold', pad=20)
    
    # Create combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.02, 0.98),
              frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    # Tight layout and save
    plt.tight_layout()
    
    # Save both PNG and PDF
    plt.savefig('gpt_qwen_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('threshold_sensitivity.pdf', bbox_inches='tight')
    
    print("✅ GPT + Qwen threshold sensitivity plot saved as gpt_qwen_threshold_analysis.png and gpt_qwen_threshold_analysis.pdf")

    return 1, 1, 1

def save_data_for_plotting(results, optimal_threshold, optimal_accuracy, optimal_cost):
    """Save the data in JSON format for further analysis"""
    
    data = {
        "thresholds": [r['threshold'] for r in results],
        "system_accuracy": [r['accuracy'] for r in results],
        "cost_per_1k": [r['cost_per_1k'] for r in results],
        "pct_to_large_model": [r['pct_to_large'] for r in results],
        "optimal_threshold": optimal_threshold,
        "optimal_accuracy": optimal_accuracy,
        "optimal_cost": optimal_cost
    }
    
    with open('gpt_qwen_threshold_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("✅ Data saved as gpt_qwen_threshold_data.json")

def main():
    """Main analysis function"""
    
    print("📊 Analyzing GPT + Qwen Threshold Sensitivity Results")
    print("📊 System: GPT-4o-mini + Qwen2.5-1.5B")
    print("=" * 50)
    
    # Load results
    results = load_threshold_results()
    
    if not results:
        print("❌ No results found!")
        return
    
    # Create plot
    optimal_threshold, optimal_accuracy, optimal_cost = create_threshold_sensitivity_plot(results)
    
    # Generate report
    #report = generate_report(results, optimal_threshold, optimal_accuracy, optimal_cost)
    #print(report)
    
    # Save data
    save_data_for_plotting(results, optimal_threshold, optimal_accuracy, optimal_cost)
    
    print("\n✅ GPT + Qwen threshold sensitivity analysis complete!")
    print("📊 Plot: gpt_qwen_threshold_analysis.png/.pdf")
    print("📝 Report: gpt_qwen_threshold_report.txt")
    print("📊 Data: gpt_qwen_threshold_data.json")

if __name__ == "__main__":
    main()