#!/usr/bin/env python3
"""
Analyze Real Threshold Sensitivity Results

Takes the actual experimental results from run_threshold_experiments.py
and creates the threshold sensitivity analysis plot with REAL data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import os

def load_threshold_experiment_results(base_dir: str = "results/experiments/threshold_sensitivity"):
    """Load results from all threshold experiments"""
    
    print(f"🔍 Loading threshold experiment results from {base_dir}")
    
    if not os.path.exists(base_dir):
        print(f"❌ Results directory not found: {base_dir}")
        return None
    
    # Look for threshold directories
    threshold_dirs = []
    for item in os.listdir(base_dir):
        if item.startswith("threshold_") and os.path.isdir(os.path.join(base_dir, item)):
            threshold_dirs.append(item)
    
    if not threshold_dirs:
        print(f"❌ No threshold result directories found in {base_dir}")
        return None
    
    print(f"📊 Found {len(threshold_dirs)} threshold experiments")
    
    results = []
    
    for threshold_dir in sorted(threshold_dirs):
        # Extract threshold from directory name
        try:
            threshold = float(threshold_dir.split("_")[1])
        except:
            continue
            
        # Load results file
        results_file = os.path.join(base_dir, threshold_dir, "evaluation_results_full.json")
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                # Extract key metrics
                cost_summary = data.get('cost_summary', {})
                summary_metrics = cost_summary.get('summary_metrics', {})
                
                result = {
                    'threshold': threshold,
                    'system_accuracy': summary_metrics.get('accuracy', 0),
                    'avg_latency_ms': summary_metrics.get('avg_latency_ms', 0),
                    'total_cost': cost_summary.get('total_cost', 0),
                    'routing_stats': data.get('routing_stats', {}),
                    'model_usage': data.get('model_usage', {})
                }
                
                # Calculate percentage to large model from evaluation_stats
                eval_stats = data.get('evaluation_stats', {})
                if eval_stats:
                    total_queries = eval_stats.get('total_queries', 0)
                    large_model_queries = eval_stats.get('large_llm_usage', 0)
                    result['pct_to_large_model'] = (large_model_queries / total_queries * 100) if total_queries > 0 else 0
                else:
                    # Fallback: try to get from routing_stats
                    routing_stats = data.get('routing_stats', {})
                    if routing_stats:
                        total_queries = routing_stats.get('total_queries', 0)
                        large_model_queries = routing_stats.get('large_model_queries', 0)
                        result['pct_to_large_model'] = (large_model_queries / total_queries * 100) if total_queries > 0 else 0
                    else:
                        result['pct_to_large_model'] = 0
                
                results.append(result)
                print(f"✅ Loaded threshold {threshold:.2f}: Acc={result['system_accuracy']:.3f}, Cost=${result['total_cost']:.4f}")
                
            except Exception as e:
                print(f"⚠️  Failed to load {results_file}: {e}")
        else:
            print(f"⚠️  Results file not found: {results_file}")
    
    if not results:
        print("❌ No valid results found")
        return None
    
    # Sort by threshold
    results.sort(key=lambda x: x['threshold'])
    
    print(f"✅ Successfully loaded {len(results)} threshold experiments")
    return results

def create_threshold_sensitivity_plot(results, output_path="real_threshold_sensitivity_analysis"):
    """Create the threshold sensitivity plot with real experimental data"""
    
    # Extract data
    thresholds = [r['threshold'] for r in results]
    accuracies = [r['system_accuracy'] for r in results]
    costs = [r['total_cost'] * 1000 for r in results]  # Convert to cost per 1K queries
    pct_large = [r['pct_to_large_model'] for r in results]
    
    print(f"📊 Creating plot with {len(results)} data points")
    print(f"Threshold range: {min(thresholds):.2f} - {max(thresholds):.2f}")
    print(f"Accuracy range: {min(accuracies):.1%} - {max(accuracies):.1%}")
    print(f"Cost range: ${min(costs):.2f} - ${max(costs):.2f} per 1K queries")
    
    # Set up paper-ready plot styling
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'font.family': 'serif',
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })
    
    # Create figure with three y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary y-axis: System Accuracy
    color1 = '#1f77b4'  # Blue
    ax1.set_xlabel('Router Confidence Threshold', fontweight='bold')
    ax1.set_ylabel('System Accuracy', color=color1, fontweight='bold')
    
    line1 = ax1.plot(thresholds, accuracies, 'o-', color=color1, linewidth=3, 
                    markersize=8, label='System Accuracy', markerfacecolor='white', 
                    markeredgewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(min(accuracies) * 0.95, max(accuracies) * 1.02)
    
    # Secondary y-axis: Total Cost
    ax2 = ax1.twinx()
    color2 = '#d62728'  # Red
    ax2.set_ylabel('Cost per 1K Queries ($)', color=color2, fontweight='bold')
    
    line2 = ax2.plot(thresholds, costs, 's-', color=color2, linewidth=3, 
                    markersize=8, label='Total Cost', markerfacecolor='white',
                    markeredgewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(costs) * 1.1)
    
    # Third y-axis: Percentage to Large Model
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color3 = '#2ca02c'  # Green
    ax3.set_ylabel('Queries to Large Model (%)', color=color3, fontweight='bold')
    
    line3 = ax3.plot(thresholds, pct_large, '^-', color=color3, linewidth=3, 
                    markersize=8, label='% to Large Model', markerfacecolor='white',
                    markeredgewidth=2, alpha=0.8)
    ax3.tick_params(axis='y', labelcolor=color3)
    ax3.set_ylim(0, 100)
    
    # Find and highlight optimal points
    # Best accuracy/cost ratio (handle zero costs)
    accuracy_cost_ratios = []
    for acc, cost in zip(accuracies, costs):
        if cost > 0:
            accuracy_cost_ratios.append(acc/(cost/1000))  # Accuracy per dollar
        else:
            accuracy_cost_ratios.append(float('inf'))  # Infinite ratio for zero cost
    
    best_ratio_idx = np.argmax(accuracy_cost_ratios)
    optimal_threshold = thresholds[best_ratio_idx]
    optimal_accuracy = accuracies[best_ratio_idx]
    optimal_cost = costs[best_ratio_idx]
    
    # Highlight the sweet spot
    ax1.axvline(x=optimal_threshold, color='orange', linestyle='--', alpha=0.8, linewidth=2)
    ax1.text(optimal_threshold + 0.02, optimal_accuracy - 0.01, 
            f'Optimal\nτ={optimal_threshold:.2f}\n{optimal_accuracy:.1%}', 
            bbox=dict(boxstyle='round,pad=0.4', facecolor='orange', alpha=0.3),
            fontsize=10, ha='left', va='top')
    
    # Add annotations for specific thresholds
    key_points = [
        (min(thresholds), 'Conservative\n(Low threshold)'),
        (max(thresholds), 'Aggressive\n(High threshold)')
    ]
    
    for threshold, label in key_points:
        if threshold in thresholds:
            idx = thresholds.index(threshold)
            ax1.annotate(label, xy=(threshold, accuracies[idx]), 
                        xytext=(10, -20), textcoords='offset points',
                        fontsize=9, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    
    # Formatting
    ax1.set_xlim(min(thresholds) - 0.05, max(thresholds) + 0.05)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_title('Router Threshold Sensitivity Analysis\nGPT-4o-mini + Llama3.2-1B System (Real Data)', 
                 fontweight='bold', pad=20)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    
    all_lines = lines1 + lines2 + lines3
    all_labels = labels1 + labels2 + labels3
    
    ax1.legend(all_lines, all_labels, loc='center left', bbox_to_anchor=(0.02, 0.5),
              frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"✅ Real threshold sensitivity plot saved as {output_path}.png and {output_path}.pdf")
    
    return optimal_threshold, optimal_accuracy, optimal_cost

def generate_analysis_report(results, optimal_threshold, optimal_accuracy, optimal_cost):
    """Generate analysis report from real experimental data"""
    
    thresholds = [r['threshold'] for r in results]
    accuracies = [r['system_accuracy'] for r in results]
    costs = [r['total_cost'] * 1000 for r in results]
    pct_large = [r['pct_to_large_model'] for r in results]
    
    # Find extreme points
    max_acc_idx = np.argmax(accuracies)
    min_cost_idx = np.argmin(costs)
    
    report = f"""
REAL THRESHOLD SENSITIVITY ANALYSIS REPORT
==========================================
System: GPT-4o-mini + Llama3.2-1B with Transformer Router
Data: {len(results)} actual experimental runs
Generated: {Path().absolute()}

EXPERIMENTAL RESULTS SUMMARY:
---------------------------
Threshold Range: {min(thresholds):.2f} - {max(thresholds):.2f}
Accuracy Range: {min(accuracies):.1%} - {max(accuracies):.1%}
Cost Range: ${min(costs):.2f} - ${max(costs):.2f} per 1K queries
Large Model Usage: {min(pct_large):.1f}% - {max(pct_large):.1f}%

KEY OPERATING POINTS (FROM REAL DATA):
-------------------------------------

🎯 OPTIMAL (Best Accuracy/Cost Ratio):
   Threshold: {optimal_threshold:.2f}
   System Accuracy: {optimal_accuracy:.1%}
   Cost per 1K queries: ${optimal_cost:.2f}
   Large Model Usage: {results[thresholds.index(optimal_threshold)]['pct_to_large_model']:.1f}%

🔒 MAXIMUM ACCURACY:
   Threshold: {thresholds[max_acc_idx]:.2f}
   System Accuracy: {accuracies[max_acc_idx]:.1%}
   Cost per 1K queries: ${costs[max_acc_idx]:.2f}
   Large Model Usage: {pct_large[max_acc_idx]:.1f}%

💰 MINIMUM COST:
   Threshold: {thresholds[min_cost_idx]:.2f}
   System Accuracy: {accuracies[min_cost_idx]:.1%}
   Cost per 1K queries: ${costs[min_cost_idx]:.2f}
   Large Model Usage: {pct_large[min_cost_idx]:.1f}%

TRADE-OFF ANALYSIS:
------------------
• Moving from min cost to max accuracy:
  - Accuracy gain: {(accuracies[max_acc_idx] - accuracies[min_cost_idx]) * 100:.1f} percentage points
  - Cost increase: ${costs[max_acc_idx] - costs[min_cost_idx]:.2f} per 1K queries
  - Large model usage increase: {pct_large[max_acc_idx] - pct_large[min_cost_idx]:.1f} percentage points

• Optimal vs Conservative:
  - Accuracy difference: {(accuracies[max_acc_idx] - optimal_accuracy) * 100:.1f} percentage points
  - Cost savings: ${costs[max_acc_idx] - optimal_cost:.2f} per 1K queries

DEPLOYMENT RECOMMENDATIONS:
--------------------------
For Production Deployment:
✅ Use threshold τ={optimal_threshold:.2f} for optimal balance
✅ Expected accuracy: {optimal_accuracy:.1%}
✅ Expected cost: ${optimal_cost:.2f} per 1K queries

For Cost-Critical Applications:
💰 Use threshold τ={thresholds[min_cost_idx]:.2f}
💰 Saves ${optimal_cost - costs[min_cost_idx]:.2f} per 1K queries
⚠️  Accuracy drops by {(optimal_accuracy - accuracies[min_cost_idx]) * 100:.1f} percentage points

For Accuracy-Critical Applications:
🎯 Use threshold τ={thresholds[max_acc_idx]:.2f}
🎯 Gains {(accuracies[max_acc_idx] - optimal_accuracy) * 100:.1f} percentage points accuracy
💸 Costs additional ${costs[max_acc_idx] - optimal_cost:.2f} per 1K queries

EXPERIMENTAL VALIDATION:
-----------------------
This analysis is based on {len(results)} real evaluation runs using:
• Actual GPT-4o-mini API calls
• Actual Llama3.2-1B local inference
• Real ARC dataset evaluation
• Measured latency and costs
• Authentic routing decisions

Data files available in: results/experiments/threshold_sensitivity/
"""
    
    return report

def main():
    """Main function to analyze real threshold experiment results"""
    
    print("📊 Analyzing Real Threshold Sensitivity Results")
    print("📊 System: GPT-4o-mini + Llama3.2-1B")
    print("=" * 50)
    
    # Load results
    results = load_threshold_experiment_results()
    
    if not results:
        print("\n❌ No experimental results found!")
        print("💡 Run threshold experiments first:")
        print("   python scripts/run_threshold_experiments.py")
        return
    
    # Create analysis plot
    optimal_threshold, optimal_accuracy, optimal_cost = create_threshold_sensitivity_plot(
        results, "gpt_qwen_real_threshold_analysis"
    )
    
    # Generate report
    report = generate_analysis_report(results, optimal_threshold, optimal_accuracy, optimal_cost)
    
    print(report)
    
    # Save report
    with open("real_threshold_sensitivity_report.txt", "w") as f:
        f.write(report)
    
    # Save processed data
    processed_data = {
        'thresholds': [r['threshold'] for r in results],
        'system_accuracy': [r['system_accuracy'] for r in results],
        'cost_per_1k': [r['total_cost'] * 1000 for r in results],
        'pct_to_large_model': [r['pct_to_large_model'] for r in results],
        'optimal_threshold': optimal_threshold,
        'optimal_accuracy': optimal_accuracy,
        'optimal_cost': optimal_cost
    }
    
    with open("real_threshold_sensitivity_data.json", "w") as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"\n✅ Real threshold sensitivity analysis complete!")
    print(f"📊 Plot: gpt_qwen_real_threshold_analysis.png/.pdf")
    print(f"📝 Report: real_threshold_sensitivity_report.txt")
    print(f"📊 Data: real_threshold_sensitivity_data.json")

if __name__ == "__main__":
    main()