#!/usr/bin/env python3
"""
Create Paper-Ready GPT + Qwen Threshold Sensitivity Chart
========================================================
Generate publication-quality charts for threshold sensitivity analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def load_gpt_qwen_data():
    """Load the GPT + Qwen threshold sensitivity data"""
    
    with open('gpt_qwen_threshold_data.json', 'r') as f:
        data = json.load(f)
    
    return data

def create_paper_ready_chart():
    """Create paper-ready threshold sensitivity chart"""
    
    # Load data
    data = load_gpt_qwen_data()
    thresholds = data['thresholds']
    accuracies = data['system_accuracy']
    costs = [c for c in data['cost_per_1k']]
    pct_large = data['pct_to_large_model']
    
    # Academic paper styling
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 16,           # Larger base font
        'axes.titlesize': 20,      # Larger title
        'axes.labelsize': 18,      # Larger axis labels
        'xtick.labelsize': 14,     # Larger tick labels
        'ytick.labelsize': 14,
        'legend.fontsize': 14,     # Larger legend
        'figure.titlesize': 22,    # Larger figure title
        'lines.linewidth': 4,      # Thicker lines
        'axes.linewidth': 2,       # Thicker axis lines
        'grid.linewidth': 1.5,
        'grid.alpha': 0.3,
        'font.weight': 'normal',
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold'
    })
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(14, 10))
    
    # Colors (academic-friendly)
    color1 = '#1f77b4'  # Blue for accuracy
    color2 = '#d62728'  # Red for cost  
    color3 = '#2ca02c'  # Green for large model usage
    
    # Plot 1: System Accuracy
    ax1.set_xlabel('Router Confidence Threshold', fontweight='bold', fontsize=18)
    ax1.set_ylabel('System Accuracy', color=color1, fontweight='bold', fontsize=18)
    line1 = ax1.plot(thresholds, accuracies, 'o-', color=color1, linewidth=4, 
                    markersize=10, label='System Accuracy', markerfacecolor='white',
                    markeredgewidth=3, alpha=0.9)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=14)
    ax1.set_ylim(min(accuracies) - 0.02, max(accuracies) + 0.02)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cost per 1K queries
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cost per 1M Queries ($)', color=color2, fontweight='bold', fontsize=18)
    line2 = ax2.plot(thresholds, costs, 's-', color=color2, linewidth=4,
                    markersize=10, label='Total Cost', markerfacecolor='white',
                    markeredgewidth=3, alpha=0.9)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=14)
    ax2.set_ylim(0, max(costs) * 1.1)
    

    # Create legend for accuracy and cost only
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.02, 0.98),
              frameon=True, fancybox=True, shadow=True, fontsize=16,
              edgecolor='black', facecolor='white', framealpha=0.9)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for title
    
    # Save both PNG and PDF with high quality
    plt.savefig('paper_gpt_qwen_threshold_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('paper_gpt_qwen_threshold_analysis.pdf', bbox_inches='tight', facecolor='white')
    
    print("🚀 Creating Paper-Ready GPT + Qwen Threshold Chart...")
    print("📝 Using academic styling with larger fonts and PDF output")
    print("✅ Paper-ready chart saved as paper_gpt_qwen_threshold_analysis.png and paper_gpt_qwen_threshold_analysis.pdf")
    print("💡 Use PDF version for best quality in LaTeX documents!")
    
    # Show key insights
    print(f"\n📊 KEY INSIGHTS:")
    print(f"📈 Accuracy range: {min(accuracies):.1%} - {max(accuracies):.1%}")
    print(f"💰 Cost range: ${min(costs):.2f} - ${max(costs):.2f} per 1M queries")
    
    plt.show()

if __name__ == "__main__":
    create_paper_ready_chart()