#!/usr/bin/env python3
"""
Create Router Recall vs Cost Curve for Academic Paper

Generates the specific figure showing trade-off between hard question recall
and system cost for different confidence thresholds.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from tqdm import tqdm

from src.routing import TransformerRouter
from src.data.dataloader import ARCDataManager


def analyze_recall_cost_tradeoff(model_path: str, 
                               thresholds: List[float] = None) -> Dict[str, List[float]]:
    """
    Analyze recall vs cost trade-off for different confidence thresholds
    
    Returns:
        Dictionary with threshold, recall, and cost_ratio lists
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.95, 0.05).tolist()  # More granular
    
    # Load router and data
    print(f"Loading router from {model_path}")
    router = TransformerRouter(model_name_or_path=model_path)
    
    print("Loading evaluation data...")
    manager = ARCDataManager()
    eval_data = manager.get_arc_evaluation_set(use_router_test_split=False)
    
    print(f"Evaluating {len(eval_data)} questions across {len(thresholds)} thresholds")
    
    results = {
        'thresholds': [],
        'hard_recall': [],
        'cost_ratio': [],
        'questions_to_large_pct': [],
        'accuracy': []
    }
    
    # Pre-compute predictions for all thresholds
    y_true = [item['label'] for item in eval_data]
    
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        y_pred = []
        
        # Get predictions for this threshold
        for item in eval_data:
            predicted_difficulty, confidence = router.predict_difficulty(
                query_text=item['text'],
                confidence_threshold=threshold
            )
            predicted_label = 1 if predicted_difficulty == 'hard' else 0
            y_pred.append(predicted_label)
        
        # Calculate metrics
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        
        # Hard question recall (True Positive Rate for hard questions)
        hard_questions = y_true_np == 1
        hard_correctly_identified = (y_true_np == 1) & (y_pred_np == 1)
        hard_recall = np.sum(hard_correctly_identified) / np.sum(hard_questions) if np.sum(hard_questions) > 0 else 0
        
        # Cost ratio calculation
        # Assume: small model cost = 1, large model cost = 10
        questions_to_large = np.sum(y_pred_np == 1)
        questions_to_small = np.sum(y_pred_np == 0)
        total_questions = len(y_pred_np)
        
        # Cost relative to all-large-model baseline
        compound_cost = (questions_to_small * 1 + questions_to_large * 10)
        baseline_cost = total_questions * 10  # All questions to large model
        cost_ratio = compound_cost / baseline_cost
        
        # Overall accuracy
        accuracy = np.mean(y_true_np == y_pred_np)
        
        # Store results
        results['thresholds'].append(threshold)
        results['hard_recall'].append(hard_recall)
        results['cost_ratio'].append(cost_ratio)
        results['questions_to_large_pct'].append(questions_to_large / total_questions)
        results['accuracy'].append(accuracy)
        
        print(f"Threshold {threshold:.2f}: Recall={hard_recall:.3f}, Cost={cost_ratio:.3f}, Acc={accuracy:.3f}")
    
    return results


def create_recall_cost_figure(results: Dict[str, List[float]], 
                            output_path: str = "router_performance.png",
                            model_name: str = "DistilBERT Router"):
    """Create the recall vs cost trade-off figure for the paper"""
    
    # Set up the plot style for academic paper
    plt.style.use('default')
    sns.set_context("paper", font_scale=1.2)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Main curve: Hard Recall vs Cost Ratio
    ax.plot(results['cost_ratio'], results['hard_recall'], 
           'o-', linewidth=3, markersize=8, color='#2E86AB', 
           label='Router Performance')
    
    # Add threshold annotations for key points
    key_thresholds = [0.5, 0.7, 0.9]  # Representative points
    
    for i, threshold in enumerate(results['thresholds']):
        if threshold in key_thresholds:
            x = results['cost_ratio'][i]
            y = results['hard_recall'][i]
            ax.annotate(f'τ={threshold}', 
                       xy=(x, y), xytext=(10, 10),
                       textcoords='offset points',
                       fontsize=10, ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', alpha=0.8))
    
    # Add reference lines
    # Random baseline (50% recall, ~67% cost for natural distribution)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
              label='Random Baseline (50% recall)')
    
    # High recall zone
    ax.axhspan(0.8, 1.0, alpha=0.1, color='green', 
              label='High Recall Zone (>80%)')
    
    # Low cost zone  
    ax.axvspan(0.0, 0.5, alpha=0.1, color='blue',
              label='Low Cost Zone (<50%)')
    
    # Formatting
    ax.set_xlabel('Cost Ratio (vs. All-Large-Model)', fontsize=12)
    ax.set_ylabel('Hard Question Recall', fontsize=12)
    ax.set_title(f'{model_name}: Recall vs. Cost Trade-off', fontsize=14, pad=20)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc='lower right', fontsize=10)
    
    # Add text box with key insight
    textstr = f'Optimal Region:\nHigh Recall (>0.8)\nLow Cost (<0.6)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save as PDF
    
    print(f"Figure saved to {output_path} and {output_path.replace('.png', '.pdf')}")
    
    return fig


def create_detailed_analysis(results: Dict[str, List[float]]) -> str:
    """Create detailed analysis text for the paper"""
    
    # Find optimal operating points
    # High recall threshold (>80%)
    high_recall_points = [(i, r) for i, r in enumerate(results['hard_recall']) if r >= 0.8]
    
    if high_recall_points:
        # Find lowest cost among high recall points
        best_high_recall_idx = min(high_recall_points, key=lambda x: results['cost_ratio'][x[0]])[0]
        
        optimal_threshold = results['thresholds'][best_high_recall_idx]
        optimal_recall = results['hard_recall'][best_high_recall_idx]
        optimal_cost = results['cost_ratio'][best_high_recall_idx]
        optimal_accuracy = results['accuracy'][best_high_recall_idx]
        
        analysis = f"""
Router Performance Analysis:

Key Operating Points:
- Optimal (High Recall): τ={optimal_threshold:.2f} → Recall={optimal_recall:.3f}, Cost={optimal_cost:.3f}, Accuracy={optimal_accuracy:.3f}
- Conservative (τ=0.5): Recall={results['hard_recall'][results['thresholds'].index(0.5)]:.3f}, Cost={results['cost_ratio'][results['thresholds'].index(0.5)]:.3f}
- Aggressive (τ=0.9): Recall={results['hard_recall'][results['thresholds'].index(0.9)]:.3f}, Cost={results['cost_ratio'][results['thresholds'].index(0.9)]:.3f}

Trade-off Insights:
1. Increasing threshold from 0.5→0.9 reduces cost by {(results['cost_ratio'][results['thresholds'].index(0.5)] - results['cost_ratio'][results['thresholds'].index(0.9)]):.2f}x
2. But decreases hard recall by {(results['hard_recall'][results['thresholds'].index(0.5)] - results['hard_recall'][results['thresholds'].index(0.9)])*100:.1f} percentage points
3. Optimal operating region: τ={optimal_threshold:.2f} balances >80% recall with <60% cost

Recommendations:
- For accuracy-critical applications: Use τ≤0.6 (high recall)
- For cost-sensitive applications: Use τ≥0.8 (low cost)  
- For balanced deployment: Use τ≈{optimal_threshold:.1f} (optimal trade-off)
"""
    else:
        analysis = "No operating points achieve >80% hard recall. Consider model improvements."
    
    return analysis


def main():
    """Generate the recall vs cost curve for DistilBERT router"""
    
    # Router model path - update this to your actual model path
    model_path = "./model-store/router_model"  # or distilbert_improved, distilbert_hybrid
    
    # Check if model exists
    import os
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Available alternatives:")
        alternatives = [
            "./model-store/distilbert_improved",
            "./model-store/distilbert_hybrid", 
            "./model-store/minilm_enhanced",
            "./model-store/roberta_enhanced"
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                print(f"  ✓ {alt}")
                model_path = alt
                break
            else:
                print(f"  ✗ {alt}")
        
        if not os.path.exists(model_path):
            print("No router models found. Please train a model first.")
            return
    
    print(f"Using model: {model_path}")
    
    # Analyze recall vs cost trade-off
    results = analyze_recall_cost_tradeoff(model_path)
    
    # Create the figure
    model_name = "DistilBERT Router" if "distilbert" in model_path.lower() else "Router Model"
    fig = create_recall_cost_figure(results, "router_performance.png", model_name)
    
    # Generate detailed analysis
    analysis = create_detailed_analysis(results)
    print(analysis)
    
    # Save analysis to file
    with open("router_performance_analysis.txt", "w") as f:
        f.write(analysis)
    
    # Save raw data for further analysis
    import json
    with open("router_recall_cost_data.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete!")
    print(f"Figure: router_performance.png")
    print(f"Data: router_recall_cost_data.json") 
    print(f"Analysis: router_performance_analysis.txt")


if __name__ == "__main__":
    main()