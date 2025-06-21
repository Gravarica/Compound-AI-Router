#!/usr/bin/env python3
"""
Comprehensive Router Model Evaluation

Evaluates multiple router models and analyzes trade-offs between accuracy, 
precision, recall, and routing effectiveness for compound AI systems.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from src.routing import TransformerRouter, RandomRouter
from src.data.dataloader import ARCDataManager
from tqdm import tqdm


class RouterEvaluator:
    def __init__(self, output_dir: str = "./results/router_evaluation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load evaluation data once
        self.manager = ARCDataManager()
        self.eval_data = self.manager.get_arc_evaluation_set(use_router_test_split=False)
        print(f"Loaded {len(self.eval_data)} evaluation samples")
        
        # Distribution analysis
        easy_count = sum(1 for item in self.eval_data if item['label'] == 0)
        hard_count = sum(1 for item in self.eval_data if item['label'] == 1)
        print(f"Evaluation set distribution - Easy: {easy_count}, Hard: {hard_count}")

    def evaluate_single_router(self, model_path: str, model_name: str, 
                             confidence_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
                             is_random: bool = False) -> Dict[str, Any]:
        """Evaluate a single router model across multiple confidence thresholds"""
        
        print(f"\n=== Evaluating {model_name} ===")
        
        try:
            if is_random:
                router = RandomRouter(seed=42)
            else:
                router = TransformerRouter(model_name_or_path=model_path)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            return None
            
        results = {
            'model_name': model_name,
            'model_path': model_path,
            'threshold_results': {},
            'roc_analysis': {},
            'routing_effectiveness': {}
        }
        
        # Collect predictions for ROC analysis
        y_true = []
        y_probs = []  # Probability of being hard
        
        print("Collecting predictions...")
        for item in tqdm(self.eval_data, desc=f"Evaluating {model_name}"):
            text = item['text']
            true_label = item['label']
            
            if is_random:
                # For random router, get prediction
                predicted_difficulty, confidence = router.predict_difficulty(
                    query_text=text, confidence_threshold=0.5
                )
                # Random router confidence is meaningless, use 0.5 as probability
                y_probs.append(0.5)
            else:
                # Get prediction with confidence
                predicted_difficulty, confidence = router.predict_difficulty(
                    query_text=text, confidence_threshold=0.5  # Use 0.5 for prob extraction
                )
                
                # Convert confidence to probability of being hard
                if predicted_difficulty == 'hard':
                    y_probs.append(confidence)
                else:
                    y_probs.append(1.0 - confidence)
            
            y_true.append(true_label)
        
        # ROC Analysis
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall Analysis  
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)
        
        results['roc_analysis'] = {
            'auc': float(roc_auc),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist(),
            'pr_auc': float(pr_auc)
        }
        
        # Evaluate across confidence thresholds
        for threshold in confidence_thresholds:
            print(f"Threshold {threshold}...")
            
            y_pred = []
            y_pred_labels = []
            confidences = []
            
            for item in self.eval_data:
                text = item['text']
                predicted_difficulty, confidence = router.predict_difficulty(
                    query_text=text, confidence_threshold=threshold
                )
                
                if is_random:
                    # Random router uses simple mapping
                    predicted_label_id = 1 if predicted_difficulty == 'hard' else 0
                else:
                    predicted_label_id = router.inv_label_map[predicted_difficulty]
                
                y_pred.append(predicted_label_id)
                y_pred_labels.append(predicted_difficulty)
                confidences.append(confidence)
            
            # Basic metrics
            accuracy = np.mean(np.array(y_true) == np.array(y_pred))
            cm = confusion_matrix(y_true, y_pred)
            
            # Detailed classification report
            report = classification_report(y_true, y_pred, target_names=['easy', 'hard'], output_dict=True)
            
            # Routing effectiveness metrics
            routing_metrics = self._calculate_routing_effectiveness(y_true, y_pred, confidences)
            
            results['threshold_results'][threshold] = {
                'accuracy': float(accuracy),
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'routing_effectiveness': routing_metrics
            }
        
        return results

    def _calculate_routing_effectiveness(self, y_true: List[int], y_pred: List[int], 
                                       confidences: List[float]) -> Dict[str, float]:
        """Calculate metrics relevant to routing effectiveness in compound AI systems"""
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # True Positives, False Positives, True Negatives, False Negatives
        tp = np.sum((y_true == 1) & (y_pred == 1))  # Correctly identified hard
        fp = np.sum((y_true == 0) & (y_pred == 1))  # Easy questions sent to large model
        tn = np.sum((y_true == 0) & (y_pred == 0))  # Correctly identified easy
        fn = np.sum((y_true == 1) & (y_pred == 0))  # Hard questions sent to small model (BAD!)
        
        # Key routing metrics
        hard_recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # How many hard questions caught
        easy_precision = tn / (tn + fp) if (tn + fp) > 0 else 0  # How many "easy" predictions were correct
        
        # Cost simulation (assume large model is 10x more expensive)
        total_questions = len(y_true)
        questions_to_large_model = np.sum(y_pred == 1)  # Predicted hard
        questions_to_small_model = np.sum(y_pred == 0)  # Predicted easy
        
        # Cost relative to all-large-model baseline
        cost_ratio = (questions_to_small_model * 1 + questions_to_large_model * 10) / (total_questions * 10)
        
        # Accuracy impact from routing errors
        # Hard questions sent to small model likely get wrong answers
        routing_accuracy_penalty = fn / total_questions  # Fraction of questions that will likely be wrong
        
        return {
            'hard_recall': float(hard_recall),
            'easy_precision': float(easy_precision), 
            'cost_ratio': float(cost_ratio),
            'questions_to_large_model_pct': float(questions_to_large_model / total_questions),
            'routing_accuracy_penalty': float(routing_accuracy_penalty),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        }

    def compare_models(self, model_configs: List[Tuple[str, str, bool]]) -> Dict[str, Any]:
        """Compare multiple router models"""
        
        all_results = {}
        
        for model_path, model_name, is_random in model_configs:
            if is_random or os.path.exists(model_path):
                result = self.evaluate_single_router(model_path, model_name, is_random=is_random)
                if result:
                    all_results[model_name] = result
            else:
                print(f"Model path not found: {model_path}")
        
        # Save raw results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"router_comparison_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        # Generate comparison analysis
        self._generate_comparison_report(all_results, timestamp)
        self._generate_visualizations(all_results, timestamp)
        
        return all_results

    def _generate_comparison_report(self, results: Dict[str, Any], timestamp: str):
        """Generate markdown analysis report"""
        
        report_file = os.path.join(self.output_dir, f"router_analysis_{timestamp}.md")
        
        with open(report_file, 'w') as f:
            f.write("# Router Model Comparison Analysis\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This analysis compares different router models for difficulty classification in a compound AI system. ")
            f.write("The key insight is that **high recall on hard questions is more critical than overall accuracy** ")
            f.write("because routing hard questions to a small model leads to poor answers, while routing easy questions ")
            f.write("to a large model only increases cost.\n\n")
            
            # Best model summary table
            f.write("## Model Performance Summary\n\n")
            f.write("| Model | Best Accuracy | Hard Recall @ 0.7 | Cost Ratio @ 0.7 | Routing Penalty @ 0.7 |\n")
            f.write("|-------|---------------|-------------------|-------------------|----------------------|\n")
            
            for model_name, result in results.items():
                if not result:
                    continue
                    
                # Find best accuracy across thresholds
                best_acc = max([res['accuracy'] for res in result['threshold_results'].values()])
                
                # Get metrics at 0.7 threshold (common choice)
                if 0.7 in result['threshold_results']:
                    metrics_07 = result['threshold_results'][0.7]['routing_effectiveness']
                    hard_recall = metrics_07['hard_recall']
                    cost_ratio = metrics_07['cost_ratio']
                    penalty = metrics_07['routing_accuracy_penalty']
                    
                    f.write(f"| {model_name} | {best_acc:.3f} | {hard_recall:.3f} | {cost_ratio:.3f} | {penalty:.3f} |\n")
            
            f.write("\n## Key Insights\n\n")
            
            f.write("### 1. The Router Accuracy Challenge\n\n")
            f.write("Training routers to predict question difficulty from text alone is inherently challenging because:\n\n")
            f.write("- **Lack of reasoning context**: Text doesn't show the cognitive steps required\n")
            f.write("- **Subjective difficulty**: What's hard for one model may be easy for another\n")
            f.write("- **Limited features**: Question text provides minimal signal about reasoning complexity\n\n")
            
            f.write("### 2. Why Hard Question Recall Matters Most\n\n")
            f.write("In compound AI systems, the asymmetric cost of routing errors makes **hard question recall** ")
            f.write("the most critical metric:\n\n")
            f.write("- **False Negatives (Hard→Small)**: Lead to wrong answers, damaging system accuracy\n")
            f.write("- **False Positives (Easy→Large)**: Increase cost but maintain answer quality\n\n")
            f.write("**Trade-off**: A router with 90% hard recall and 60% overall accuracy may be preferable ")
            f.write("to one with 80% overall accuracy but only 70% hard recall.\n\n")
            
            f.write("### 3. Random Router Baseline\n\n")
            f.write("The random router provides a critical baseline for understanding router value:\n\n")
            f.write("- **Expected accuracy**: ~67% (matching class distribution)\n")
            f.write("- **Expected hard recall**: ~50% (random chance)\n")
            f.write("- **Cost ratio**: ~0.67 (67% of questions to large model)\n\n")
            f.write("**Any learned router must significantly outperform these baselines to justify complexity.**\n\n")
            
            f.write("### 4. Model-Specific Analysis\n\n")
            
            for model_name, result in results.items():
                if not result:
                    continue
                    
                f.write(f"#### {model_name}\n\n")
                
                # ROC AUC
                roc_auc = result['roc_analysis']['auc']
                f.write(f"- **ROC AUC**: {roc_auc:.3f}\n")
                
                # Best threshold analysis
                best_threshold = None
                best_hard_recall = 0
                
                for threshold, metrics in result['threshold_results'].items():
                    hard_recall = metrics['routing_effectiveness']['hard_recall']
                    if hard_recall > best_hard_recall:
                        best_hard_recall = hard_recall
                        best_threshold = threshold
                
                if best_threshold:
                    best_metrics = result['threshold_results'][best_threshold]
                    effectiveness = best_metrics['routing_effectiveness']
                    
                    f.write(f"- **Best hard recall**: {best_hard_recall:.3f} @ threshold {best_threshold}\n")
                    f.write(f"- **Cost impact**: {effectiveness['cost_ratio']:.3f}x baseline cost\n")
                    f.write(f"- **Questions to large model**: {effectiveness['questions_to_large_model_pct']:.1%}\n")
                    f.write(f"- **Routing accuracy penalty**: {effectiveness['routing_accuracy_penalty']:.3f}\n\n")
            
            f.write("## Trade-off Analysis Framework\n\n")
            f.write("When evaluating routers for compound AI systems, consider these key trade-offs:\n\n")
            
            f.write("### 1. **Accuracy vs Cost Trade-off**\n")
            f.write("- **High accuracy routers** may route most questions to large models (high cost)\n")
            f.write("- **Cost-efficient routers** may sacrifice some accuracy for better resource utilization\n")
            f.write("- **Optimal point**: Depends on cost differential between models and accuracy requirements\n\n")
            
            f.write("### 2. **Precision vs Recall Trade-off**\n")
            f.write("- **High precision** (few false positives): Minimize unnecessary large model usage\n")
            f.write("- **High recall** (few false negatives): Minimize hard questions sent to small models\n")
            f.write("- **Recommendation**: Bias toward high recall due to asymmetric error costs\n\n")
            
            f.write("### 3. **Complexity vs Performance Trade-off**\n")
            f.write("- **Simple models** (like random) have lower latency and maintenance overhead\n")
            f.write("- **Complex models** may achieve better routing accuracy but add system complexity\n")
            f.write("- **Decision criterion**: Performance improvement must justify operational overhead\n\n")
            
            f.write("### 4. **Conservative vs Aggressive Routing**\n")
            f.write("- **Conservative**: Route more questions to large model (higher cost, safer accuracy)\n")
            f.write("- **Aggressive**: Route more questions to small model (lower cost, riskier accuracy)\n")
            f.write("- **Tuning**: Adjust confidence thresholds based on cost/accuracy priorities\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **Establish minimum hard recall threshold** (e.g., >80%) before optimizing other metrics\n")
            f.write("2. **Compare against random baseline** - if improvement is marginal, use simpler approaches\n")
            f.write("3. **Consider confidence threshold tuning** as primary optimization lever\n")
            f.write("4. **Evaluate end-to-end system performance** including router latency overhead\n")
            f.write("5. **Monitor router performance drift** as underlying models and data evolve\n")
            f.write("6. **Consider ensemble or calibrated approaches** for production systems\n\n")
            
            f.write("## Technical Details\n\n")
            f.write("- **Evaluation dataset**: Full ARC test set (3,548 questions)\n")
            f.write("- **Easy questions**: 2,376 (ARC-Easy)\n") 
            f.write("- **Hard questions**: 1,172 (ARC-Challenge)\n")
            f.write("- **Cost model**: Small model = 1x, Large model = 10x\n")
        
        print(f"Analysis report saved to: {report_file}")

    def _generate_visualizations(self, results: Dict[str, Any], timestamp: str):
        """Generate comparison visualizations"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. ROC Curves
        plt.figure(figsize=(10, 8))
        for model_name, result in results.items():
            if not result:
                continue
            roc_data = result['roc_analysis']
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    label=f"{model_name} (AUC = {roc_data['auc']:.3f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Router Model ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"roc_comparison_{timestamp}.png"), dpi=300)
        plt.close()
        
        # 2. Hard Recall vs Cost Trade-off
        plt.figure(figsize=(12, 8))
        
        for model_name, result in results.items():
            if not result:
                continue
                
            thresholds = []
            hard_recalls = []
            cost_ratios = []
            
            for threshold, metrics in result['threshold_results'].items():
                effectiveness = metrics['routing_effectiveness']
                thresholds.append(threshold)
                hard_recalls.append(effectiveness['hard_recall'])
                cost_ratios.append(effectiveness['cost_ratio'])
            
            plt.scatter(cost_ratios, hard_recalls, s=100, alpha=0.7, label=model_name)
            
            # Connect points with lines
            sorted_pairs = sorted(zip(cost_ratios, hard_recalls))
            cost_sorted, recall_sorted = zip(*sorted_pairs)
            plt.plot(cost_sorted, recall_sorted, alpha=0.5, linewidth=1)
        
        plt.xlabel('Cost Ratio (vs All-Large-Model)')
        plt.ylabel('Hard Question Recall')
        plt.title('Hard Recall vs Cost Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Add ideal region annotation
        plt.axhspan(0.9, 1.0, alpha=0.2, color='green', label='High Recall Zone')
        plt.axvspan(0, 0.5, alpha=0.2, color='green', label='Low Cost Zone')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"recall_cost_tradeoff_{timestamp}.png"), dpi=300)
        plt.close()
        
        # 3. Threshold Analysis Heatmap
        models = [name for name in results.keys() if results[name]]
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Create data for heatmap
        hard_recall_data = []
        for model_name in models:
            recalls = []
            for threshold in thresholds:
                if threshold in results[model_name]['threshold_results']:
                    recall = results[model_name]['threshold_results'][threshold]['routing_effectiveness']['hard_recall']
                    recalls.append(recall)
                else:
                    recalls.append(np.nan)
            hard_recall_data.append(recalls)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(hard_recall_data, 
                   xticklabels=[f"{t:.1f}" for t in thresholds],
                   yticklabels=models,
                   annot=True, fmt='.3f', cmap='RdYlGn',
                   cbar_kws={'label': 'Hard Question Recall'})
        plt.title('Hard Question Recall by Model and Confidence Threshold')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"threshold_heatmap_{timestamp}.png"), dpi=300)
        plt.close()
        
        print(f"Visualizations saved to: {self.output_dir}")


def main():
    evaluator = RouterEvaluator()
    
    # Define models to evaluate (path, name, is_random)
    model_configs = [
        ("random", "Random-Baseline", True),  # Random router baseline
        ("./model-store/router_model", "DistilBERT-Original", False),
        ("./model-store/distilbert_hybrid", "DistilBERT-Hybrid", False), 
        ("./model-store/minilm_enhanced", "MiniLM-Enhanced", False),
        ("./model-store/roberta_enhanced", "RoBERTa-Enhanced", False),
    ]
    
    # Run comprehensive evaluation
    results = evaluator.compare_models(model_configs)
    
    print("\n=== Evaluation Complete ===")
    print(f"Results and analysis saved to: {evaluator.output_dir}")
    
    # Print quick summary
    print("\n=== Quick Summary ===")
    for model_name, result in results.items():
        if result and 0.7 in result['threshold_results']:
            metrics = result['threshold_results'][0.7]
            effectiveness = metrics['routing_effectiveness']
            print(f"{model_name:20s} | Acc: {metrics['accuracy']:.3f} | Hard Recall: {effectiveness['hard_recall']:.3f} | Cost: {effectiveness['cost_ratio']:.3f}")
    
    print(f"\nDetailed analysis saved to: {evaluator.output_dir}")


if __name__ == "__main__":
    main()