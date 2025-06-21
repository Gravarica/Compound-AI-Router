#!/usr/bin/env python3
"""
ARC Dataset Suitability Analyzer
Analyzes whether ARC dataset provides sufficient signal for routing decisions
and recommends alternative datasets and approaches.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ARCDatasetAnalyzer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.baseline_results = {}
        self.compound_results = {}
        
    def load_results(self):
        """Load baseline and compound results for ARC analysis."""
        print("🔍 Loading experiment results for ARC analysis...")
        
        # Load baseline results
        baseline_dir = self.results_dir / "baselines"
        if baseline_dir.exists():
            for model_dir in baseline_dir.iterdir():
                if model_dir.is_dir() and model_dir.name not in ['large_llm_only', 'small_llm_only']:
                    result_files = list(model_dir.glob("*_full.json"))
                    if result_files:
                        try:
                            with open(result_files[0], 'r') as f:
                                data = json.load(f)
                                self.baseline_results[model_dir.name] = data
                                print(f"✅ Loaded baseline: {model_dir.name}")
                        except Exception as e:
                            print(f"❌ Failed to load {model_dir.name}: {e}")
        
        # Load compound results (transformer router only)
        transformer_dir = self.results_dir / "experiments" / "compound" / "transformer_router"
        if transformer_dir.exists():
            for exp_dir in transformer_dir.iterdir():
                if exp_dir.is_dir():
                    result_files = list(exp_dir.glob("*_full.json"))
                    if result_files:
                        try:
                            with open(result_files[0], 'r') as f:
                                data = json.load(f)
                                self.compound_results[exp_dir.name] = data
                                print(f"✅ Loaded compound: {exp_dir.name}")
                        except Exception as e:
                            print(f"❌ Failed to load {exp_dir.name}: {e}")
    
    def analyze_arc_difficulty_distribution(self):
        """Analyze the distribution of easy vs hard questions in ARC."""
        print("\n📊 Analyzing ARC Dataset Difficulty Distribution...")
        
        # Aggregate all results to understand difficulty distribution
        all_results = []
        for data in self.baseline_results.values():
            results = data.get('results', [])
            all_results.extend(results)
        
        if not all_results:
            print("❌ No baseline results found")
            return None
        
        # Count difficulty distribution
        easy_count = sum(1 for r in all_results if r.get('true_difficulty') == 'easy')
        hard_count = sum(1 for r in all_results if r.get('true_difficulty') == 'hard')
        total_count = easy_count + hard_count
        
        print(f"ARC Dataset Composition:")
        print(f"• Easy questions: {easy_count} ({easy_count/total_count:.1%})")
        print(f"• Hard questions: {hard_count} ({hard_count/total_count:.1%})")
        
        return {
            'easy_count': easy_count,
            'hard_count': hard_count,
            'total_count': total_count,
            'easy_ratio': easy_count / total_count,
            'hard_ratio': hard_count / total_count
        }
    
    def analyze_model_performance_gaps(self):
        """Analyze performance gaps between small and large models on ARC."""
        print("\n📈 Analyzing Model Performance Gaps...")
        
        # Extract model performances
        model_performances = {}
        
        model_mapping = {
            'llama3_2_1b': 'Llama3.2 1B',
            'llama3_2_3b': 'Llama3.2 3B',
            'gemma2_2b': 'Gemma2 2B',
            'qwen2_5_1_5b': 'Qwen2.5 1.5B',
            'phi': 'Phi-2',
            'openai_gpt4o_mini': 'GPT-4o-mini',
            'claude_haiku': 'Claude Haiku'
        }
        
        for model_key, display_name in model_mapping.items():
            if model_key in self.baseline_results:
                data = self.baseline_results[model_key]
                cost_summary = data.get('cost_summary', {})
                summary_metrics = cost_summary.get('summary_metrics', {})
                
                # Analyze performance by difficulty
                results = data.get('results', [])
                easy_correct = sum(1 for r in results if r.get('true_difficulty') == 'easy' and r.get('correct'))
                easy_total = sum(1 for r in results if r.get('true_difficulty') == 'easy')
                hard_correct = sum(1 for r in results if r.get('true_difficulty') == 'hard' and r.get('correct')  )
                hard_total = sum(1 for r in results if r.get('true_difficulty') == 'hard')
                
                model_performances[display_name] = {
                    'overall_accuracy': summary_metrics.get('accuracy', 0),
                    'easy_accuracy': easy_correct / easy_total if easy_total > 0 else 0,
                    'hard_accuracy': hard_correct / hard_total if hard_total > 0 else 0,
                    'difficulty_gap': (easy_correct / easy_total - hard_correct / hard_total) if easy_total > 0 and hard_total > 0 else 0,
                    'model_type': 'Large' if model_key in ['openai_gpt4o_mini', 'claude_haiku'] else 'Small'
                }
        
        return model_performances
    
    def calculate_routing_potential(self, model_performances):
        """Calculate the theoretical routing potential based on model performance gaps."""
        print("\n🎯 Calculating Routing Potential...")
        
        # Find best small and large models
        small_models = {k: v for k, v in model_performances.items() if v['model_type'] == 'Small'}
        large_models = {k: v for k, v in model_performances.items() if v['model_type'] == 'Large'}
        
        if not small_models or not large_models:
            print("❌ Insufficient model data for routing analysis")
            return None
        
        best_small = max(small_models.items(), key=lambda x: x[1]['overall_accuracy'])
        best_large = max(large_models.items(), key=lambda x: x[1]['overall_accuracy'])
        
        # Calculate theoretical optimal routing
        best_small_easy = best_small[1]['easy_accuracy']
        best_large_hard = best_large[1]['hard_accuracy']
        
        # Assume 50/50 split for theoretical calculation
        theoretical_optimal = (best_small_easy + best_large_hard) / 2
        
        # Calculate actual performance gaps
        performance_gap = best_large[1]['overall_accuracy'] - best_small[1]['overall_accuracy']
        easy_performance_gap = best_large[1]['easy_accuracy'] - best_small[1]['easy_accuracy'] 
        hard_performance_gap = best_large[1]['hard_accuracy'] - best_small[1]['hard_accuracy']
        
        routing_potential = {
            'best_small_model': best_small[0],
            'best_large_model': best_large[0],
            'small_overall_acc': best_small[1]['overall_accuracy'],
            'large_overall_acc': best_large[1]['overall_accuracy'],
            'small_easy_acc': best_small[1]['easy_accuracy'],
            'small_hard_acc': best_small[1]['hard_accuracy'],
            'large_easy_acc': best_large[1]['easy_accuracy'],
            'large_hard_acc': best_large[1]['hard_accuracy'],
            'overall_performance_gap': performance_gap,
            'easy_performance_gap': easy_performance_gap,
            'hard_performance_gap': hard_performance_gap,
            'theoretical_optimal_acc': theoretical_optimal,
            'routing_benefit': theoretical_optimal - best_small[1]['overall_accuracy']
        }
        
        return routing_potential
    
    def create_arc_suitability_analysis(self, difficulty_dist, model_performances, routing_potential):
        """Create comprehensive visualization of ARC dataset suitability for routing."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ARC Dataset Suitability for Routing Analysis', fontsize=16, fontweight='bold')
        
        # 1. Difficulty Distribution (Top Left)
        if difficulty_dist:
            ax1 = axes[0, 0]
            labels = ['Easy Questions', 'Hard Questions']
            sizes = [difficulty_dist['easy_count'], difficulty_dist['hard_count']]
            colors = ['lightgreen', 'lightcoral']
            
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                              startangle=90)
            ax1.set_title('ARC Question Difficulty Distribution')
            
            # Add count annotations
            ax1.text(0, -1.3, f"Total Questions: {difficulty_dist['total_count']}", 
                    ha='center', fontsize=10, fontweight='bold')
        
        # 2. Model Performance by Difficulty (Top Middle)
        if model_performances:
            ax2 = axes[0, 1]
            
            models = list(model_performances.keys())
            easy_accs = [model_performances[m]['easy_accuracy'] * 100 for m in models]
            hard_accs = [model_performances[m]['hard_accuracy'] * 100 for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, easy_accs, width, label='Easy Questions', alpha=0.8, color='lightgreen')
            bars2 = ax2.bar(x + width/2, hard_accs, width, label='Hard Questions', alpha=0.8, color='lightcoral')
            
            ax2.set_xlabel('Models')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Model Performance by Question Difficulty')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. Performance Gaps (Top Right)
        if model_performances:
            ax3 = axes[0, 2]
            
            small_models = {k: v for k, v in model_performances.items() if v['model_type'] == 'Small'}
            large_models = {k: v for k, v in model_performances.items() if v['model_type'] == 'Large'}
            
            # Calculate average performances
            if small_models and large_models:
                avg_small_overall = np.mean([v['overall_accuracy'] for v in small_models.values()]) * 100
                avg_large_overall = np.mean([v['overall_accuracy'] for v in large_models.values()]) * 100
                avg_small_easy = np.mean([v['easy_accuracy'] for v in small_models.values()]) * 100  
                avg_large_easy = np.mean([v['easy_accuracy'] for v in large_models.values()]) * 100
                avg_small_hard = np.mean([v['hard_accuracy'] for v in small_models.values()]) * 100
                avg_large_hard = np.mean([v['hard_accuracy'] for v in large_models.values()]) * 100
                
                categories = ['Overall', 'Easy Questions', 'Hard Questions']
                small_avgs = [avg_small_overall, avg_small_easy, avg_small_hard]
                large_avgs = [avg_large_overall, avg_large_easy, avg_large_hard]
                
                x = np.arange(len(categories))
                width = 0.35
                
                ax3.bar(x - width/2, small_avgs, width, label='Small Models Avg', alpha=0.8, color='skyblue')
                ax3.bar(x + width/2, large_avgs, width, label='Large Models Avg', alpha=0.8, color='orange')
                
                ax3.set_ylabel('Accuracy (%)')
                ax3.set_title('Small vs Large Model Performance')
                ax3.set_xticks(x)
                ax3.set_xticklabels(categories)
                ax3.legend()
                ax3.grid(axis='y', alpha=0.3)
                
                # Add gap annotations
                for i, (small, large) in enumerate(zip(small_avgs, large_avgs)):
                    gap = large - small
                    ax3.annotate(f'Gap: {gap:.1f}%', 
                               xy=(i, max(small, large) + 2), 
                               ha='center', fontsize=9, fontweight='bold')
        
        # 4. Routing Potential Analysis (Bottom Left)
        if routing_potential:
            ax4 = axes[1, 0]
            
            scenarios = ['Small Model Only', 'Large Model Only', 'Theoretical Optimal\n(Perfect Routing)']
            accuracies = [
                routing_potential['small_overall_acc'] * 100,
                routing_potential['large_overall_acc'] * 100,
                routing_potential['theoretical_optimal_acc'] * 100
            ]
            colors = ['lightblue', 'orange', 'lightgreen']
            
            bars = ax4.bar(scenarios, accuracies, color=colors, alpha=0.8)
            ax4.set_ylabel('Accuracy (%)')
            ax4.set_title('Routing Potential Analysis')
            ax4.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Add routing benefit annotation
            benefit = routing_potential['routing_benefit'] * 100
            ax4.text(0.5, max(accuracies) * 0.8, f'Max Routing Benefit: {benefit:.1f}%',
                    transform=ax4.transData, ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 5. Current Router Performance vs Potential (Bottom Middle)
        ax5 = axes[1, 1]
        
        # This would need actual router results - placeholder for now
        current_router_acc = 66  # Your current DistilBERT router accuracy
        random_baseline = 50     # Random routing baseline
        
        if routing_potential:
            theoretical_max = routing_potential['theoretical_optimal_acc'] * 100
            current_system_acc = 79  # From your compound results
            
            performance_metrics = ['Random Routing', 'Current DistilBERT', 'Theoretical Maximum']
            router_accs = [random_baseline, current_router_acc, 100]  # Router accuracy
            system_accs = [60, current_system_acc, theoretical_max]   # System accuracy (estimated)
            
            x = np.arange(len(performance_metrics))
            width = 0.35
            
            bars1 = ax5.bar(x - width/2, router_accs, width, label='Router Accuracy', alpha=0.8, color='lightcoral')
            bars2 = ax5.bar(x + width/2, system_accs, width, label='System Accuracy', alpha=0.8, color='lightgreen')
            
            ax5.set_ylabel('Accuracy (%)')
            ax5.set_title('Current vs Potential Performance')
            ax5.set_xticks(x)
            ax5.set_xticklabels(performance_metrics, rotation=15)
            ax5.legend()
            ax5.grid(axis='y', alpha=0.3)
            ax5.set_ylim(0, 105)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2, height + 1,
                            f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
        
        # 6. Dataset Suitability Assessment (Bottom Right)
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create assessment summary
        assessment_text = "ARC Dataset Suitability Assessment\n\n"
        
        if routing_potential:
            benefit = routing_potential['routing_benefit'] * 100
            overall_gap = routing_potential['overall_performance_gap'] * 100
            
            if benefit > 10:
                suitability = "HIGH"
                color = 'green'
                assessment_text += f"✅ HIGH SUITABILITY\n"
                assessment_text += f"• Significant routing benefit: {benefit:.1f}%\n"
                assessment_text += f"• Large performance gap: {overall_gap:.1f}%\n"
            elif benefit > 5:
                suitability = "MEDIUM"
                color = 'orange'
                assessment_text += f"⚠️ MEDIUM SUITABILITY\n"
                assessment_text += f"• Moderate routing benefit: {benefit:.1f}%\n"
                assessment_text += f"• Performance gap: {overall_gap:.1f}%\n"
            else:
                suitability = "LOW"
                color = 'red'
                assessment_text += f"❌ LOW SUITABILITY\n"
                assessment_text += f"• Limited routing benefit: {benefit:.1f}%\n"
                assessment_text += f"• Small performance gap: {overall_gap:.1f}%\n"
            
            assessment_text += f"\nCurrent Router Gap to Optimal:\n"
            assessment_text += f"• Router Accuracy: 66% vs 100% (34% gap)\n"
            assessment_text += f"• System needs {(100-current_router_acc)/100*benefit:.1f}% improvement\n"
            
            # Add recommendations
            assessment_text += f"\nRecommendations:\n"
            if benefit < 5:
                assessment_text += f"• Try alternative datasets (MMLU, GSM8K)\n"
                assessment_text += f"• Consider confidence-based routing\n"
            else:
                assessment_text += f"• Improve router architecture\n"
                assessment_text += f"• Add more training data\n"
                assessment_text += f"• Try ensemble approaches\n"
        
        ax6.text(0.05, 0.95, assessment_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('arc_dataset_suitability_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_recommendations_report(self, difficulty_dist, model_performances, routing_potential):
        """Generate detailed recommendations based on ARC analysis."""
        print("\n" + "="*80)
        print("📊 ARC DATASET SUITABILITY ANALYSIS REPORT")
        print("="*80)
        
        if routing_potential:
            benefit = routing_potential['routing_benefit'] * 100
            overall_gap = routing_potential['overall_performance_gap'] * 100
            
            print(f"\n🎯 ROUTING POTENTIAL ASSESSMENT")
            print(f"-" * 40)
            print(f"Best Small Model: {routing_potential['best_small_model']}")
            print(f"Best Large Model: {routing_potential['best_large_model']}")
            print(f"Overall Performance Gap: {overall_gap:.1f}%")
            print(f"Theoretical Routing Benefit: {benefit:.1f}%")
            print(f"Theoretical Optimal Accuracy: {routing_potential['theoretical_optimal_acc']:.1%}")
            
            print(f"\n📈 CURRENT ROUTER ASSESSMENT")
            print(f"-" * 40)
            print(f"Current DistilBERT Router Accuracy: 66%")
            print(f"Gap to Perfect Router: 34%")
            print(f"Current System Accuracy: ~79%")
            
            if benefit > 10:
                print(f"✅ ARC DATASET: HIGH SUITABILITY for routing")
                print(f"   Significant potential benefit justifies router development")
            elif benefit > 5:
                print(f"⚠️  ARC DATASET: MEDIUM SUITABILITY for routing")
                print(f"   Moderate benefit, but router improvement needed")
            else:
                print(f"❌ ARC DATASET: LOW SUITABILITY for routing")
                print(f"   Limited benefit, consider alternative datasets")
        
        print(f"\n💡 STRATEGIC RECOMMENDATIONS")
        print(f"-" * 40)
        
        recommendations = []
        
        if routing_potential and routing_potential['routing_benefit'] > 0.05:  # >5% benefit
            recommendations.extend([
                "🔧 ROUTER IMPROVEMENT APPROACHES:",
                "   • Try DeBERTa or RoBERTa instead of DistilBERT",
                "   • Increase model size (BERT-large vs BERT-base)",
                "   • Add confidence-based routing threshold tuning",
                "   • Implement ensemble routing (multiple router consensus)",
                "   • Use gradient boosting or other ML approaches",
                "",
                "📊 TRAINING DATA ENHANCEMENT:",
                "   • Augment ARC training data with synthetic examples",
                "   • Use active learning to identify difficult boundary cases",
                "   • Cross-train on multiple datasets simultaneously",
            ])
        else:
            recommendations.extend([
                "⚠️  ARC LIMITED ROUTING SIGNAL - TRY ALTERNATIVE DATASETS:",
                "   • MMLU: Broad knowledge across 57 subjects",
                "   • GSM8K: Mathematical reasoning problems", 
                "   • HellaSwag: Commonsense reasoning",
                "   • RACE: Reading comprehension",
                "   • DROP: Discrete reasoning over paragraphs",
                "",
                "🔄 ALTERNATIVE ROUTING STRATEGIES:",
                "   • Confidence-based routing (model uncertainty)",
                "   • Token-length based routing (complexity proxy)",
                "   • Multi-stage routing (cascade approach)",
                "   • Ensemble routing with multiple criteria",
            ])
        
        recommendations.extend([
            "",
            "📈 EVALUATION STRATEGY:",
            "   • Test router on multiple datasets before deployment",
            "   • Use cross-dataset evaluation to assess generalization",
            "   • Monitor router performance degradation over time",
            "   • A/B testing with random baseline for comparison"
        ])
        
        for rec in recommendations:
            print(rec)
        
        print(f"\n🎯 NEXT STEPS PRIORITY")
        print(f"-" * 40)
        
        if routing_potential and routing_potential['routing_benefit'] > 0.1:  # >10% benefit
            next_steps = [
                "1. 🚀 HIGH PRIORITY: Improve DistilBERT router (good dataset signal)",
                "2. 🔧 Try DeBERTa-v3-base for better routing accuracy",
                "3. 📊 Implement confidence-based routing thresholds",
                "4. 🧪 A/B test improved router vs current system"
            ]
        else:
            next_steps = [
                "1. 📊 HIGH PRIORITY: Evaluate on MMLU and GSM8K datasets",
                "2. 🔄 Implement confidence-based routing as alternative",
                "3. 🧪 Test with synthetic routing datasets",
                "4. 📈 Consider non-ML routing strategies (length, keywords)"
            ]
        
        for step in next_steps:
            print(step)
        
        print("\n" + "="*80)
        print("ARC suitability analysis complete!")
        print("Generated visualization: arc_dataset_suitability_analysis.png")
        print("="*80)
    
    def run_analysis(self):
        """Run the complete ARC dataset suitability analysis."""
        print("🚀 Starting ARC Dataset Suitability Analysis")
        print("="*60)
        
        # Load all data
        self.load_results()
        
        if not self.baseline_results:
            print("❌ No baseline results found!")
            return
        
        # Analyze dataset characteristics
        difficulty_dist = self.analyze_arc_difficulty_distribution()
        model_performances = self.analyze_model_performance_gaps()
        routing_potential = self.calculate_routing_potential(model_performances)
        
        # Generate visualizations
        print("\n📈 Generating ARC suitability analysis...")
        self.create_arc_suitability_analysis(difficulty_dist, model_performances, routing_potential)
        
        # Generate recommendations
        self.generate_recommendations_report(difficulty_dist, model_performances, routing_potential)

if __name__ == "__main__":
    analyzer = ARCDatasetAnalyzer()
    analyzer.run_analysis()