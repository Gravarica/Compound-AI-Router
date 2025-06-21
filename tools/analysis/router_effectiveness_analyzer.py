#!/usr/bin/env python3
"""
Router Effectiveness Analyzer
Compares DistilBERT Transformer Router vs Random Router vs Oracle Router
to determine if the router brings value to the compound AI system.
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

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

class RouterEffectivenessAnalyzer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.router_results = {}
        self.baseline_results = {}
        
    def load_router_experiments(self):
        """Load all router experiment results for comparison."""
        print("🔍 Loading router experiment results...")
        
        # Load transformer router results
        transformer_dir = self.results_dir / "experiments" / "compound" / "transformer_router"
        if transformer_dir.exists():
            for exp_dir in transformer_dir.iterdir():
                if exp_dir.is_dir():
                    result_files = list(exp_dir.glob("*_full.json"))
                    if result_files:
                        try:
                            with open(result_files[0], 'r') as f:
                                data = json.load(f)
                                self.router_results[f"transformer_{exp_dir.name}"] = {
                                    'data': data,
                                    'router_type': 'DistilBERT Transformer',
                                    'experiment': exp_dir.name
                                }
                                print(f"✅ Loaded transformer: {exp_dir.name}")
                        except Exception as e:
                            print(f"❌ Failed to load {exp_dir.name}: {e}")
        
        # Look for random router results
        random_patterns = ["random", "baseline_random"]
        for pattern in random_patterns:
            random_files = list(self.results_dir.rglob(f"*{pattern}*_full.json"))
            for file_path in random_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        self.router_results[f"random_{file_path.stem}"] = {
                            'data': data,
                            'router_type': 'Random Router',
                            'experiment': file_path.stem
                        }
                        print(f"✅ Loaded random router: {file_path.stem}")
                except Exception as e:
                    print(f"❌ Failed to load {file_path}: {e}")
        
        # Look for oracle router results
        oracle_patterns = ["oracle"]
        for pattern in oracle_patterns:
            oracle_files = list(self.results_dir.rglob(f"*{pattern}*_full.json"))
            for file_path in oracle_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        self.router_results[f"oracle_{file_path.stem}"] = {
                            'data': data,
                            'router_type': 'Oracle Router',
                            'experiment': file_path.stem
                        }
                        print(f"✅ Loaded oracle router: {file_path.stem}")
                except Exception as e:
                    print(f"❌ Failed to load {file_path}: {e}")
        
        # Load baseline results for comparison
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
    
    def extract_router_metrics(self, data: Dict) -> Dict:
        """Extract routing-specific metrics from experiment data."""
        cost_summary = data.get('cost_summary', {})
        summary_metrics = cost_summary.get('summary_metrics', {})
        eval_stats = cost_summary.get('evaluation_stats', {})
        
        # Calculate router accuracy if we have individual results
        results = data.get('results', [])
        router_correct = 0
        router_decisions = 0
        
        # Analyze individual routing decisions
        easy_routed_to_small = 0
        hard_routed_to_large = 0
        easy_total = 0
        hard_total = 0
        
        for result in results:
            true_difficulty = result.get('true_difficulty', '')
            chosen_llm = result.get('chosen_llm', '')
            
            if true_difficulty and chosen_llm:
                router_decisions += 1
                
                if true_difficulty == 'easy':
                    easy_total += 1
                    if chosen_llm == 'small':
                        easy_routed_to_small += 1
                        router_correct += 1
                elif true_difficulty == 'hard':
                    hard_total += 1
                    if chosen_llm == 'large':
                        hard_routed_to_large += 1
                        router_correct += 1
        
        router_accuracy = router_correct / router_decisions if router_decisions > 0 else 0
        small_model_usage = eval_stats.get('small_llm_usage', 0)
        total_queries = eval_stats.get('total_queries', 0)
        small_usage_ratio = small_model_usage / total_queries if total_queries > 0 else 0
        
        return {
            'system_accuracy': summary_metrics.get('accuracy', 0),
            'avg_latency_ms': summary_metrics.get('avg_latency_ms', 0),
            'total_cost': summary_metrics.get('total_cost', 0),
            'cost_per_query': summary_metrics.get('cost_per_query', 0),
            'router_accuracy': router_accuracy,
            'small_usage_ratio': small_usage_ratio,
            'total_queries': total_queries,
            'easy_correct_routing': easy_routed_to_small / easy_total if easy_total > 0 else 0,
            'hard_correct_routing': hard_routed_to_large / hard_total if hard_total > 0 else 0,
            'routing_decisions': router_decisions
        }
    
    def create_router_comparison_dataframe(self) -> pd.DataFrame:
        """Create comparison DataFrame for router analysis."""
        comparison_data = []
        
        for name, info in self.router_results.items():
            metrics = self.extract_router_metrics(info['data'])
            metrics['experiment_name'] = name
            metrics['router_type'] = info['router_type']
            metrics['experiment'] = info['experiment']
            
            # Parse model combination from experiment name
            exp_name = info['experiment']
            if 'gpt' in exp_name.lower():
                metrics['large_model'] = 'GPT-4o-mini'
            elif 'claude' in exp_name.lower():
                metrics['large_model'] = 'Claude Haiku'
            else:
                metrics['large_model'] = 'Unknown'
                
            if 'gemma' in exp_name.lower():
                metrics['small_model'] = 'Gemma2 2B'
            elif 'qwen' in exp_name.lower():
                metrics['small_model'] = 'Qwen2.5 1.5B'
            elif 'phi' in exp_name.lower():
                metrics['small_model'] = 'Phi-2'
            elif '3b' in exp_name.lower():
                metrics['small_model'] = 'Llama3.2 3B'
            elif '1b' in exp_name.lower():
                metrics['small_model'] = 'Llama3.2 1B'
            else:
                metrics['small_model'] = 'Unknown'
            
            comparison_data.append(metrics)
        
        return pd.DataFrame(comparison_data)
    
    def analyze_router_effectiveness(self, df: pd.DataFrame):
        """Create comprehensive router effectiveness analysis."""
        if df.empty:
            print("❌ No router data available for analysis")
            return
            
        # Create the analysis plots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Overall title
        fig.suptitle('Router Effectiveness Analysis: DistilBERT vs Random vs Oracle', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Router Accuracy Comparison (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        router_acc_data = df.groupby('router_type')['router_accuracy'].mean()
        bars1 = ax1.bar(router_acc_data.index, router_acc_data.values * 100, 
                       color=['#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Router Decision Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Router Accuracy (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, router_acc_data.values * 100):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add benchmark line at 66%
        ax1.axhline(y=66, color='red', linestyle='--', alpha=0.7, label='Current DistilBERT (66%)')
        ax1.legend()
        
        # 2. System Accuracy Comparison (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        sys_acc_data = df.groupby('router_type')['system_accuracy'].mean()
        bars2 = ax2.bar(sys_acc_data.index, sys_acc_data.values * 100,
                       color=['#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_title('Overall System Accuracy', fontsize=14, fontweight='bold')
        ax2.set_ylabel('System Accuracy (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars2, sys_acc_data.values * 100):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Cost Efficiency Comparison (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        cost_data = df.groupby('router_type')['cost_per_query'].mean()
        bars3 = ax3.bar(cost_data.index, cost_data.values * 1000000,
                       color=['#ff7f0e', '#2ca02c', '#d62728'])
        ax3.set_title('Cost per Query', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Cost per Query ($/Million)')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars3, cost_data.values * 1000000):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'${val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Router vs System Performance Scatter (Second Row Left)
        ax4 = fig.add_subplot(gs[1, :2])
        colors = {'DistilBERT Transformer': '#ff7f0e', 'Random Router': '#2ca02c', 'Oracle Router': '#d62728'}
        
        for router_type in df['router_type'].unique():
            data_subset = df[df['router_type'] == router_type]
            ax4.scatter(data_subset['router_accuracy'] * 100, 
                       data_subset['system_accuracy'] * 100,
                       label=router_type, s=100, alpha=0.7, color=colors.get(router_type, 'blue'))
        
        ax4.set_xlabel('Router Accuracy (%)')
        ax4.set_ylabel('System Accuracy (%)')
        ax4.set_title('Router Accuracy vs System Performance', fontsize=14, fontweight='bold')
        ax4.grid(alpha=0.3)
        ax4.legend()
        
        # Add diagonal reference line
        min_val, max_val = 0, 100
        ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect Correlation')
        
        # 5. Small Model Usage Comparison (Second Row Right)
        ax5 = fig.add_subplot(gs[1, 2])
        usage_data = df.groupby('router_type')['small_usage_ratio'].mean()
        bars5 = ax5.bar(usage_data.index, usage_data.values * 100,
                       color=['#ff7f0e', '#2ca02c', '#d62728'])
        ax5.set_title('Small Model Usage Rate', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Small Model Usage (%)')
        ax5.set_ylim(0, 100)
        ax5.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars5, usage_data.values * 100):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Router Performance by Model Combination (Third Row)
        ax6 = fig.add_subplot(gs[2, :])
        
        # Filter for DistilBERT results only
        distilbert_df = df[df['router_type'] == 'DistilBERT Transformer']
        if not distilbert_df.empty:
            distilbert_df['combination'] = distilbert_df['small_model'] + ' → ' + distilbert_df['large_model']
            
            # Create grouped bar chart
            x_pos = np.arange(len(distilbert_df['combination']))
            width = 0.35
            
            bars1 = ax6.bar(x_pos - width/2, distilbert_df['router_accuracy'] * 100, width, 
                           label='Router Accuracy', alpha=0.8, color='#ff7f0e')
            bars2 = ax6.bar(x_pos + width/2, distilbert_df['system_accuracy'] * 100, width,
                           label='System Accuracy', alpha=0.8, color='#1f77b4')
            
            ax6.set_xlabel('Model Combination')
            ax6.set_ylabel('Accuracy (%)')
            ax6.set_title('DistilBERT Router Performance by Model Combination', fontsize=14, fontweight='bold')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(distilbert_df['combination'], rotation=45, ha='right')
            ax6.legend()
            ax6.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars1, distilbert_df['router_accuracy'] * 100):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
            for bar, val in zip(bars2, distilbert_df['system_accuracy'] * 100):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
        
        # 7. Statistical Summary (Bottom Row)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Create summary statistics table
        summary_stats = []
        for router_type in df['router_type'].unique():
            data_subset = df[df['router_type'] == router_type]
            if not data_subset.empty:
                stats = {
                    'Router Type': router_type,
                    'Avg Router Acc': f"{data_subset['router_accuracy'].mean():.1%}",
                    'Avg System Acc': f"{data_subset['system_accuracy'].mean():.1%}",
                    'Avg Cost/Query': f"${data_subset['cost_per_query'].mean() * 1000000:.2f}/M",
                    'Avg Small Usage': f"{data_subset['small_usage_ratio'].mean():.1%}",
                    'Sample Size': len(data_subset)
                }
                summary_stats.append(stats)
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            table = ax7.table(cellText=summary_df.values, colLabels=summary_df.columns,
                             cellLoc='center', loc='center', bbox=[0.1, 0.3, 0.8, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#E6E6FA')
                else:
                    cell.set_facecolor('#F8F8FF')
        
        ax7.set_title('Router Performance Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('router_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_router_insights_report(self, df: pd.DataFrame):
        """Generate detailed insights about router effectiveness."""
        print("\n" + "="*80)
        print("🎯 ROUTER EFFECTIVENESS ANALYSIS REPORT")
        print("="*80)
        
        if df.empty:
            print("❌ No router data available for analysis")
            return
        
        # Overall router performance comparison
        print("\n📊 ROUTER PERFORMANCE COMPARISON")
        print("-" * 50)
        
        router_summary = df.groupby('router_type').agg({
            'router_accuracy': ['mean', 'std', 'count'],
            'system_accuracy': ['mean', 'std'],
            'cost_per_query': 'mean',
            'small_usage_ratio': 'mean'
        }).round(3)
        
        print("Router Decision Accuracy:")
        for router_type in df['router_type'].unique():
            data_subset = df[df['router_type'] == router_type]
            if not data_subset.empty:
                mean_acc = data_subset['router_accuracy'].mean()
                std_acc = data_subset['router_accuracy'].std()
                count = len(data_subset)
                print(f"• {router_type:20}: {mean_acc:.1%} ± {std_acc:.1%} (n={count})")
        
        print("\nSystem Accuracy:")
        for router_type in df['router_type'].unique():
            data_subset = df[df['router_type'] == router_type]
            if not data_subset.empty:
                mean_acc = data_subset['system_accuracy'].mean()
                print(f"• {router_type:20}: {mean_acc:.1%}")
        
        # Key insights
        print("\n🔍 KEY INSIGHTS")
        print("-" * 50)
        
        insights = []
        
        # Compare DistilBERT to Random
        distilbert_data = df[df['router_type'] == 'DistilBERT Transformer']
        random_data = df[df['router_type'] == 'Random Router']
        oracle_data = df[df['router_type'] == 'Oracle Router']
        
        if not distilbert_data.empty and not random_data.empty:
            distilbert_router_acc = distilbert_data['router_accuracy'].mean()
            random_router_acc = random_data['router_accuracy'].mean()
            
            if distilbert_router_acc > random_router_acc + 0.05:  # 5% threshold
                insights.append(f"✅ DistilBERT router significantly outperforms random routing "
                              f"({distilbert_router_acc:.1%} vs {random_router_acc:.1%})")
            else:
                insights.append(f"⚠️  DistilBERT router only marginally better than random routing "
                              f"({distilbert_router_acc:.1%} vs {random_router_acc:.1%})")
            
            # System accuracy comparison
            distilbert_sys_acc = distilbert_data['system_accuracy'].mean()
            random_sys_acc = random_data['system_accuracy'].mean()
            
            sys_diff = distilbert_sys_acc - random_sys_acc
            insights.append(f"📈 System accuracy difference (DistilBERT vs Random): {sys_diff:+.1%}")
        
        # Oracle comparison
        if not oracle_data.empty and not distilbert_data.empty:
            oracle_router_acc = oracle_data['router_accuracy'].mean()
            oracle_sys_acc = oracle_data['system_accuracy'].mean()
            distilbert_router_acc = distilbert_data['router_accuracy'].mean()
            distilbert_sys_acc = distilbert_data['system_accuracy'].mean()
            
            router_gap = oracle_router_acc - distilbert_router_acc
            sys_gap = oracle_sys_acc - distilbert_sys_acc
            
            insights.append(f"🎯 Router accuracy gap to oracle: {router_gap:.1%}")
            insights.append(f"🎯 System accuracy gap to oracle: {sys_gap:.1%}")
            
            if router_gap > 0.2:  # 20% gap
                insights.append("❌ Large gap to oracle suggests significant room for router improvement")
            elif router_gap > 0.1:  # 10% gap
                insights.append("⚠️  Moderate gap to oracle suggests router improvement needed")
            else:
                insights.append("✅ Small gap to oracle suggests router is performing well")
        
        # Router accuracy assessment
        if not distilbert_data.empty:
            avg_router_acc = distilbert_data['router_accuracy'].mean()
            if avg_router_acc < 0.6:
                insights.append("❌ Router accuracy below 60% - consider alternative routing strategies")
            elif avg_router_acc < 0.7:
                insights.append("⚠️  Router accuracy below 70% - room for significant improvement")
            else:
                insights.append("✅ Router accuracy above 70% - performing reasonably well")
        
        # Dataset suitability assessment
        all_router_accs = df['router_accuracy'].values
        if len(all_router_accs) > 1:
            router_variance = np.var(all_router_accs)
            if router_variance < 0.01:  # Low variance
                insights.append("⚠️  Low variance in router accuracy across experiments suggests "
                              "ARC dataset may have limited routing signal")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i:2d}. {insight}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 50)
        
        recommendations = []
        
        if not distilbert_data.empty:
            avg_router_acc = distilbert_data['router_accuracy'].mean()
            
            if avg_router_acc < 0.65:
                recommendations.append("🔧 Retrain router with different architecture (e.g., RoBERTa, DeBERTa)")
                recommendations.append("📊 Evaluate on additional datasets (MMLU, GSM8K, HellaSwag)")
                recommendations.append("🎯 Consider confidence-based routing instead of binary classification")
                
            if not random_data.empty:
                sys_improvement = distilbert_data['system_accuracy'].mean() - random_data['system_accuracy'].mean()
                if sys_improvement < 0.02:  # Less than 2% improvement
                    recommendations.append("⚡ Current router provides minimal benefit - consider alternative approaches")
                    recommendations.append("📈 Try ensemble routing or multi-stage routing strategies")
        
        # Dataset recommendations
        if len(df) > 0:
            recommendations.append("📊 Test on diverse datasets to validate routing effectiveness:")
            recommendations.append("   • MMLU: Multi-domain knowledge evaluation")
            recommendations.append("   • GSM8K: Mathematical reasoning")
            recommendations.append("   • HellaSwag: Commonsense reasoning")
            recommendations.append("   • RACE: Reading comprehension")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")
        
        print("\n" + "="*80)
        print("Router effectiveness analysis complete!")
        print("Generated visualization: router_effectiveness_analysis.png")
        print("="*80)
    
    def run_analysis(self):
        """Run the complete router effectiveness analysis."""
        print("🚀 Starting Router Effectiveness Analysis")
        print("="*60)
        
        # Load all router experiment data
        self.load_router_experiments()
        
        if not self.router_results:
            print("❌ No router experiment results found!")
            print("💡 Make sure you have run experiments with different router types:")
            print("   • DistilBERT Transformer Router")
            print("   • Random Router") 
            print("   • Oracle Router")
            return
        
        # Create comparison DataFrame
        df = self.create_router_comparison_dataframe()
        
        print(f"\n📊 Loaded {len(df)} router experiments")
        print(f"Router types found: {df['router_type'].unique().tolist()}")
        
        # Generate analysis
        if not df.empty:
            print("\n📈 Generating router effectiveness analysis...")
            self.analyze_router_effectiveness(df)
            self.generate_router_insights_report(df)
        else:
            print("❌ No valid router data found for analysis")

if __name__ == "__main__":
    analyzer = RouterEffectivenessAnalyzer()
    analyzer.run_analysis()