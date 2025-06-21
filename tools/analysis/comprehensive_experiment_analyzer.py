#!/usr/bin/env python3
"""
Comprehensive Experiment Analyzer
Analyzes and visualizes all baseline and compound AI results from the router experiments.
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

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveAnalyzer:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.baseline_results = {}
        self.compound_results = {}
        
    def load_all_results(self):
        """Load all baseline and compound AI results."""
        print("🔍 Loading all experiment results...")
        
        # Load baseline results
        baseline_dir = self.results_dir / "baselines"
        if baseline_dir.exists():
            for model_dir in baseline_dir.iterdir():
                if model_dir.is_dir():
                    result_files = list(model_dir.glob("*_full.json"))
                    if result_files:
                        try:
                            with open(result_files[0], 'r') as f:
                                data = json.load(f)
                                self.baseline_results[model_dir.name] = data
                                print(f"✅ Loaded baseline: {model_dir.name}")
                        except Exception as e:
                            print(f"❌ Failed to load {model_dir.name}: {e}")
        
        # Load compound AI results
        compound_dir = self.results_dir / "experiments" / "compound" / "transformer_router"
        if compound_dir.exists():
            for exp_dir in compound_dir.iterdir():
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
    
    def extract_metrics(self, data: Dict, experiment_type: str = "baseline") -> Dict:
        """Extract key metrics from experiment data."""
        if experiment_type == "baseline":
            cost_summary = data.get('cost_summary', {})
            summary_metrics = cost_summary.get('summary_metrics', {})
            
            return {
                'accuracy': summary_metrics.get('accuracy', 0),
                'avg_latency_ms': summary_metrics.get('avg_latency_ms', 0),
                'total_cost': summary_metrics.get('total_cost', 0),
                'cost_per_query': summary_metrics.get('cost_per_query', 0),
                'cost_per_correct': summary_metrics.get('cost_per_correct_answer', 0),
                'total_queries': cost_summary.get('evaluation_stats', {}).get('total_queries', 0)
            }
        else:  # compound
            cost_summary = data.get('cost_summary', {})
            summary_metrics = cost_summary.get('summary_metrics', {})
            eval_stats = cost_summary.get('evaluation_stats', {})
            
            return {
                'accuracy': summary_metrics.get('accuracy', 0),
                'avg_latency_ms': summary_metrics.get('avg_latency_ms', 0),
                'total_cost': summary_metrics.get('total_cost', 0),
                'cost_per_query': summary_metrics.get('cost_per_query', 0),
                'cost_per_correct': summary_metrics.get('cost_per_correct_answer', 0),
                'total_queries': eval_stats.get('total_queries', 0),
                'small_llm_usage': eval_stats.get('small_llm_usage', 0),
                'large_llm_usage': eval_stats.get('large_llm_usage', 0),
                'small_llm_ratio': eval_stats.get('small_llm_usage', 0) / max(eval_stats.get('total_queries', 1), 1)
            }
    
    def create_comparison_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create comparison DataFrames for analysis."""
        
        # Baseline DataFrame
        baseline_data = []
        for name, data in self.baseline_results.items():
            metrics = self.extract_metrics(data, "baseline")
            metrics['model'] = name
            metrics['type'] = 'baseline'
            baseline_data.append(metrics)
        
        baseline_df = pd.DataFrame(baseline_data)
        
        # Compound DataFrame  
        compound_data = []
        for name, data in self.compound_results.items():
            metrics = self.extract_metrics(data, "compound")
            metrics['experiment'] = name
            metrics['type'] = 'compound'
            
            # Parse experiment name to extract small and large models
            if 'gpt' in name:
                metrics['large_model'] = 'GPT-4o-mini'
            elif 'claude' in name:
                metrics['large_model'] = 'Claude Haiku'
            else:
                metrics['large_model'] = 'Unknown'
                
            if 'gemma' in name:
                metrics['small_model'] = 'Gemma2 2B'
            elif 'qwen' in name:
                metrics['small_model'] = 'Qwen2.5 1.5B'
            elif 'phi' in name:
                metrics['small_model'] = 'Phi-2'
            elif '3b' in name:
                metrics['small_model'] = 'Llama3.2 3B'
            elif '1b' in name:
                metrics['small_model'] = 'Llama3.2 1B'
            else:
                metrics['small_model'] = 'Unknown'
                
            compound_data.append(metrics)
        
        compound_df = pd.DataFrame(compound_data)
        
        return baseline_df, compound_df
    
    def plot_baseline_comparison(self, baseline_df: pd.DataFrame):
        """Create comprehensive baseline comparison plots."""
        if baseline_df.empty:
            print("No baseline data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Baseline Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Clean model names for display
        display_names = {
            'llama3_2_3b': 'Llama3.2 3B',
            'llama3_2_1b': 'Llama3.2 1B', 
            'gemma2_2b': 'Gemma2 2B',
            'qwen2_5_1_5b': 'Qwen2.5 1.5B',
            'phi': 'Phi-2',
            'openai_gpt4o_mini': 'GPT-4o-mini',
            'claude_haiku': 'Claude Haiku'
        }
        
        baseline_df['display_name'] = baseline_df['model'].map(display_names).fillna(baseline_df['model'])
        
        # Sort by accuracy for consistent ordering
        baseline_df_sorted = baseline_df.sort_values('accuracy', ascending=True)
        
        # 1. Accuracy Comparison
        axes[0,0].barh(baseline_df_sorted['display_name'], baseline_df_sorted['accuracy'] * 100)
        axes[0,0].set_xlabel('Accuracy (%)')
        axes[0,0].set_title('Model Accuracy')
        axes[0,0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(baseline_df_sorted['accuracy'] * 100):
            axes[0,0].text(v + 0.5, i, f'{v:.1f}%', va='center')
        
        # 2. Latency Comparison
        axes[0,1].barh(baseline_df_sorted['display_name'], baseline_df_sorted['avg_latency_ms'])
        axes[0,1].set_xlabel('Average Latency (ms)')
        axes[0,1].set_title('Model Latency')
        axes[0,1].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(baseline_df_sorted['avg_latency_ms']):
            axes[0,1].text(v + max(baseline_df_sorted['avg_latency_ms']) * 0.01, i, f'{v:.0f}ms', va='center')
        
        # 3. Cost per Query
        axes[1,0].barh(baseline_df_sorted['display_name'], baseline_df_sorted['cost_per_query'] * 1000000)
        axes[1,0].set_xlabel('Cost per Query (USD per million)')
        axes[1,0].set_title('Cost Efficiency')
        axes[1,0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(baseline_df_sorted['cost_per_query'] * 1000000):
            axes[1,0].text(v + max(baseline_df_sorted['cost_per_query'] * 1000000) * 0.01, i, f'${v:.1f}', va='center')
        
        # 4. Performance vs Cost Scatter
        baseline_df_sorted['perf_cost_ratio'] = baseline_df_sorted['accuracy'] / (baseline_df_sorted['cost_per_query'] + 1e-10)
        
        scatter = axes[1,1].scatter(baseline_df_sorted['cost_per_query'] * 1000000, 
                                  baseline_df_sorted['accuracy'] * 100,
                                  s=100, alpha=0.7)
        
        # Add model labels to scatter points
        for i, name in enumerate(baseline_df_sorted['display_name']):
            axes[1,1].annotate(name, 
                             (baseline_df_sorted['cost_per_query'].iloc[i] * 1000000, 
                              baseline_df_sorted['accuracy'].iloc[i] * 100),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1,1].set_xlabel('Cost per Query (USD per million)')
        axes[1,1].set_ylabel('Accuracy (%)')
        axes[1,1].set_title('Performance vs Cost')
        axes[1,1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis_baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_compound_analysis(self, compound_df: pd.DataFrame):
        """Create compound AI system analysis plots."""
        if compound_df.empty:
            print("No compound data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Compound AI System Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Accuracy by Small Model and Large Model
        if 'small_model' in compound_df.columns and 'large_model' in compound_df.columns:
            pivot_acc = compound_df.pivot_table(values='accuracy', 
                                               index='small_model', 
                                               columns='large_model', 
                                               aggfunc='mean')
            
            sns.heatmap(pivot_acc * 100, annot=True, fmt='.1f', cmap='RdYlGn', 
                       ax=axes[0,0], cbar_kws={'label': 'Accuracy (%)'})
            axes[0,0].set_title('Accuracy Heatmap by Model Combination')
            axes[0,0].set_xlabel('Large Model')
            axes[0,0].set_ylabel('Small Model')
        
        # 2. Small Model Usage Ratios
        if 'small_llm_ratio' in compound_df.columns:
            usage_data = compound_df.groupby('small_model')['small_llm_ratio'].mean()
            axes[0,1].bar(usage_data.index, usage_data.values * 100)
            axes[0,1].set_title('Small Model Usage Rate')
            axes[0,1].set_ylabel('Usage Rate (%)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, v in enumerate(usage_data.values * 100):
                axes[0,1].text(i, v + 1, f'{v:.1f}%', ha='center')
        
        # 3. Cost Efficiency Comparison
        cost_data = compound_df.groupby(['small_model', 'large_model']).agg({
            'cost_per_query': 'mean',
            'accuracy': 'mean'
        }).reset_index()
        
        cost_data['efficiency'] = cost_data['accuracy'] / (cost_data['cost_per_query'] + 1e-10)
        
        if not cost_data.empty:
            cost_data['combination'] = cost_data['small_model'] + ' → ' + cost_data['large_model']
            cost_sorted = cost_data.sort_values('efficiency', ascending=True)
            
            bars = axes[1,0].barh(cost_sorted['combination'], cost_sorted['efficiency'])
            axes[1,0].set_xlabel('Performance/Cost Ratio')
            axes[1,0].set_title('Cost Efficiency by Model Combination')
            axes[1,0].grid(axis='x', alpha=0.3)
        
        # 4. Latency vs Accuracy Trade-off
        axes[1,1].scatter(compound_df['avg_latency_ms'], compound_df['accuracy'] * 100, 
                         alpha=0.7, s=100)
        
        # Add labels for each point
        for i, exp in enumerate(compound_df['experiment']):
            axes[1,1].annotate(exp.replace('transformer_router_', '').replace('_', ' '), 
                             (compound_df['avg_latency_ms'].iloc[i], 
                              compound_df['accuracy'].iloc[i] * 100),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1,1].set_xlabel('Average Latency (ms)')
        axes[1,1].set_ylabel('Accuracy (%)')
        axes[1,1].set_title('Latency vs Accuracy Trade-off')
        axes[1,1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis_compound_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_routing_effectiveness_analysis(self, baseline_df: pd.DataFrame, compound_df: pd.DataFrame):
        """Analyze routing effectiveness by comparing compound vs baseline performance."""
        if baseline_df.empty or compound_df.empty:
            print("Insufficient data for routing analysis")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Routing Effectiveness Analysis', fontsize=16, fontweight='bold')
        
        # Create model mapping for comparison
        model_mapping = {
            'gemma2_2b': 'Gemma2 2B',
            'qwen2_5_1_5b': 'Qwen2.5 1.5B', 
            'phi': 'Phi-2',
            'llama3_2_3b': 'Llama3.2 3B',
            'llama3_2_1b': 'Llama3.2 1B'
        }
        
        # 1. Accuracy Improvement from Routing
        improvements = []
        for _, compound_row in compound_df.iterrows():
            small_model = compound_row['small_model']
            
            # Find corresponding baseline
            baseline_match = None
            for baseline_name, display_name in model_mapping.items():
                if display_name == small_model:
                    baseline_row = baseline_df[baseline_df['model'] == baseline_name]
                    if not baseline_row.empty:
                        baseline_match = baseline_row.iloc[0]
                        break
            
            if baseline_match is not None:
                improvement = (compound_row['accuracy'] - baseline_match['accuracy']) * 100
                improvements.append({
                    'combination': f"{small_model} → {compound_row['large_model']}",
                    'improvement': improvement,
                    'baseline_acc': baseline_match['accuracy'] * 100,
                    'compound_acc': compound_row['accuracy'] * 100,
                    'small_model': small_model
                })
        
        if improvements:
            imp_df = pd.DataFrame(improvements)
            imp_df_sorted = imp_df.sort_values('improvement', ascending=True)
            
            bars = axes[0,0].barh(imp_df_sorted['combination'], imp_df_sorted['improvement'])
            axes[0,0].set_xlabel('Accuracy Improvement (%)')
            axes[0,0].set_title('Accuracy Gain from Compound AI')
            axes[0,0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[0,0].grid(axis='x', alpha=0.3)
            
            # Color bars based on positive/negative improvement
            for bar, improvement in zip(bars, imp_df_sorted['improvement']):
                bar.set_color('green' if improvement > 0 else 'red')
                bar.set_alpha(0.7)
        
        # 2. Cost Impact Analysis
        cost_impacts = []
        for _, compound_row in compound_df.iterrows():
            small_model = compound_row['small_model']
            
            # Find corresponding baseline
            for baseline_name, display_name in model_mapping.items():
                if display_name == small_model:
                    baseline_row = baseline_df[baseline_df['model'] == baseline_name]
                    if not baseline_row.empty:
                        baseline_match = baseline_row.iloc[0]
                        cost_change = ((compound_row['cost_per_query'] - baseline_match['cost_per_query']) / 
                                     baseline_match['cost_per_query']) * 100
                        cost_impacts.append({
                            'combination': f"{small_model} → {compound_row['large_model']}",
                            'cost_change': cost_change,
                            'small_model': small_model
                        })
                        break
        
        if cost_impacts:
            cost_df = pd.DataFrame(cost_impacts)
            cost_df_sorted = cost_df.sort_values('cost_change', ascending=True)
            
            bars = axes[0,1].barh(cost_df_sorted['combination'], cost_df_sorted['cost_change'])
            axes[0,1].set_xlabel('Cost Change (%)')
            axes[0,1].set_title('Cost Impact of Compound AI')
            axes[0,1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[0,1].grid(axis='x', alpha=0.3)
            
            # Color bars
            for bar, change in zip(bars, cost_df_sorted['cost_change']):
                bar.set_color('red' if change > 0 else 'green')
                bar.set_alpha(0.7)
        
        # 3. Router Decision Analysis (Small Model Usage)
        if 'small_llm_ratio' in compound_df.columns:
            usage_by_small = compound_df.groupby('small_model')['small_llm_ratio'].mean()
            
            axes[1,0].pie(usage_by_small.values, labels=usage_by_small.index, autopct='%1.1f%%')
            axes[1,0].set_title('Average Small Model Usage by Model Type')
        
        # 4. Performance vs Cost Efficiency Matrix
        if improvements and cost_impacts:
            # Merge improvement and cost data
            analysis_data = []
            for imp, cost in zip(improvements, cost_impacts):
                if imp['combination'] == cost['combination']:
                    analysis_data.append({
                        'combination': imp['combination'],
                        'accuracy_gain': imp['improvement'],
                        'cost_change': cost['cost_change'],
                        'small_model': imp['small_model']
                    })
            
            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)
                
                scatter = axes[1,1].scatter(analysis_df['cost_change'], analysis_df['accuracy_gain'], 
                                          s=100, alpha=0.7)
                
                # Add quadrant lines
                axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1,1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Add labels
                for i, comb in enumerate(analysis_df['combination']):
                    axes[1,1].annotate(comb.replace(' → ', '\n→'), 
                                     (analysis_df['cost_change'].iloc[i], 
                                      analysis_df['accuracy_gain'].iloc[i]),
                                     xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                axes[1,1].set_xlabel('Cost Change (%)')
                axes[1,1].set_ylabel('Accuracy Gain (%)')
                axes[1,1].set_title('Routing Effectiveness Matrix')
                axes[1,1].grid(alpha=0.3)
                
                # Add quadrant labels
                axes[1,1].text(0.05, 0.95, 'Better & Cheaper', transform=axes[1,1].transAxes, 
                             fontsize=10, ha='left', va='top', 
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                axes[1,1].text(0.95, 0.95, 'Better & Expensive', transform=axes[1,1].transAxes, 
                             fontsize=10, ha='right', va='top',
                             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('analysis_routing_effectiveness.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, baseline_df: pd.DataFrame, compound_df: pd.DataFrame):
        """Generate a comprehensive text summary report."""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE EXPERIMENT ANALYSIS REPORT")
        print("="*80)
        
        # Baseline Analysis
        print("\n🎯 BASELINE MODEL PERFORMANCE")
        print("-" * 40)
        
        if not baseline_df.empty:
            # Clean model names
            display_names = {
                'llama3_2_3b': 'Llama3.2 3B',
                'llama3_2_1b': 'Llama3.2 1B', 
                'gemma2_2b': 'Gemma2 2B',
                'qwen2_5_1_5b': 'Qwen2.5 1.5B',
                'phi': 'Phi-2',
                'openai_gpt4o_mini': 'GPT-4o-mini',
                'claude_haiku': 'Claude Haiku'
            }
            
            baseline_df['display_name'] = baseline_df['model'].map(display_names).fillna(baseline_df['model'])
            baseline_sorted = baseline_df.sort_values('accuracy', ascending=False)
            
            print("Model Performance Ranking (by accuracy):")
            for i, row in baseline_sorted.iterrows():
                print(f"{i+1:2d}. {row['display_name']:15} | "
                      f"Acc: {row['accuracy']:6.1%} | "
                      f"Latency: {row['avg_latency_ms']:6.0f}ms | "
                      f"Cost/Query: ${row['cost_per_query']*1000000:6.2f}/M")
            
            # Best performers
            best_accuracy = baseline_sorted.iloc[0]
            best_cost = baseline_df.loc[baseline_df['cost_per_query'].idxmin()]
            best_latency = baseline_df.loc[baseline_df['avg_latency_ms'].idxmin()]
            
            print(f"\n🏆 BEST PERFORMERS:")
            print(f"Highest Accuracy: {best_accuracy['display_name']} ({best_accuracy['accuracy']:.1%})")
            print(f"Lowest Cost:      {best_cost['display_name']} (${best_cost['cost_per_query']*1000000:.2f}/M)")
            print(f"Fastest Response: {best_latency['display_name']} ({best_latency['avg_latency_ms']:.0f}ms)")
        
        # Compound AI Analysis
        print("\n🚀 COMPOUND AI SYSTEM PERFORMANCE")
        print("-" * 40)
        
        if not compound_df.empty:
            compound_sorted = compound_df.sort_values('accuracy', ascending=False)
            
            print("Compound System Ranking (by accuracy):")
            for i, row in compound_sorted.iterrows():
                small_usage = row.get('small_llm_ratio', 0) * 100
                print(f"{i+1:2d}. {row['small_model']:12} → {row['large_model']:12} | "
                      f"Acc: {row['accuracy']:6.1%} | "
                      f"Small Usage: {small_usage:5.1f}% | "
                      f"Cost: ${row['cost_per_query']*1000000:6.2f}/M")
            
            # Best compound system
            best_compound = compound_sorted.iloc[0]
            print(f"\n🏆 BEST COMPOUND SYSTEM:")
            print(f"{best_compound['small_model']} → {best_compound['large_model']}")
            print(f"Accuracy: {best_compound['accuracy']:.1%}")
            print(f"Small Model Usage: {best_compound.get('small_llm_ratio', 0)*100:.1f}%")
        
        # Routing Effectiveness
        if not baseline_df.empty and not compound_df.empty:
            print("\n⚡ ROUTING EFFECTIVENESS")
            print("-" * 40)
            
            # Calculate improvements
            model_mapping = {
                'gemma2_2b': 'Gemma2 2B',
                'qwen2_5_1_5b': 'Qwen2.5 1.5B', 
                'phi': 'Phi-2',
                'llama3_2_3b': 'Llama3.2 3B'
            }
            
            improvements = []
            for _, compound_row in compound_df.iterrows():
                small_model = compound_row['small_model']
                
                for baseline_name, display_name in model_mapping.items():
                    if display_name == small_model:
                        baseline_row = baseline_df[baseline_df['model'] == baseline_name]
                        if not baseline_row.empty:
                            baseline_match = baseline_row.iloc[0]
                            acc_improvement = (compound_row['accuracy'] - baseline_match['accuracy']) * 100
                            cost_change = ((compound_row['cost_per_query'] - baseline_match['cost_per_query']) / 
                                         baseline_match['cost_per_query']) * 100
                            
                            improvements.append({
                                'combination': f"{small_model} → {compound_row['large_model']}",
                                'acc_improvement': acc_improvement,
                                'cost_change': cost_change
                            })
                            break
            
            if improvements:
                print("Routing Impact Analysis:")
                for imp in sorted(improvements, key=lambda x: x['acc_improvement'], reverse=True):
                    acc_symbol = "📈" if imp['acc_improvement'] > 0 else "📉"
                    cost_symbol = "💰" if imp['cost_change'] > 0 else "💵"
                    print(f"{acc_symbol} {imp['combination']:25} | "
                          f"Acc: {imp['acc_improvement']:+6.1f}% | "
                          f"Cost: {cost_symbol} {imp['cost_change']:+6.1f}%")
        
        print("\n📈 KEY INSIGHTS & RECOMMENDATIONS")
        print("-" * 40)
        
        insights = []
        
        if not baseline_df.empty:
            # Cost efficiency insight
            baseline_df['efficiency'] = baseline_df['accuracy'] / (baseline_df['cost_per_query'] + 1e-10)
            most_efficient = baseline_df.loc[baseline_df['efficiency'].idxmax()]
            insights.append(f"Most cost-efficient baseline: {most_efficient.get('display_name', most_efficient['model'])}")
        
        if not compound_df.empty:
            # Best routing strategy
            best_compound = compound_df.loc[compound_df['accuracy'].idxmax()]
            insights.append(f"Best compound strategy: {best_compound['small_model']} → {best_compound['large_model']}")
            
            # Router usage patterns
            avg_small_usage = compound_df.get('small_llm_ratio', pd.Series([0])).mean() * 100
            if avg_small_usage > 0:
                insights.append(f"Average small model usage: {avg_small_usage:.1f}%")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        print("\n" + "="*80)
        print("Analysis complete! Check generated visualization files:")
        print("• analysis_baseline_comparison.png")
        print("• analysis_compound_comparison.png") 
        print("• analysis_routing_effectiveness.png")
        print("="*80)
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        print("🚀 Starting Comprehensive Experiment Analysis")
        print("="*60)
        
        # Load all data
        self.load_all_results()
        
        if not self.baseline_results and not self.compound_results:
            print("❌ No results found to analyze!")
            return
        
        # Create comparison DataFrames
        baseline_df, compound_df = self.create_comparison_dataframes()
        
        print(f"\n📊 Loaded {len(baseline_df)} baseline and {len(compound_df)} compound experiments")
        
        # Generate visualizations
        if not baseline_df.empty:
            print("\n📈 Generating baseline comparison plots...")
            self.plot_baseline_comparison(baseline_df)
        
        if not compound_df.empty:
            print("\n📈 Generating compound AI analysis plots...")
            self.plot_compound_analysis(compound_df)
        
        if not baseline_df.empty and not compound_df.empty:
            print("\n📈 Generating routing effectiveness analysis...")
            self.create_routing_effectiveness_analysis(baseline_df, compound_df)
        
        # Generate summary report
        self.generate_summary_report(baseline_df, compound_df)

if __name__ == "__main__":
    analyzer = ComprehensiveAnalyzer()
    analyzer.run_analysis()