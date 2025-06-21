#!/usr/bin/env python3
"""
Interactive experiment dashboard for Compound AI Router results.
Provides comprehensive analysis and comparison of different router approaches.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import argparse

class ExperimentDashboard:
    """Interactive dashboard for experiment analysis."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.experiments = {}
        self.load_experiments()
    
    def load_experiments(self):
        """Load all available experiment results."""
        
        # Load router type experiments
        router_types = ["llm", "transformer", "oracle", "random"]
        for router_type in router_types:
            router_dir = self.results_dir / "experiments" / "router_types" / router_type
            if router_dir.exists():
                for result_file in router_dir.glob("*.json"):
                    exp_name = f"{router_type}_{result_file.stem}"
                    try:
                        with open(result_file, 'r') as f:
                            self.experiments[exp_name] = {
                                "data": json.load(f),
                                "type": "router_comparison",
                                "router_type": router_type,
                                "file": result_file
                            }
                    except Exception as e:
                        print(f"⚠️  Could not load {result_file}: {e}")
        
        # Load baseline experiments
        baseline_dir = self.results_dir / "baselines" / "large_llm_only"
        if baseline_dir.exists():
            for result_file in baseline_dir.glob("*.json"):
                exp_name = f"baseline_{result_file.stem}"
                try:
                    with open(result_file, 'r') as f:
                        self.experiments[exp_name] = {
                            "data": json.load(f),
                            "type": "baseline",
                            "file": result_file
                        }
                except Exception as e:
                    print(f"⚠️  Could not load {result_file}: {e}")
    
    def show_experiment_summary(self):
        """Display summary of all loaded experiments."""
        
        print("📊 EXPERIMENT DASHBOARD")
        print("=" * 60)
        
        if not self.experiments:
            print("❌ No experiments found!")
            print("   Make sure results are in the correct directory structure:")
            print("   - results/experiments/router_types/{llm,transformer,oracle,random}/")
            print("   - results/baselines/large_llm_only/")
            return
        
        print(f"📁 Loaded {len(self.experiments)} experiments:\n")
        
        # Group by type
        by_type = {}
        for name, exp in self.experiments.items():
            exp_type = exp["type"]
            if exp_type not in by_type:
                by_type[exp_type] = []
            by_type[exp_type].append(name)
        
        for exp_type, exp_names in by_type.items():
            print(f"🔹 {exp_type.upper()}: {len(exp_names)} experiments")
            for name in sorted(exp_names)[:3]:  # Show first 3
                print(f"   • {name}")
            if len(exp_names) > 3:
                print(f"   • ... and {len(exp_names) - 3} more")
            print()
    
    def compare_router_performance(self):
        """Compare performance across different router types."""
        
        router_experiments = {
            name: exp for name, exp in self.experiments.items() 
            if exp["type"] == "router_comparison"
        }
        
        if not router_experiments:
            print("❌ No router experiments found for comparison!")
            return
        
        print("🎯 ROUTER PERFORMANCE COMPARISON")
        print("=" * 50)
        
        comparison_data = []
        
        for name, exp in router_experiments.items():
            data = exp["data"]
            if "summary" in data:
                summary = data["summary"]
                comparison_data.append({
                    "experiment": name,
                    "router_type": exp["router_type"],
                    "overall_accuracy": summary.get("overall_accuracy", 0),
                    "router_accuracy": summary.get("router_performance", {}).get("accuracy", 0),
                    "small_llm_usage": summary.get("routing_distribution", {}).get("small_llm", 0),
                    "large_llm_usage": summary.get("routing_distribution", {}).get("large_llm", 0),
                    "total_tokens": summary.get("large_llm_token_usage", {}).get("total_tokens", 0)
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
            
            # Generate performance plot
            self.plot_router_comparison(df)
        else:
            print("❌ No comparable data found!")
    
    def plot_router_comparison(self, df: pd.DataFrame):
        """Generate router comparison plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Router Performance Comparison', fontsize=16)
        
        # Overall accuracy
        axes[0, 0].bar(df['router_type'], df['overall_accuracy'])
        axes[0, 0].set_title('Overall System Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Router accuracy
        axes[0, 1].bar(df['router_type'], df['router_accuracy'])
        axes[0, 1].set_title('Router Classification Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Model usage distribution
        width = 0.35
        x = range(len(df))
        axes[1, 0].bar([i - width/2 for i in x], df['small_llm_usage'], width, label='Small LLM')
        axes[1, 0].bar([i + width/2 for i in x], df['large_llm_usage'], width, label='Large LLM')
        axes[1, 0].set_title('Model Usage Distribution')
        axes[1, 0].set_ylabel('Usage Count')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(df['router_type'], rotation=45)
        axes[1, 0].legend()
        
        # Token efficiency
        axes[1, 1].bar(df['router_type'], df['total_tokens'])
        axes[1, 1].set_title('Token Usage (Large LLM)')
        axes[1, 1].set_ylabel('Total Tokens')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.results_dir / "visualizations" / "dashboards" / "router_comparison.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        print(f"\n📊 Router comparison plot saved to: {output_path}")
        plt.show()
    
    def analyze_router_effectiveness(self, experiment_name: str):
        """Detailed analysis of router effectiveness for a specific experiment."""
        
        if experiment_name not in self.experiments:
            available = list(self.experiments.keys())
            print(f"❌ Experiment '{experiment_name}' not found!")
            print(f"Available experiments: {available[:5]}...")
            return
        
        exp = self.experiments[experiment_name]
        data = exp["data"]
        
        if "results" not in data:
            print(f"❌ No detailed results found in {experiment_name}")
            return
        
        results = data["results"]
        
        print(f"🔍 DETAILED ANALYSIS: {experiment_name}")
        print("=" * 50)
        
        # Calculate routing impact scenarios
        scenarios = {
            'correct_route_correct_answer': 0,
            'correct_route_wrong_answer': 0,
            'wrong_route_correct_answer': 0,
            'wrong_route_wrong_answer': 0,
        }
        
        for result in results:
            if "predicted_difficulty" in result and "true_difficulty" in result:
                router_correct = (result['predicted_difficulty'] == result['true_difficulty'])
                answer_correct = result.get('correct', False)
                
                if router_correct and answer_correct:
                    scenarios['correct_route_correct_answer'] += 1
                elif router_correct and not answer_correct:
                    scenarios['correct_route_wrong_answer'] += 1
                elif not router_correct and answer_correct:
                    scenarios['wrong_route_correct_answer'] += 1
                else:
                    scenarios['wrong_route_wrong_answer'] += 1
        
        total = len(results)
        print(f"📊 Routing Impact Analysis (out of {total} queries):")
        for scenario, count in scenarios.items():
            print(f"  {scenario}: {count} ({count/total:.1%})")
        
        # Calculate potential improvements
        wrong_routing = scenarios['wrong_route_correct_answer'] + scenarios['wrong_route_wrong_answer']
        print(f"\n🎯 Improvement Potential:")
        print(f"  Queries affected by wrong routing: {wrong_routing} ({wrong_routing/total:.1%})")
        
        return scenarios
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        
        print("📋 COMPREHENSIVE EXPERIMENT REPORT")
        print("=" * 60)
        
        self.show_experiment_summary()
        print("\n")
        self.compare_router_performance()
        
        # Find best performing experiments
        router_experiments = {
            name: exp for name, exp in self.experiments.items() 
            if exp["type"] == "router_comparison" and "summary" in exp["data"]
        }
        
        if router_experiments:
            print(f"\n🏆 TOP PERFORMING EXPERIMENTS:")
            performance_data = []
            for name, exp in router_experiments.items():
                summary = exp["data"]["summary"]
                performance_data.append({
                    "name": name,
                    "overall_accuracy": summary.get("overall_accuracy", 0),
                    "router_accuracy": summary.get("router_performance", {}).get("accuracy", 0)
                })
            
            # Sort by overall accuracy
            performance_data.sort(key=lambda x: x["overall_accuracy"], reverse=True)
            
            for i, exp in enumerate(performance_data[:3], 1):
                print(f"  {i}. {exp['name']}")
                print(f"     Overall: {exp['overall_accuracy']:.1%}, Router: {exp['router_accuracy']:.1%}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("  1. Focus on improving router accuracy for better cost efficiency")
        print("  2. Test with bigger LLM performance gaps to show router value")
        print("  3. Implement confidence-based routing strategies")
        print("  4. Analyze threshold sensitivity for optimal routing decisions")

def main():
    """Main CLI interface."""
    
    parser = argparse.ArgumentParser(description="Compound AI Router Experiment Dashboard")
    parser.add_argument("--results-dir", default="results", help="Results directory path")
    parser.add_argument("--command", choices=["summary", "compare", "analyze", "report"], 
                       default="report", help="Command to run")
    parser.add_argument("--experiment", help="Specific experiment to analyze")
    
    args = parser.parse_args()
    
    dashboard = ExperimentDashboard(args.results_dir)
    
    if args.command == "summary":
        dashboard.show_experiment_summary()
    elif args.command == "compare":
        dashboard.compare_router_performance()
    elif args.command == "analyze" and args.experiment:
        dashboard.analyze_router_effectiveness(args.experiment)
    elif args.command == "report":
        dashboard.generate_comprehensive_report()
    else:
        print("❌ Invalid command or missing experiment name for analysis")
        print("Usage examples:")
        print("  python experiment_dashboard.py --command summary")
        print("  python experiment_dashboard.py --command compare")
        print("  python experiment_dashboard.py --command analyze --experiment llm_phi2_router")
        print("  python experiment_dashboard.py --command report")

if __name__ == "__main__":
    main()