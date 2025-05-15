# src/visualization/results_visualizer.py

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
from matplotlib.gridspec import GridSpec


class ResultsVisualizer:
    """
    Visualizes and analyzes CompoundAI system test results.
    """

    def __init__(self, results_dir: str, output_dir: Optional[str] = None):
        """
        Initialize the visualizer.

        Args:
            results_dir: Directory containing test results
            output_dir: Directory to save visualizations (defaults to results_dir/visualizations)
        """
        self.results_dir = results_dir
        self.output_dir = output_dir or os.path.join(results_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)

        # Set default styling
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("talk")

    def load_results(self, pattern: str = "*_full.json") -> Dict[str, Any]:
        """
        Load test results from files matching the pattern.

        Args:
            pattern: File pattern to match

        Returns:
            Dictionary of test names mapped to results
        """
        results = {}

        # Find all files matching the pattern
        files = glob.glob(os.path.join(self.results_dir, pattern))

        for file_path in files:
            try:
                # Extract test name from file name
                test_name = os.path.basename(file_path).replace("_full.json", "")

                # Load results
                with open(file_path, 'r') as f:
                    results[test_name] = json.load(f)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return results

    def save_individual_visualizations(self, results: Dict[str, Any], prefix: str = "individual"):
        """
        Save each visualization as a separate file.

        Args:
            results: Dictionary of test results
            prefix: Prefix for filenames
        """
        # Create a directory for individual visualizations
        individual_dir = os.path.join(self.output_dir, "individual")
        os.makedirs(individual_dir, exist_ok=True)

        # 1. Accuracy comparison
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        self._plot_improved_accuracy_comparison(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_accuracy_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Router performance
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        self._plot_improved_router_performance(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_router_performance.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Model usage
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        self._plot_improved_model_usage(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_model_usage.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Latency comparison
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        self._plot_improved_latency_comparison(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_latency_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Cost savings
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        self._plot_improved_cost_savings(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_cost_savings.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Accuracy by difficulty
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        self._plot_improved_accuracy_by_difficulty(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_accuracy_by_difficulty.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 7. Token usage
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        self._plot_improved_token_usage(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_token_usage.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 8. Radar chart (multi-metric comparison)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')  # <-- This is the key line

        # Plot and save
        self._plot_radar_chart(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_radar_chart.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Individual visualizations saved to {individual_dir}")

    def _plot_improved_accuracy_comparison(self, results: Dict[str, Any], ax):
        """Plot improved accuracy comparison between compound and baseline"""
        test_names = []
        compound_acc = []
        baseline_acc = []
        difference = []

        for name, result in results.items():
            if 'comparison' in result and 'accuracy' in result['comparison']:
                test_names.append(name)
                compound_acc.append(result['comparison']['accuracy']['compound'] * 100)
                baseline_acc.append(result['comparison']['accuracy']['baseline'] * 100)
                difference.append(result['comparison']['accuracy']['difference'] * 100)

        x = np.arange(len(test_names))
        width = 0.35

        compound_bars = ax.bar(x - width / 2, compound_acc, width, color='#4CAF50', label='Compound System')
        baseline_bars = ax.bar(x + width / 2, baseline_acc, width, color='#2196F3', label='Baseline')

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')

        # Add direct labels instead of legend
        for i, bar in enumerate(compound_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                    f"Compound\n{compound_acc[i]:.1f}%",
                    ha='center', va='bottom', fontsize=9)

        for i, bar in enumerate(baseline_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                    f"Baseline\n{baseline_acc[i]:.1f}%",
                    ha='center', va='bottom', fontsize=9)

        # Add improvement text between bars
        for i in range(len(x)):
            mid_x = (compound_bars[i].get_x() + compound_bars[i].get_width() + baseline_bars[i].get_x()) / 2
            max_height = max(compound_bars[i].get_height(), baseline_bars[i].get_height())
            ax.text(mid_x, max_height + 5,
                    f"Δ: {difference[i]:+.1f}%",
                    ha='center', va='bottom', color='#E91E63', fontweight='bold')

        ax.set_ylim(0, max(max(compound_acc), max(baseline_acc)) * 1.25)

    def _plot_improved_router_performance(self, results: Dict[str, Any], ax):
        """Plot improved router performance metrics"""
        test_names = []
        router_acc = []
        false_pos = []
        false_neg = []

        for name, result in results.items():
            if 'comparison' in result and 'router_performance' in result['comparison']:
                test_names.append(name)
                perf = result['comparison']['router_performance']
                router_acc.append(perf['accuracy'] * 100)
                false_pos.append(perf['false_positives'])
                false_neg.append(perf['false_negatives'])

        # Create a figure with 2 subplots side by side
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121)  # Accuracy subplot
        ax2 = fig.add_subplot(122)  # Errors subplot

        # Plot accuracy as a bar chart
        accuracy_bars = ax1.bar(test_names, router_acc, color='#673AB7')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Router Accuracy')
        ax1.set_xticklabels(test_names, rotation=45, ha='right')

        # Add labels on top of bars
        for i, bar in enumerate(accuracy_bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 1,
                     f"{router_acc[i]:.1f}%", ha='center', va='bottom')

        # Plot false positives and negatives as a grouped bar chart
        x = np.arange(len(test_names))
        width = 0.35

        ax2.bar(x - width / 2, false_pos, width, color='#FF5722', label='False Positives')
        ax2.bar(x + width / 2, false_neg, width, color='#FFC107', label='False Negatives')

        ax2.set_ylabel('Count')
        ax2.set_title('Router Errors')
        ax2.set_xticks(x)
        ax2.set_xticklabels(test_names, rotation=45, ha='right')
        ax2.legend(loc='upper right')

        return fig

    def _plot_improved_model_usage(self, results: Dict[str, Any], ax):
        """Plot improved small vs large model usage"""
        test_names = []
        small_usage = []
        large_usage = []

        for name, result in results.items():
            if 'comparison' in result and 'resource_utilization' in result['comparison']:
                test_names.append(name)
                util = result['comparison']['resource_utilization']
                small_usage.append(util['small_llm_usage'] * 100)
                large_usage.append((1 - util['small_llm_usage']) * 100)

        # Create stacked bar chart
        bars1 = ax.bar(test_names, small_usage, color='#4CAF50', label='Small LLM')
        bars2 = ax.bar(test_names, large_usage, bottom=small_usage, color='#2196F3', label='Large LLM')

        ax.set_ylabel('Usage (%)')
        ax.set_title('Model Usage Distribution')
        ax.set_xticklabels(test_names, rotation=45, ha='right')

        # Add percentage labels directly on the bars
        for i, (small, large) in enumerate(zip(small_usage, large_usage)):
            # Add small LLM label only if big enough
            if small > 10:
                ax.text(i, small / 2, f"Small LLM\n{small:.1f}%",
                        ha='center', va='center', color='white', fontweight='bold')

            # Add large LLM label only if big enough
            if large > 10:
                ax.text(i, small + large / 2, f"Large LLM\n{large:.1f}%",
                        ha='center', va='center', color='white', fontweight='bold')

    def _plot_improved_latency_comparison(self, results: Dict[str, Any], ax):
        """Plot improved latency comparison with speedup values"""
        test_names = []
        compound_time = []
        baseline_time = []
        speedup = []

        for name, result in results.items():
            if 'comparison' in result and 'average_time_ms' in result['comparison']:
                test_names.append(name)
                times = result['comparison']['average_time_ms']
                compound_time.append(times['compound'])
                baseline_time.append(times['baseline'])
                speedup.append(times['speedup'])

        x = np.arange(len(test_names))
        width = 0.35

        compound_bars = ax.bar(x - width / 2, compound_time, width, color='#4CAF50', label='Compound')
        baseline_bars = ax.bar(x + width / 2, baseline_time, width, color='#2196F3', label='Baseline')

        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Comparison (with Speedup Factor)')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')

        # Add direct labels
        for i, bar in enumerate(compound_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 5,
                    f"Compound\n{compound_time[i]:.0f} ms",
                    ha='center', va='bottom', fontsize=9)

        for i, bar in enumerate(baseline_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 5,
                    f"Baseline\n{baseline_time[i]:.0f} ms",
                    ha='center', va='bottom', fontsize=9)

        # Add speedup text between bars with arrows
        for i in range(len(x)):
            mid_x = (compound_bars[i].get_x() + compound_bars[i].get_width() + baseline_bars[i].get_x()) / 2
            max_height = max(compound_bars[i].get_height(), baseline_bars[i].get_height())
            ax.text(mid_x, max_height + 20,
                    f"Speedup: {speedup[i]:.1f}x",
                    ha='center', va='bottom', color='#E91E63', fontweight='bold')

        # Set a reasonable y-limit to accommodate labels
        ax.set_ylim(0, max(max(compound_time), max(baseline_time)) * 1.3)

    def _plot_improved_cost_savings(self, results: Dict[str, Any], ax):
        """Plot improved cost savings with percentages"""
        test_names = []
        compound_cost = []
        baseline_cost = []
        savings_pct = []
        savings_amount = []

        for name, result in results.items():
            if 'comparison' in result and 'api_costs' in result['comparison']:
                test_names.append(name)
                costs = result['comparison']['api_costs']
                compound_cost.append(costs['compound']['total_cost'])
                baseline_cost.append(costs['baseline']['total_cost'])
                savings_pct.append(costs['savings']['percentage'])
                savings_amount.append(costs['savings']['amount'])

        x = np.arange(len(test_names))
        width = 0.35

        compound_bars = ax.bar(x - width / 2, compound_cost, width, color='#4CAF50', label='Compound')
        baseline_bars = ax.bar(x + width / 2, baseline_cost, width, color='#2196F3', label='Baseline')

        ax.set_ylabel('Cost ($)')
        ax.set_title('API Cost Comparison (with Savings)')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')

        # Add direct labels
        for i, bar in enumerate(compound_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f"Compound\n${compound_cost[i]:.4f}",
                    ha='center', va='bottom', fontsize=9)

        for i, bar in enumerate(baseline_bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f"Baseline\n${baseline_cost[i]:.4f}",
                    ha='center', va='bottom', fontsize=9)

        # Add savings text below the bars
        for i in range(len(x)):
            mid_x = (compound_bars[i].get_x() + compound_bars[i].get_width() + baseline_bars[i].get_x()) / 2
            ax.text(mid_x, 0.02,
                    f"Savings: ${savings_amount[i]:.4f} ({savings_pct[i]:.1f}%)",
                    ha='center', va='bottom', color='#E91E63', fontweight='bold')

        # Set a reasonable y-limit to accommodate labels
        ax.set_ylim(0, max(max(compound_cost), max(baseline_cost)) * 1.3)

    def _plot_improved_accuracy_by_difficulty(self, results: Dict[str, Any], ax):
        """Plot improved accuracy by difficulty"""
        test_names = []
        easy_compound = []
        easy_baseline = []
        hard_compound = []
        hard_baseline = []

        for name, result in results.items():
            if 'comparison' in result and 'accuracy_by_difficulty' in result['comparison']:
                test_names.append(name)
                acc = result['comparison']['accuracy_by_difficulty']
                easy_compound.append(acc['easy']['compound'] * 100)
                easy_baseline.append(acc['easy']['baseline'] * 100)
                hard_compound.append(acc['hard']['compound'] * 100)
                hard_baseline.append(acc['hard']['baseline'] * 100)

        # Set up positions for grouped bars
        x = np.arange(len(test_names))
        width = 0.2  # narrower bars

        # Plot the four types of bars
        ax.bar(x - width * 1.5, easy_compound, width, color='#4CAF50', label='Compound (Easy)')
        ax.bar(x - width / 2, easy_baseline, width, color='#81C784', label='Baseline (Easy)')
        ax.bar(x + width / 2, hard_compound, width, color='#2196F3', label='Compound (Hard)')
        ax.bar(x + width * 1.5, hard_baseline, width, color='#64B5F6', label='Baseline (Hard)')

        # Add direct labels instead of a complex legend
        # Create a custom box at the top for the legend
        textstr = '\n'.join((
            'Easy Questions:',
            '  ■ Compound System (dark green)',
            '  ■ Baseline (light green)',
            'Hard Questions:',
            '  ■ Compound System (dark blue)',
            '  ■ Baseline (light blue)'))

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy by Question Difficulty')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')

    def _plot_improved_token_usage(self, results: Dict[str, Any], ax):
        """Plot improved token usage comparison"""
        test_names = []
        compound_input = []
        compound_output = []
        baseline_input = []
        baseline_output = []

        for name, result in results.items():
            if 'comparison' in result and 'api_costs' in result['comparison']:
                test_names.append(name)

                compound_tokens = result['comparison']['api_costs']['compound']
                baseline_tokens = result['comparison']['api_costs']['baseline']

                compound_input.append(compound_tokens['total_input_tokens'])
                compound_output.append(compound_tokens['total_output_tokens'])
                baseline_input.append(baseline_tokens['total_input_tokens'])
                baseline_output.append(baseline_tokens['total_output_tokens'])

        x = np.arange(len(test_names))
        width = 0.2

        # Plot with improved colors and spacing
        compound_in_bars = ax.bar(x - width * 1.5, compound_input, width, color='#4CAF50',
                                  label='Compound Input')
        compound_out_bars = ax.bar(x - width / 2, compound_output, width, color='#66BB6A',
                                   label='Compound Output')
        baseline_in_bars = ax.bar(x + width / 2, baseline_input, width, color='#2196F3',
                                  label='Baseline Input')
        baseline_out_bars = ax.bar(x + width * 1.5, baseline_output, width, color='#64B5F6',
                                   label='Baseline Output')

        ax.set_ylabel('Token Count')
        ax.set_title('Token Usage Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')

        # Create a custom box for the legend
        textstr = '\n'.join((
            'Compound System:',
            '  ■ Input Tokens (dark green)',
            '  ■ Output Tokens (light green)',
            'Baseline:',
            '  ■ Input Tokens (dark blue)',
            '  ■ Output Tokens (light blue)'))

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        # Add total token count labels
        for i, result_name in enumerate(test_names):
            compound_total = compound_input[i] + compound_output[i]
            baseline_total = baseline_input[i] + baseline_output[i]
            savings_pct = (baseline_total - compound_total) / baseline_total * 100 if baseline_total > 0 else 0

            ax.text(i, max(compound_input[i], compound_output[i], baseline_input[i], baseline_output[i]) * 1.05,
                    f"Token Savings: {savings_pct:.1f}%",
                    ha='center', va='bottom', color='#E91E63', fontweight='bold')

    def _plot_radar_chart(self, results: Dict[str, Any], ax):
        """Create a radar chart comparing key metrics across configurations"""
        # Extract key metrics for radar chart
        test_names = []
        metrics = {
            'Accuracy': [],
            'Router Accuracy': [],
            'Small LLM Usage': [],
            'Cost Savings': [],
            'Speedup': []
        }

        for name, result in results.items():
            if 'comparison' not in result:
                continue

            test_names.append(name)
            comp = result['comparison']

            # Extract metrics (normalize to 0-1 range)
            metrics['Accuracy'].append(comp['accuracy']['compound'])
            metrics['Router Accuracy'].append(comp['router_performance']['accuracy'])
            metrics['Small LLM Usage'].append(comp['resource_utilization']['small_llm_usage'])
            metrics['Cost Savings'].append(comp['api_costs']['savings']['percentage'] / 100)
            metrics['Speedup'].append(min(comp['average_time_ms']['speedup'] / 5, 1.0))  # Normalize to max of 1

        if not test_names:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
            return

        # Set up radar chart
        categories = list(metrics.keys())
        N = len(categories)

        # Set angles for each metric (evenly distributed)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Set up subplot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw category labels at the angle of each category
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Draw y-axis labels (0-100%)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.set_ylim(0, 1)

        # Plot each configuration as a separate line
        for i, name in enumerate(test_names):
            values = [metrics[cat][i] for cat in categories]
            values += values[:1]  # Close the loop

            # Plot the configuration line
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=name)
            ax.fill(angles, values, alpha=0.1)

        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Add title
        plt.title('Multi-Metric Comparison', size=15, y=1.1)

    def create_summary_dashboard(self, results: Dict[str, Any], title: str = "CompoundAI System Results"):
        """
        Create a comprehensive dashboard of results.

        Args:
            results: Dictionary of test results
            title: Dashboard title
        """
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig)

        # Plot accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_improved_accuracy_comparison(results, ax1)

        # Plot router performance
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_improved_router_performance(results, ax2)

        # Plot model usage
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_improved_model_usage(results, ax3)

        # Plot latency comparison
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_improved_latency_comparison(results, ax4)

        # Plot cost savings
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_improved_cost_savings(results, ax5)

        # Plot accuracy by difficulty
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_improved_accuracy_by_difficulty(results, ax6)

        # Plot token usage
        ax7 = fig.add_subplot(gs[2, 0:2])
        self._plot_improved_token_usage(results, ax7)

        # Plot radar chart
        ax8 = fig.add_subplot(gs[2, 2], projection='polar')
        self._plot_radar_chart(results, ax8)

        # Set title
        fig.suptitle(title, fontsize=24, y=0.98)

        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        plt.savefig(os.path.join(self.output_dir, "dashboard.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Save individual visualizations
        self.save_individual_visualizations(results)

    def create_detailed_report(self, results: Dict[str, Any]):
        """
        Create a detailed breakdown of each test result.

        Args:
            results: Dictionary of test results
        """
        for test_name, result in results.items():
            if 'comparison' not in result:
                continue

            # Create directory for detailed reports
            detailed_dir = os.path.join(self.output_dir, f"details_{test_name}")
            os.makedirs(detailed_dir, exist_ok=True)

            # Create figure
            fig = plt.figure(figsize=(15, 20))
            gs = GridSpec(5, 2, figure=fig)

            # Accuracy pie chart
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_accuracy_pie(result, ax1)

            # Save individual visualization
            plt.figure(figsize=(8, 8))
            self._plot_accuracy_pie(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "accuracy_pie.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # Router confusion matrix
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_router_confusion(result, ax2)

            # Save individual visualization
            plt.figure(figsize=(8, 8))
            self._plot_router_confusion(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "router_confusion.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # Latency distribution
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_latency_distribution(result, ax3)

            # Save individual visualization
            plt.figure(figsize=(8, 8))
            self._plot_latency_distribution(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "latency_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # Token usage breakdown
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_token_breakdown(result, ax4)

            # Save individual visualization
            plt.figure(figsize=(8, 8))
            self._plot_token_breakdown(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "token_breakdown.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # Performance by question type
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_performance_by_question(result, ax5)

            # Save individual visualization
            plt.figure(figsize=(12, 6))
            self._plot_performance_by_question(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "performance_by_question.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # Error analysis
            ax6 = fig.add_subplot(gs[3, :])
            self._plot_error_analysis(result, ax6)

            # Save individual visualization
            plt.figure(figsize=(12, 6))
            self._plot_error_analysis(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "error_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # Add key metrics table
            ax7 = fig.add_subplot(gs[4, :])
            self._add_metrics_table(result, ax7)

            # Save individual visualization
            plt.figure(figsize=(12, 6))
            self._add_metrics_table(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "metrics_table.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # Set title
            fig.suptitle(f"Detailed Analysis: {test_name}", fontsize=24, y=0.98)

            # Adjust layout
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            # Save figure
            plt.savefig(os.path.join(self.output_dir, f"{test_name}_detailed.png"), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Detailed report for {test_name} saved to {detailed_dir}")

    def _plot_accuracy_pie(self, result: Dict[str, Any], ax):
        """Plot accuracy as pie chart"""
        if 'compound_results' not in result:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return

        # Count correct and incorrect
        correct = sum(1 for r in result['compound_results'] if r.get('correct'))
        incorrect = len(result['compound_results']) - correct

        # Create pie chart
        ax.pie([correct, incorrect], labels=['Correct', 'Incorrect'],
               autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
        ax.set_title('Accuracy')

    def _plot_router_confusion(self, result: Dict[str, Any], ax):
        """Plot router confusion matrix"""
        if 'compound_results' not in result:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return

        # Calculate confusion matrix
        true_easy = 0
        false_easy = 0
        true_hard = 0
        false_hard = 0

        for r in result['compound_results']:
            if r.get('true_difficulty') == 'easy':
                if r.get('predicted_difficulty') == 'easy':
                    true_easy += 1
                else:
                    false_hard += 1
            else:
                if r.get('predicted_difficulty') == 'hard':
                    true_hard += 1
                else:
                    false_easy += 1

        # Create confusion matrix
        cm = np.array([[true_easy, false_hard], [false_easy, true_hard]])

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Easy', 'Hard'],
                    yticklabels=['Easy', 'Hard'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Router Confusion Matrix')

    def _plot_latency_distribution(self, result: Dict[str, Any], ax):
        """Plot latency distribution"""
        if 'compound_results' not in result:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return

        # Extract latencies
        latencies = [r.get('total_time_ms', 0) for r in result['compound_results']]

        # Plot histogram
        ax.hist(latencies, bins=20, alpha=0.7)
        ax.axvline(np.mean(latencies), color='r', linestyle='dashed', linewidth=1,
                   label=f'Mean: {np.mean(latencies):.2f}ms')

        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Latency Distribution')
        ax.legend()

    def _plot_token_breakdown(self, result: Dict[str, Any], ax):
        """Plot token usage breakdown"""
        if 'comparison' not in result or 'api_costs' not in result['comparison']:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return

        # Extract token usage
        compound = result['comparison']['api_costs']['compound']
        baseline = result['comparison']['api_costs']['baseline']

        # Prepare data
        categories = ['Input Tokens', 'Output Tokens']
        compound_values = [compound['total_input_tokens'], compound['total_output_tokens']]
        baseline_values = [baseline['total_input_tokens'], baseline['total_output_tokens']]

        # Plot grouped bar chart
        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width / 2, compound_values, width, label='Compound')
        ax.bar(x + width / 2, baseline_values, width, label='Baseline')

        ax.set_ylabel('Token Count')
        ax.set_title('Token Usage Breakdown')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

    def _plot_performance_by_question(self, result: Dict[str, Any], ax):
        """Plot performance by question type"""
        if 'compound_results' not in result or not result['compound_results']:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return

        # Group results by query length as a proxy for complexity
        query_lengths = [len(r.get('query', '')) for r in result['compound_results']]
        bins = [0, 50, 100, 150, 200, float('inf')]
        bin_labels = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long']

        # Assign bin indices
        bin_indices = np.digitize(query_lengths, bins[1:])

        # Calculate accuracy by bin
        accuracies = []
        counts = []

        for i in range(len(bin_labels)):
            bin_results = [r for r, idx in zip(result['compound_results'], bin_indices) if idx == i]

            if bin_results:
                accuracy = sum(1 for r in bin_results if r.get('correct')) / len(bin_results)
                accuracies.append(accuracy * 100)
                counts.append(len(bin_results))
            else:
                accuracies.append(0)
                counts.append(0)

        # Plot bar chart
        ax.bar(bin_labels, accuracies, alpha=0.7)

        # Add count labels
        for i, (acc, count) in enumerate(zip(accuracies, counts)):
            if count > 0:
                ax.text(i, acc + 2, f"n={count}", ha='center')

        ax.set_xlabel('Query Length')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Performance by Query Length')
        ax.set_ylim(0, 105)

    def _plot_error_analysis(self, result: Dict[str, Any], ax):
        """Plot error analysis"""
        if 'compound_results' not in result:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return

        # Analyze errors by model and difficulty
        categories = ['Small LLM - Easy', 'Small LLM - Hard', 'Large LLM - Easy', 'Large LLM - Hard']
        correct = [0, 0, 0, 0]
        incorrect = [0, 0, 0, 0]

        for r in result['compound_results']:
            llm_type = 0 if r.get('chosen_llm') == 'small' else 2
            difficulty = 0 if r.get('true_difficulty') == 'easy' else 1
            idx = llm_type + difficulty

            if r.get('correct'):
                correct[idx] += 1
            else:
                incorrect[idx] += 1

        # Plot stacked bar chart
        x = np.arange(len(categories))
        width = 0.7

        ax.bar(x, correct, width, label='Correct')
        ax.bar(x, incorrect, width, bottom=correct, label='Incorrect')

        # Add percentage labels
        for i in range(len(categories)):
            total = correct[i] + incorrect[i]
            if total > 0:
                pct_correct = correct[i] / total * 100
                ax.text(i, correct[i] / 2, f"{pct_correct:.1f}%", ha='center', va='center')

                if incorrect[i] > 0:
                    pct_incorrect = incorrect[i] / total * 100
                    ax.text(i, correct[i] + incorrect[i] / 2, f"{pct_incorrect:.1f}%", ha='center', va='center')

        ax.set_ylabel('Count')
        ax.set_title('Error Analysis by Model and Question Difficulty')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()

    def _add_metrics_table(self, result: Dict[str, Any], ax):
        """Add key metrics table"""
        if 'comparison' not in result:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return

        # Hide axes
        ax.axis('off')

        # Extract key metrics
        comp = result['comparison']

        table_data = [
            ['Metric', 'Compound System', 'Baseline', 'Difference/Ratio'],
            ['Accuracy', f"{comp['accuracy']['compound']:.2%}", f"{comp['accuracy']['baseline']:.2%}",
             f"{comp['accuracy']['difference']:.2%}"],
            ['Avg. Latency', f"{comp['average_time_ms']['compound']:.2f}ms",
             f"{comp['average_time_ms']['baseline']:.2f}ms",
             f"{comp['average_time_ms']['speedup']:.2f}x"],
            ['API Cost', f"${comp['api_costs']['compound']['total_cost']:.4f}",
             f"${comp['api_costs']['baseline']['total_cost']:.4f}",
             f"${comp['api_costs']['savings']['amount']:.4f} ({comp['api_costs']['savings']['percentage']:.2f}%)"],
            ['Router Accuracy', f"{comp['router_performance']['accuracy']:.2%}", "", ""],
            ['Small LLM Usage', f"{comp['resource_utilization']['small_llm_usage']:.2%}", "", ""]
        ]

        # Create table
        table = ax.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            colWidths=[0.25, 0.25, 0.25, 0.25]
        )

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        # Color header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#f0f0f0')

        ax.set_title('Key Metrics Summary', pad=20)

    def create_comparison_visualization(self, results: Dict[str, Any], focus_metrics: List[str] = None):
        """
        Create visualization comparing different test configurations.

        Args:
            results: Dictionary of test results
            focus_metrics: List of metrics to focus on (default: accuracy, latency, cost savings)
        """
        if not focus_metrics:
            focus_metrics = ['accuracy', 'latency', 'cost_savings']

        # Check if we have enough data
        if len(results) < 2:
            print("Need at least 2 test results for comparison visualization")
            return

        comparison_dir = os.path.join(self.output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)

        # Create figure
        fig, axes = plt.subplots(len(focus_metrics), 1, figsize=(12, 5 * len(focus_metrics)))

        # If only one metric, axes is not a list
        if len(focus_metrics) == 1:
            axes = [axes]

        # Process each metric
        for i, metric in enumerate(focus_metrics):
            if metric == 'accuracy':
                self._plot_comparison_accuracy(results, axes[i])
            elif metric == 'latency':
                self._plot_comparison_latency(results, axes[i])
            elif metric == 'cost_savings':
                self._plot_comparison_cost_savings(results, axes[i])
            elif metric == 'router_accuracy':
                self._plot_comparison_router_accuracy(results, axes[i])
            elif metric == 'model_usage':
                self._plot_comparison_model_usage(results, axes[i])

            plt.figure(figsize=(10, 6))
            ax = plt.gca()

            if metric == 'accuracy':
                self._plot_comparison_accuracy(results, ax)
            elif metric == 'latency':
                self._plot_comparison_latency(results, ax)
            elif metric == 'cost_savings':
                self._plot_comparison_cost_savings(results, ax)
            elif metric == 'router_accuracy':
                self._plot_comparison_router_accuracy(results, ax)
            elif metric == 'model_usage':
                self._plot_comparison_model_usage(results, ax)

            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f"comparison_{metric}.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # Set title
        fig.suptitle(f"Comparison of Different Configurations", fontsize=16, y=0.98)

        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        plt.savefig(os.path.join(self.output_dir, "configuration_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_comparison_accuracy(self, results: Dict[str, Any], ax):
        """Plot accuracy comparison across configs"""
        test_names = []
        compound_acc = []
        baseline_acc = []

        for name, result in results.items():
            if 'comparison' in result and 'accuracy' in result['comparison']:
                test_names.append(name)
                compound_acc.append(result['comparison']['accuracy']['compound'] * 100)
                baseline_acc.append(result['comparison']['accuracy']['baseline'] * 100)

        # Create DataFrame
        df = pd.DataFrame({
            'Configuration': test_names,
            'Compound System': compound_acc,
            'Baseline': baseline_acc
        })

        # Plot
        df.plot(x='Configuration', y=['Compound System', 'Baseline'], kind='bar', ax=ax)

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Comparison Across Configurations')
        ax.set_xticklabels(test_names, rotation=45, ha='right')

        # Add accuracy improvement labels
        for i, (comp, base) in enumerate(zip(compound_acc, baseline_acc)):
            diff = comp - base
            ax.text(i, max(comp, base) + 1, f"{diff:+.1f}%", ha='center')

    def _plot_comparison_latency(self, results: Dict[str, Any], ax):
        """Plot latency comparison across configs"""
        test_names = []
        compound_latency = []
        baseline_latency = []
        speedup = []

        for name, result in results.items():
            if 'comparison' in result and 'average_time_ms' in result['comparison']:
                test_names.append(name)
                times = result['comparison']['average_time_ms']
                compound_latency.append(times['compound'])
                baseline_latency.append(times['baseline'])
                speedup.append(times['speedup'])

        # Create DataFrame
        df = pd.DataFrame({
            'Configuration': test_names,
            'Compound System': compound_latency,
            'Baseline': baseline_latency,
            'Speedup': speedup
        })

        # Plot latencies
        df.plot(x='Configuration', y=['Compound System', 'Baseline'], kind='bar', ax=ax)

        # Plot speedup on secondary axis
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(test_names)), df['Speedup'], 'o-', color='green', label='Speedup')

        ax.set_ylabel('Latency (ms)')
        ax2.set_ylabel('Speedup Factor')
        ax.set_title('Latency Comparison Across Configurations')
        ax.set_xticklabels(test_names, rotation=45, ha='right')

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    def _plot_comparison_cost_savings(self, results: Dict[str, Any], ax):
        """Plot cost savings comparison across configs"""
        test_names = []
        compound_cost = []
        baseline_cost = []
        savings_pct = []

        for name, result in results.items():
            if 'comparison' in result and 'api_costs' in result['comparison']:
                test_names.append(name)
                costs = result['comparison']['api_costs']
                compound_cost.append(costs['compound']['total_cost'])
                baseline_cost.append(costs['baseline']['total_cost'])
                savings_pct.append(costs['savings']['percentage'])

        # Create DataFrame
        df = pd.DataFrame({
            'Configuration': test_names,
            'Compound System': compound_cost,
            'Baseline': baseline_cost,
            'Savings': savings_pct
        })

        # Plot costs
        df.plot(x='Configuration', y=['Compound System', 'Baseline'], kind='bar', ax=ax)

        # Plot savings percentage on secondary axis
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(test_names)), df['Savings'], 'o-', color='green', label='Savings %')

        ax.set_ylabel('Cost ($)')
        ax2.set_ylabel('Savings (%)')
        ax.set_title('API Cost Comparison Across Configurations')
        ax.set_xticklabels(test_names, rotation=45, ha='right')

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    def _plot_comparison_router_accuracy(self, results: Dict[str, Any], ax):
        """Plot router accuracy comparison across configs"""
        test_names = []
        router_acc = []
        false_pos = []
        false_neg = []

        for name, result in results.items():
            if 'comparison' in result and 'router_performance' in result['comparison']:
                test_names.append(name)
                perf = result['comparison']['router_performance']
                router_acc.append(perf['accuracy'] * 100)
                false_pos.append(perf['false_positives'])
                false_neg.append(perf['false_negatives'])

        # Create DataFrame
        df = pd.DataFrame({
            'Configuration': test_names,
            'Router Accuracy': router_acc,
            'False Positives': false_pos,
            'False Negatives': false_neg
        })

        # Plot router accuracy
        ax.bar(np.arange(len(test_names)), df['Router Accuracy'], label='Router Accuracy (%)')

        # Plot error counts on secondary axis
        ax2 = ax.twinx()
        width = 0.2
        ax2.bar(np.arange(len(test_names)) - width / 2, df['False Positives'], width,
                color='red', label='False Positives')
        ax2.bar(np.arange(len(test_names)) + width / 2, df['False Negatives'], width,
                color='orange', label='False Negatives')

        ax.set_ylabel('Accuracy (%)')
        ax2.set_ylabel('Count')
        ax.set_title('Router Performance Across Configurations')
        ax.set_xticks(np.arange(len(test_names)))
        ax.set_xticklabels(test_names, rotation=45, ha='right')

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    def _plot_comparison_model_usage(self, results: Dict[str, Any], ax):
        """Plot model usage comparison across configs"""
        test_names = []
        small_usage = []
        large_usage = []

        for name, result in results.items():
            if 'comparison' in result and 'resource_utilization' in result['comparison']:
                test_names.append(name)
                util = result['comparison']['resource_utilization']
                small_usage.append(util['small_llm_usage'] * 100)
                large_usage.append((1 - util['small_llm_usage']) * 100)

        # Create DataFrame
        df = pd.DataFrame({
            'Configuration': test_names,
            'Small LLM': small_usage,
            'Large LLM': large_usage
        })

        # Create stacked bar chart
        df.plot(x='Configuration', y=['Small LLM', 'Large LLM'],
                kind='bar', stacked=True, ax=ax)

        ax.set_ylabel('Usage (%)')
        ax.set_title('Model Usage Distribution Across Configurations')
        ax.set_xticklabels(test_names, rotation=45, ha='right')