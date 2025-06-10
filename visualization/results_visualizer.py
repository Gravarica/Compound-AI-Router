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
    Visualizes and analyzes CompoundAI system test results with a focus on
    creating clear, scientific, and publication-quality plots.
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

        # Set a professional and accessible default style for scientific papers
        plt.style.use('seaborn-v0_8-ticks')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 14,
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black'
        })

        # Define a consistent, colorblind-friendly color palette
        self.palette = {
            'compound_primary': '#0072B2',  # Blue
            'compound_secondary': '#56B4E9', # Light Blue
            'baseline_primary': '#009E73',  # Green
            'baseline_secondary': '#81C784', # Light Green
            'highlight_positive': '#D55E00', # Vermillion
            'highlight_neutral': '#CC79A7', # Pink
            'false_positive': '#E69F00', # Orange
            'false_negative': '#8B4513'  # Brown
        }
        
        self.hatches = ['/', '\\', 'x', '+', '.', '*']


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

    def _add_bar_labels(self, ax: plt.Axes, fmt: Union[str, callable] = "{:.1f}", **kwargs):
        """Helper to add labels to bars cleanly."""
        for container in ax.containers:
            if callable(fmt):
                labels = [fmt(v) for v in container.datavalues]
            else:
                # Use printf-style formatting for compatibility with formats like '%d', '%.1f%%'
                labels = [fmt % v for v in container.datavalues]
            ax.bar_label(container, labels=labels, **kwargs)

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
        fig, ax = plt.subplots(figsize=(10, 7))
        self._plot_improved_accuracy_comparison(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_accuracy_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2a. Router Accuracy
        fig, ax = plt.subplots(figsize=(10, 7))
        self._plot_router_accuracy(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_router_accuracy.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2b. Router Errors
        fig, ax = plt.subplots(figsize=(10, 7))
        self._plot_router_errors(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_router_errors.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Model usage
        fig, ax = plt.subplots(figsize=(10, 7))
        self._plot_improved_model_usage(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_model_usage.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Latency comparison
        fig, ax = plt.subplots(figsize=(10, 7))
        self._plot_improved_latency_comparison(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_latency_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Cost savings
        fig, ax = plt.subplots(figsize=(10, 7))
        self._plot_improved_cost_savings(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_cost_savings.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Accuracy by difficulty
        fig, ax = plt.subplots(figsize=(12, 7))
        self._plot_improved_accuracy_by_difficulty(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_accuracy_by_difficulty.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 7. Token usage
        fig, ax = plt.subplots(figsize=(12, 7))
        self._plot_improved_token_usage(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_token_usage.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 8. Radar chart (multi-metric comparison)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
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
        width = 0.4

        compound_bars = ax.bar(x - width / 2, compound_acc, width, 
                               color=self.palette['compound_primary'], 
                               edgecolor='black',
                               label='Compound System')
        baseline_bars = ax.bar(x + width / 2, baseline_acc, width, 
                               color=self.palette['baseline_primary'], 
                               edgecolor='black',
                               label='Baseline')

        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

        self._add_bar_labels(ax, fmt='%.1f%%', padding=3, fontsize=12)

        max_height = max(max(compound_acc), max(baseline_acc))
        ax.set_ylim(0, max_height * 1.2)

        # Add a line for the difference
        ax2 = ax.twinx()
        ax2.plot(x, difference, color=self.palette['highlight_positive'], marker='D', markersize=6, linestyle='--', label='Accuracy Delta')
        ax2.set_ylabel('Accuracy Point Difference', color=self.palette['highlight_positive'])
        ax2.tick_params(axis='y', colors=self.palette['highlight_positive'])
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.1f}pp'))

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.7)
        sns.despine(ax=ax, right=False)
        sns.despine(ax=ax2, left=False)

    def _plot_router_accuracy(self, results: Dict[str, Any], ax):
        """Plot router accuracy."""
        test_names = []
        router_acc = []

        for name, result in results.items():
            if 'comparison' in result and 'router_performance' in result['comparison']:
                test_names.append(name)
                perf = result['comparison']['router_performance']
                router_acc.append(perf['accuracy'] * 100)

        if not test_names:
            ax.text(0.5, 0.5, "No Router Data", ha='center', va='center')
            return

        bars = ax.bar(test_names, router_acc, color=self.palette['highlight_neutral'], width=0.6, edgecolor='black')
        ax.set_ylabel('Accuracy')
        ax.set_title('Router Accuracy', pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        self._add_bar_labels(ax, fmt='%.1f%%', padding=3, fontsize=12)
        sns.despine(ax=ax)

    def _plot_router_errors(self, results: Dict[str, Any], ax):
        """Plot router errors (false positives and negatives)."""
        test_names = []
        false_pos = []
        false_neg = []

        for name, result in results.items():
            if 'comparison' in result and 'router_performance' in result['comparison']:
                test_names.append(name)
                perf = result['comparison']['router_performance']
                false_pos.append(perf['false_positives'])
                false_neg.append(perf['false_negatives'])

        if not test_names:
            ax.text(0.5, 0.5, "No Router Data", ha='center', va='center')
            return

        x = np.arange(len(test_names))
        width = 0.4

        ax.bar(x - width / 2, false_pos, width, 
               color=self.palette['false_positive'], edgecolor='black', 
               label='False Positives (Easy -> Hard)')
        ax.bar(x + width / 2, false_neg, width, 
               color=self.palette['false_negative'], edgecolor='black',
               label='False Negatives (Hard -> Easy)')

        ax.set_ylabel('Error Count')
        ax.set_title('Router Errors', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.legend(frameon=True, facecolor='white', framealpha=0.7)
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        self._add_bar_labels(ax, fmt='%d', padding=3, fontsize=12)
        max_val = max(max(false_pos) if false_pos else [0], max(false_neg) if false_neg else [0])
        ax.set_ylim(0, max_val * 1.25)
        sns.despine(ax=ax)

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
        bars1 = ax.bar(test_names, small_usage, 
                       color=self.palette['baseline_primary'], edgecolor='black',
                       label='Small LLM')
        bars2 = ax.bar(test_names, large_usage, bottom=small_usage, 
                       color=self.palette['compound_primary'], edgecolor='black',
                       label='Large LLM')

        ax.set_ylabel('Usage')
        ax.set_title('Model Usage Distribution', pad=20)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.legend(frameon=True, facecolor='white', framealpha=0.7)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

        # Add percentage labels directly on the bars
        for bar in bars1:
            h = bar.get_height()
            if h > 5:
                ax.text(bar.get_x() + bar.get_width() / 2., h / 2., f"{h:.1f}%",
                        ha='center', va='center', color='white', fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
        for bar in bars2:
            h = bar.get_height()
            h_bottom = bar.get_y()
            if h > 5:
                ax.text(bar.get_x() + bar.get_width() / 2., h_bottom + h / 2., f"{h:.1f}%",
                        ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        sns.despine(ax=ax)

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
        width = 0.4

        compound_bars = ax.bar(x - width / 2, compound_time, width, 
                               color=self.palette['compound_primary'], edgecolor='black',
                               label='Compound Latency')
        baseline_bars = ax.bar(x + width / 2, baseline_time, width, 
                               color=self.palette['baseline_primary'], edgecolor='black',
                               label='Baseline Latency')

        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Comparison and Speedup', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        self._add_bar_labels(ax, fmt='%.0f ms', padding=3, fontsize=12)

        ax.set_ylim(0, max(max(compound_time), max(baseline_time)) * 1.2)

        # Plot speedup on a secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(x, speedup, color=self.palette['highlight_positive'], marker='D', markersize=6, linestyle='--', label='Speedup Factor')
        ax2.set_ylabel('Speedup (X)', color=self.palette['highlight_positive'])
        ax2.tick_params(axis='y', colors=self.palette['highlight_positive'])
        ax2.set_ylim(bottom=1)

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.7)
        sns.despine(ax=ax, right=False)
        sns.despine(ax=ax2, left=False)

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
        width = 0.4

        compound_bars = ax.bar(x - width / 2, compound_cost, width, 
                               color=self.palette['compound_primary'], edgecolor='black',
                               label='Compound Cost')
        baseline_bars = ax.bar(x + width / 2, baseline_cost, width, 
                               color=self.palette['baseline_primary'], edgecolor='black',
                               label='Baseline Cost')

        ax.set_ylabel('Total Cost ($)')
        ax.set_title('API Cost Comparison and Savings', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        self._add_bar_labels(ax, fmt='$%.4f', padding=3, fontsize=12)
        ax.set_ylim(0, max(max(compound_cost), max(baseline_cost)) * 1.2)

        # Plot savings percentage on a secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(x, savings_pct, color=self.palette['highlight_positive'], marker='D', markersize=6, linestyle='--', label='Savings (%)')
        ax2.set_ylabel('Savings (%)', color=self.palette['highlight_positive'])
        ax2.tick_params(axis='y', colors=self.palette['highlight_positive'])
        ax2.set_ylim(0, 105)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.7)
        sns.despine(ax=ax, right=False)
        sns.despine(ax=ax2, left=False)

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
        ax.bar(x - width * 1.5, easy_compound, width, color=self.palette['compound_primary'], edgecolor='black', label='Compound (Easy)')
        ax.bar(x - width / 2, easy_baseline, width, color=self.palette['baseline_primary'], edgecolor='black', label='Baseline (Easy)')
        ax.bar(x + width / 2, hard_compound, width, color=self.palette['compound_secondary'], edgecolor='black', label='Compound (Hard)')
        ax.bar(x + width * 1.5, hard_baseline, width, color=self.palette['baseline_secondary'], edgecolor='black', label='Baseline (Hard)')

        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Question Difficulty', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.legend(frameon=True, facecolor='white', framealpha=0.7)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        self._add_bar_labels(ax, fmt='%.1f%%', padding=3, fontsize=10)
        sns.despine(ax=ax)

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
        ax.bar(x - width * 1.5, compound_input, width, color=self.palette['compound_primary'], edgecolor='black', label='Compound Input')
        ax.bar(x - width / 2, compound_output, width, color=self.palette['compound_secondary'], edgecolor='black', label='Compound Output')
        ax.bar(x + width / 2, baseline_input, width, color=self.palette['baseline_primary'], edgecolor='black', label='Baseline Input')
        ax.bar(x + width * 1.5, baseline_output, width, color=self.palette['baseline_secondary'], edgecolor='black', label='Baseline Output')

        ax.set_ylabel('Token Count')
        ax.set_title('Token Usage Comparison', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.legend(frameon=True, facecolor='white', framealpha=0.7)
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        self._add_bar_labels(ax, fmt='%d', padding=3, fontsize=10)

        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        sns.despine(ax=ax)

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
            metrics['Cost Savings'].append(max(0, comp['api_costs']['savings']['percentage'] / 100))
            metrics['Speedup'].append(min(max(0, comp['average_time_ms']['speedup'] -1) / 4, 1.0))  # Normalize speedup (1x to 5x scale)

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
        ax.set_xticklabels(categories, fontsize=16)

        # Draw y-axis labels (0-100%)
        ax.set_rlabel_position(180) # Move radial labels to avoid overlap
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], color="dimgray", size=12)
        ax.set_ylim(0, 1)

        # Plot each configuration as a separate line
        color_cycle = plt.cm.get_cmap('viridis', len(test_names))
        for i, name in enumerate(test_names):
            values = [metrics[cat][i] for cat in categories]
            values += values[:1]  # Close the loop

            # Plot the configuration line
            ax.plot(angles, values, color=color_cycle(i), linewidth=2.5, linestyle='solid', label=name)
            ax.fill(angles, values, color=color_cycle(i), alpha=0.2)

        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), frameon=True, facecolor='white', framealpha=0.7)

        # Add title
        ax.set_title('Multi-Metric Comparison', size=24, y=1.2)

    def create_summary_dashboard(self, results: Dict[str, Any], title: str = "CompoundAI System Performance Dashboard"):
        """
        Create a redesigned, comprehensive dashboard of results.

        Args:
            results: Dictionary of test results
            title: Dashboard title
        """
        # Create a figure with a more spacious layout (4 rows, 2 columns)
        fig = plt.figure(figsize=(22, 28))
        gs = GridSpec(4, 2, figure=fig, hspace=0.8, wspace=0.4)

        fig.suptitle(title, fontsize=40, y=0.98, weight='bold')

        # Plot accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_improved_accuracy_comparison(results, ax1)

        # Plot latency comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_improved_latency_comparison(results, ax2)

        # Plot cost savings
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_improved_cost_savings(results, ax3)

        # Plot model usage
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_improved_model_usage(results, ax4)

        # Plot router accuracy
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_router_accuracy(results, ax5)

        # Plot router errors
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_router_errors(results, ax6)

        # Plot accuracy by difficulty
        ax7 = fig.add_subplot(gs[3, 0])
        self._plot_improved_accuracy_by_difficulty(results, ax7)

        # Plot token usage
        ax8 = fig.add_subplot(gs[3, 1])
        self._plot_improved_token_usage(results, ax8)

        # Adjust layout
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save figure
        save_path = os.path.join(self.output_dir, "summary_dashboard.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary dashboard saved to {save_path}")

        # Save individual visualizations as they are still valuable
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
               autopct='%1.1f%%', startangle=90,
               colors=[self.palette['compound_primary'], self.palette['highlight_positive']],
               wedgeprops={'edgecolor': 'black', 'linewidth': 1},
               textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'white'})
        ax.set_title('Overall Accuracy', fontsize=18, pad=20)

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
                    xticklabels=['Predicted Easy', 'Predicted Hard'],
                    yticklabels=['True Easy', 'True Hard'], ax=ax, annot_kws={"size": 16}, cbar=True)
        ax.set_xlabel('Predicted Difficulty', fontsize=14)
        ax.set_ylabel('True Difficulty', fontsize=14)
        ax.set_title('Router Confusion Matrix', fontsize=18, pad=20)

    def _plot_latency_distribution(self, result: Dict[str, Any], ax):
        """Plot latency distribution"""
        if 'compound_results' not in result:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return

        # Extract latencies
        latencies = [r.get('total_time_ms', 0) for r in result['compound_results']]

        # Plot histogram
        sns.histplot(latencies, bins=25, ax=ax, color=self.palette['compound_primary'], kde=True)
        mean_latency = np.mean(latencies)
        ax.axvline(mean_latency, color=self.palette['highlight_positive'], linestyle='dashed', linewidth=2.5,
                   label=f'Mean: {mean_latency:.2f}ms')

        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Latency Distribution', fontsize=18, pad=20)
        ax.legend()
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        sns.despine(ax=ax)

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
        width = 0.4

        ax.bar(x - width / 2, compound_values, width, label='Compound', color=self.palette['compound_primary'], edgecolor='black')
        ax.bar(x + width / 2, baseline_values, width, label='Baseline', color=self.palette['baseline_primary'], edgecolor='black')

        ax.set_ylabel('Total Token Count')
        ax.set_title('Token Usage Breakdown', fontsize=18, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        self._add_bar_labels(ax, fmt='%d', padding=3)
        sns.despine(ax=ax)

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
        ax.bar(bin_labels, accuracies, color=self.palette['compound_primary'], alpha=0.8, width=0.7, edgecolor='black')

        # Add count labels
        for i, (acc, count) in enumerate(zip(accuracies, counts)):
            if count > 0:
                ax.text(i, acc + 2, f"n={count}", ha='center', fontsize=12)

        ax.set_xlabel('Query Length Category')
        ax.set_ylabel('Accuracy')
        ax.set_title('Performance by Query Length', fontsize=18, pad=20)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        sns.despine(ax=ax)

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

        ax.bar(x, correct, width, label='Correct', color=self.palette['compound_primary'], edgecolor='black')
        ax.bar(x, incorrect, width, bottom=correct, label='Incorrect', color=self.palette['highlight_positive'], edgecolor='black')

        # Add percentage labels
        for i in range(len(categories)):
            total = correct[i] + incorrect[i]
            if total > 0:
                pct_correct = correct[i] / total * 100
                if correct[i] / total > 0.05: # Only label if space
                    ax.text(i, correct[i] / 2, f"{pct_correct:.1f}%", ha='center', va='center', color='white', fontsize=12, fontweight='bold')

                if incorrect[i] > 0 and incorrect[i] / total > 0.05:
                    pct_incorrect = incorrect[i] / total * 100
                    ax.text(i, correct[i] + incorrect[i] / 2, f"{pct_incorrect:.1f}%", ha='center', va='center', color='black', fontsize=12, fontweight='bold')

        ax.set_ylabel('Count')
        ax.set_title('Error Analysis by Model and Question Difficulty', fontsize=18, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        sns.despine(ax=ax)

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
            colWidths=[0.3, 0.2, 0.2, 0.3]
        )

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2.5)

        # Color header row and bold text
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#f0f0f0')
            table[(0, i)].set_text_props(weight='bold')

        # Bold first column
        for i in range(1, len(table_data)):
            table[(i, 0)].set_text_props(weight='bold')

        ax.set_title('Key Metrics Summary', pad=40, fontsize=22)

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
        fig.suptitle(f"Comparison of Test Configurations", fontsize=24, y=1.02)

        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        # Save figure
        save_path = os.path.join(self.output_dir, "configuration_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Configuration comparison saved to {save_path}")

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
        df.plot(x='Configuration', y=['Compound System', 'Baseline'], kind='bar', ax=ax,
                color=[self.palette['compound_primary'], self.palette['baseline_primary']], 
                edgecolor='black',
                width=0.8)

        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison Across Configurations', fontsize=18, pad=20)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

        # Add accuracy improvement labels
        for i, (comp, base) in enumerate(zip(compound_acc, baseline_acc)):
            diff = comp - base
            ax.text(i, max(comp, base) + 2, f"{diff:+.1f}pp", ha='center', color=self.palette['highlight_positive'], fontweight='bold')

        ax.set_ylim(bottom=0)
        sns.despine(ax=ax)

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
        df.plot(x='Configuration', y=['Compound System', 'Baseline'], kind='bar', ax=ax,
                color=[self.palette['compound_primary'], self.palette['baseline_primary']], 
                edgecolor='black',
                width=0.8)

        # Plot speedup on secondary axis
        ax2 = ax.twinx()
        ax2.plot(ax.get_xticks(), df['Speedup'], 'o-', color=self.palette['highlight_positive'], label='Speedup Factor (X)', markersize=8)

        ax.set_ylabel('Latency (ms)')
        ax2.set_ylabel('Speedup Factor', color=self.palette['highlight_positive'])
        ax2.tick_params(axis='y', colors=self.palette['highlight_positive'])
        ax.set_title('Latency Comparison Across Configurations', fontsize=18, pad=20)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax2.set_ylim(bottom=0)
        sns.despine(ax=ax, right=False)
        sns.despine(ax=ax2, left=False)

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
        df.plot(x='Configuration', y=['Compound System', 'Baseline'], kind='bar', ax=ax,
                color=[self.palette['compound_primary'], self.palette['baseline_primary']],
                edgecolor='black', 
                width=0.8)

        # Plot savings percentage on secondary axis
        ax2 = ax.twinx()
        ax2.plot(ax.get_xticks(), df['Savings'], 'o-', color=self.palette['highlight_positive'], label='Savings (%)', markersize=8)

        ax.set_ylabel('Cost ($)')
        ax2.set_ylabel('Savings (%)', color=self.palette['highlight_positive'])
        ax2.tick_params(axis='y', colors=self.palette['highlight_positive'])
        ax.set_title('API Cost Comparison Across Configurations', fontsize=18, pad=20)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax2.set_ylim(0, 105)
        sns.despine(ax=ax, right=False)
        sns.despine(ax=ax2, left=False)

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
        x = np.arange(len(test_names))
        ax2.bar(x - width / 2, df['False Positives'], width,
                color=self.palette['false_positive'], edgecolor='black', label='False Positives')
        ax2.bar(x + width / 2, df['False Negatives'], width,
                color=self.palette['false_negative'], edgecolor='black', label='False Negatives')

        ax.set_ylabel('Accuracy')
        ax2.set_ylabel('Error Count')
        ax.set_title('Router Performance Across Configurations', fontsize=18, pad=20)
        ax.set_xticks(np.arange(len(test_names)))
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax.set_ylim(0, 105)
        ax2.set_ylim(bottom=0)
        sns.despine(ax=ax, right=False)
        sns.despine(ax=ax2, left=False)

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
                kind='bar', stacked=True, ax=ax,
                color=[self.palette['baseline_primary'], self.palette['compound_primary']],
                edgecolor='black',
                width=0.8)

        ax.set_ylabel('Usage')
        ax.set_title('Model Usage Distribution Across Configurations', fontsize=18, pad=20)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.legend(title='LLM Type')
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        sns.despine(ax=ax)