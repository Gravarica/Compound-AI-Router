import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import PercentFormatter, FuncFormatter
import matplotlib.patheffects as path_effects


class ResultsVisualizerV2:
    """
    Visualizes and analyzes CompoundAI system test results with scientific paper-quality plots.
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

        # Set scientific style for plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create custom color palette - more scientific and distinguishable
        self.palette = {
            'compound': '#1f77b4',  # Blue
            'baseline': '#d62728',  # Red
            'easy': '#2ca02c',      # Green
            'hard': '#ff7f0e',      # Orange
            'small': '#9467bd',     # Purple
            'large': '#8c564b',     # Brown
            'gain': '#17becf',      # Cyan
            'loss': '#e377c2'       # Pink
        }
        
        # Set consistent font sizes for all plots
        self.title_fontsize = 16
        self.axis_label_fontsize = 14
        self.tick_fontsize = 12
        self.legend_fontsize = 12
        self.annotation_fontsize = 10
        
        # Set figure dpi for high-quality output
        self.dpi = 300
        
        # Set global matplotlib parameters
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Computer Modern Roman']
        mpl.rcParams['text.usetex'] = False  # Set to True if LaTeX is available
        mpl.rcParams['axes.labelsize'] = self.axis_label_fontsize
        mpl.rcParams['axes.titlesize'] = self.title_fontsize
        mpl.rcParams['xtick.labelsize'] = self.tick_fontsize
        mpl.rcParams['ytick.labelsize'] = self.tick_fontsize
        mpl.rcParams['legend.fontsize'] = self.legend_fontsize
        mpl.rcParams['figure.titlesize'] = self.title_fontsize + 2

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
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        self._plot_scientific_accuracy_comparison(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_accuracy_comparison.png"), dpi=self.dpi, bbox_inches='tight')
        plt.savefig(os.path.join(individual_dir, f"{prefix}_accuracy_comparison.pdf"), bbox_inches='tight')
        plt.close()

        # 2. Router performance
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        self._plot_scientific_router_performance(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_router_performance.png"), dpi=self.dpi, bbox_inches='tight')
        plt.savefig(os.path.join(individual_dir, f"{prefix}_router_performance.pdf"), bbox_inches='tight')
        plt.close()

        # 3. Model usage
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        self._plot_scientific_model_usage(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_model_usage.png"), dpi=self.dpi, bbox_inches='tight')
        plt.savefig(os.path.join(individual_dir, f"{prefix}_model_usage.pdf"), bbox_inches='tight')
        plt.close()

        # 4. Latency comparison
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        self._plot_scientific_latency_comparison(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_latency_comparison.png"), dpi=self.dpi, bbox_inches='tight')
        plt.savefig(os.path.join(individual_dir, f"{prefix}_latency_comparison.pdf"), bbox_inches='tight')
        plt.close()

        # 5. Cost savings
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        self._plot_scientific_cost_savings(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_cost_savings.png"), dpi=self.dpi, bbox_inches='tight')
        plt.savefig(os.path.join(individual_dir, f"{prefix}_cost_savings.pdf"), bbox_inches='tight')
        plt.close()

        # 6. Accuracy by difficulty
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        self._plot_scientific_accuracy_by_difficulty(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_accuracy_by_difficulty.png"), dpi=self.dpi, bbox_inches='tight')
        plt.savefig(os.path.join(individual_dir, f"{prefix}_accuracy_by_difficulty.pdf"), bbox_inches='tight')
        plt.close()

        # 7. Token usage
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        self._plot_scientific_token_usage(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_token_usage.png"), dpi=self.dpi, bbox_inches='tight')
        plt.savefig(os.path.join(individual_dir, f"{prefix}_token_usage.pdf"), bbox_inches='tight')
        plt.close()

        # 8. Radar chart comparison
        fig = plt.figure(figsize=(10, 8), dpi=self.dpi)
        ax = fig.add_subplot(111, polar=True)
        self._plot_scientific_radar_chart(results, ax)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"{prefix}_radar_chart.png"), dpi=self.dpi, bbox_inches='tight')
        plt.savefig(os.path.join(individual_dir, f"{prefix}_radar_chart.pdf"), bbox_inches='tight')
        plt.close()

        print(f"Individual visualizations saved to {individual_dir}")

    def _plot_scientific_accuracy_comparison(self, results: Dict[str, Any], ax):
        """Plot improved scientific accuracy comparison between compound and baseline"""
        import numpy as np

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

        df = pd.DataFrame({
            'Test Configuration': test_names,
            'Compound AI System': compound_acc,
            'Baseline': baseline_acc,
            'Improvement': difference
        }).sort_values('Improvement', ascending=False)

        y_pos = np.arange(len(df))
        bar_height = 0.4

        ax.barh(y_pos - bar_height / 2, df['Compound AI System'], height=bar_height,
                color=self.palette['compound'], alpha=0.8, label='Compound AI System')
        ax.barh(y_pos + bar_height / 2, df['Baseline'], height=bar_height,
                color=self.palette['baseline'], alpha=0.8, label='Baseline')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['Test Configuration'])

        # Add improvement annotations
        for i, row in df.iterrows():
            improvement = row['Improvement']
            max_val = max(row['Compound AI System'], row['Baseline'])
            position = max_val + 2
            color = self.palette['gain'] if improvement > 0 else self.palette['loss']
            text = f"{improvement:+.1f}%"
            ax.annotate(text,
                        xy=(max_val, i),
                        xytext=(position, i),
                        fontsize=self.annotation_fontsize,
                        color=color,
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color=color))

        ax.set_xlabel('Accuracy (%)', fontsize=self.axis_label_fontsize)
        ax.set_title('Model Accuracy Comparison', fontsize=self.title_fontsize, fontweight='bold')
        ax.set_xlim(0, max(df['Compound AI System'].max(), df['Baseline'].max()) * 1.3)
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, facecolor='white', edgecolor='gray')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.axvline(0, color='black', linestyle='-', alpha=0.2)

    def _plot_scientific_router_performance(self, results: Dict[str, Any], ax):
        """Plot improved scientific router performance metrics"""
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

        df = pd.DataFrame({
            'Test Configuration': test_names,
            'Router Accuracy': router_acc,
            'False Positives': false_pos,
            'False Negatives': false_neg
        })

        # Sort by router accuracy 
        df = df.sort_values('Router Accuracy', ascending=False)

        # Create a more scientific twin-axis plot
        color1 = self.palette['compound']
        color2 = self.palette['baseline']

        # Plot router accuracy as bars
        ax.bar(df['Test Configuration'], df['Router Accuracy'], 
              color=color1, alpha=0.7, label='Router Accuracy')
        
        # Create twin axis for error counts
        ax2 = ax.twinx()
        
        # Plot errors as markers
        width = 0.2
        ax2.scatter(np.arange(len(df))-width, df['False Positives'], 
                   s=100, marker='o', color=self.palette['hard'], 
                   label='False Positives', edgecolors='white', linewidth=1, zorder=10)
        ax2.scatter(np.arange(len(df))+width, df['False Negatives'], 
                   s=100, marker='s', color=self.palette['easy'], 
                   label='False Negatives', edgecolors='white', linewidth=1, zorder=10)
        
        # Add lines between markers to show trends
        ax2.plot(np.arange(len(df))-width, df['False Positives'], 
                color=self.palette['hard'], linestyle='--', alpha=0.5)
        ax2.plot(np.arange(len(df))+width, df['False Negatives'], 
                color=self.palette['easy'], linestyle='--', alpha=0.5)

        # Add value labels to the bars
        for i, value in enumerate(df['Router Accuracy']):
            ax.text(i, value + 1, f"{value:.1f}%", 
                   ha='center', va='bottom', 
                   color=color1, fontweight='bold', fontsize=self.annotation_fontsize)
            
        # Add value labels to the markers
        for i, (fp, fn) in enumerate(zip(df['False Positives'], df['False Negatives'])):
            ax2.text(i-width, fp + 1, str(fp), 
                    ha='center', va='bottom',
                    color=self.palette['hard'], fontweight='bold', fontsize=self.annotation_fontsize)
            ax2.text(i+width, fn + 1, str(fn), 
                    ha='center', va='bottom',
                    color=self.palette['easy'], fontweight='bold', fontsize=self.annotation_fontsize)

        # Set labels and title
        ax.set_ylabel('Router Accuracy (%)', fontsize=self.axis_label_fontsize, color=color1)
        ax2.set_ylabel('Error Count', fontsize=self.axis_label_fontsize, color='black')
        ax.set_title('Router Performance Analysis', fontsize=self.title_fontsize, fontweight='bold')
        
        # Configure axis ticks
        ax.tick_params(axis='y', colors=color1)
        ax2.tick_params(axis='y', colors='black')
        
        # Set x-ticks
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['Test Configuration'], rotation=45, ha='right')
        
        # Adjust axis limits
        ax.set_ylim(0, 105)  # For percentage
        max_error = max(df['False Positives'].max(), df['False Negatives'].max())
        ax2.set_ylim(0, max_error * 1.3)  # For error counts
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
                 bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, 
                 framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Remove spines for a cleaner look
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

    def _plot_scientific_model_usage(self, results: Dict[str, Any], ax):
        """Plot improved scientific small vs large model usage"""
        test_names = []
        small_usage = []
        large_usage = []

        for name, result in results.items():
            if 'comparison' in result and 'resource_utilization' in result['comparison']:
                test_names.append(name)
                util = result['comparison']['resource_utilization']
                small_usage.append(util['small_llm_usage'] * 100)
                large_usage.append((1 - util['small_llm_usage']) * 100)

        # Create dataframe
        df = pd.DataFrame({
            'Test Configuration': test_names,
            'Small LLM': small_usage,
            'Large LLM': large_usage
        })
        
        # Sort by small LLM usage for better comparison
        df = df.sort_values('Small LLM', ascending=False)
        
        # Plot as a horizontal stacked bar for better readability
        df.plot(x='Test Configuration', y=['Small LLM', 'Large LLM'], 
                kind='barh', stacked=True, ax=ax, 
                color=[self.palette['small'], self.palette['large']], 
                width=0.7, alpha=0.8)
        
        # Add percentage labels directly on the bars
        for i, row in enumerate(df.itertuples()):
            # Small LLM label
            if row._2 > 10:  # Only add if segment is large enough
                small_text = ax.text(row._2/2, i, f"{row._2:.1f}%", 
                           ha='center', va='center', 
                           color='white', fontweight='bold',
                           fontsize=self.annotation_fontsize)
                small_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black', alpha=0.2)])
            
            # Large LLM label
            if row._3 > 10:  # Only add if segment is large enough
                large_text = ax.text(row._2 + row._3/2, i, f"{row._3:.1f}%", 
                           ha='center', va='center', 
                           color='white', fontweight='bold',
                           fontsize=self.annotation_fontsize)
                large_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black', alpha=0.2)])

        # Configure axis and labels
        ax.set_xlabel('Percentage of Queries (%)', fontsize=self.axis_label_fontsize)
        ax.set_title('Model Usage Distribution', fontsize=self.title_fontsize, fontweight='bold')
        
        # Set x-axis to percentage format
        ax.xaxis.set_major_formatter(PercentFormatter(100))
        
        # Add grid for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Improve legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['Small LLM (Efficient)', 'Large LLM (Powerful)'], 
                 loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                 frameon=True, framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a note about efficiency
        efficiency_note = "Higher Small LLM usage indicates better routing efficiency"
        ax.text(0.5, -0.2, efficiency_note, 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')

    def _plot_scientific_latency_comparison(self, results: Dict[str, Any], ax):
        """Plot improved scientific latency comparison with speedup values"""
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

        # Create dataframe
        df = pd.DataFrame({
            'Test Configuration': test_names,
            'Compound AI System': compound_time,
            'Baseline': baseline_time,
            'Speedup': speedup
        })
        
        # Sort by speedup for better visualization
        df = df.sort_values('Speedup', ascending=False)
        
        # Plot latency data
        width = 0.4
        x = np.arange(len(df))
        
        # Create bars
        baseline_bars = ax.bar(x - width/2, df['Baseline'], width, 
                              color=self.palette['baseline'], alpha=0.8, label='Baseline')
        compound_bars = ax.bar(x + width/2, df['Compound AI System'], width,
                              color=self.palette['compound'], alpha=0.8, label='Compound AI System')
        
        # Create twin axis for speedup
        ax2 = ax.twinx()
        
        # Plot speedup as a line with markers
        speedup_line = ax2.plot(x, df['Speedup'], 'o-', color=self.palette['gain'], 
                               linewidth=2, label='Speedup Factor', marker='D', markersize=8)
        
        # Add a horizontal line at y=1 for speedup reference
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        # Add value annotations to each bar
        for i, bar in enumerate(baseline_bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f"{df['Baseline'].iloc[i]:.0f}ms", 
                   ha='center', va='bottom', 
                   color=self.palette['baseline'], fontweight='bold',
                   fontsize=self.annotation_fontsize)
                   
        for i, bar in enumerate(compound_bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                   f"{df['Compound AI System'].iloc[i]:.0f}ms", 
                   ha='center', va='bottom',
                   color=self.palette['compound'], fontweight='bold',
                   fontsize=self.annotation_fontsize)
                   
        # Add speedup values
        for i, speedup_val in enumerate(df['Speedup']):
            ax2.text(i, speedup_val + 0.1, f"{speedup_val:.1f}×",
                    ha='center', va='bottom',
                    color=self.palette['gain'], fontweight='bold',
                    fontsize=self.annotation_fontsize)
        
        # Configure axes
        ax.set_ylabel('Latency (ms)', fontsize=self.axis_label_fontsize)
        ax2.set_ylabel('Speedup Factor', fontsize=self.axis_label_fontsize, color=self.palette['gain'])
        ax.set_title('Latency Comparison and Speedup', fontsize=self.title_fontsize, fontweight='bold')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(df['Test Configuration'], rotation=45, ha='right')
        
        # Color the y-axis label on the right
        ax2.tick_params(axis='y', colors=self.palette['gain'])
        
        # Set axis limits
        max_latency = max(df['Baseline'].max(), df['Compound AI System'].max())
        ax.set_ylim(0, max_latency * 1.3)
        ax2.set_ylim(0, max(df['Speedup'].max() * 1.2, 3.0))  # Ensure we at least see up to 3x
        
        # Create a unified legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                 bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True,
                 framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Remove top spine for cleaner look
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        # Add explanatory note
        note = "Higher speedup factor indicates better performance"
        ax.text(0.5, -0.2, note,
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')

    def _plot_scientific_cost_savings(self, results: Dict[str, Any], ax):
        """Plot improved scientific cost savings with percentages"""
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

        # Create dataframe
        df = pd.DataFrame({
            'Test Configuration': test_names,
            'Compound AI System': compound_cost,
            'Baseline': baseline_cost,
            'Savings (%)': savings_pct,
            'Savings ($)': savings_amount
        })
        
        # Sort by savings percentage for better visualization
        df = df.sort_values('Savings (%)', ascending=False)
        
        # Create a more scientific horizontal bar chart
        df.plot(x='Test Configuration', y=['Baseline', 'Compound AI System'], 
               kind='barh', ax=ax, color=[self.palette['baseline'], self.palette['compound']], 
               width=0.7, alpha=0.8)
        
        # Add annotations for cost values
        for i, row in enumerate(df.itertuples()):
            # Baseline cost

            baseline_val = row.Baseline
            ax.text(baseline_val / 2, i - 0.17, f"${baseline_val:.4f}",
                   ha='center', va='center', 
                   color='white', fontweight='bold',
                   fontsize=self.annotation_fontsize)
            
            # Compound cost
            compound_val = row._2
            ax.text(compound_val / 2, i + 0.17, f"${compound_val:.4f}",
                   ha='center', va='center', 
                   color='white', fontweight='bold',
                   fontsize=self.annotation_fontsize)
            
            # Savings amount and percentage to the right

            amount = row._5
            percentage = row._4
            ax.text(max(baseline_val, compound_val) * 1.05, i,
                   f"${amount:.4f} ({percentage:.1f}%)",
                   ha='left', va='center', 
                   color=self.palette['gain'], fontweight='bold',
                   fontsize=self.annotation_fontsize)
        
        # Configure axes
        ax.set_xlabel('Cost ($)', fontsize=self.axis_label_fontsize)
        ax.set_title('API Cost Comparison and Savings', fontsize=self.title_fontsize, fontweight='bold')
        
        # Add grid for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Improve legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                 frameon=True, framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set axis limits to accommodate annotations
        max_cost = max(df['Baseline'].max(), df['Compound AI System'].max())
        ax.set_xlim(0, max_cost * 1.4)
        
        # Add explanatory note
        note = "Higher percentage indicates greater cost efficiency"
        ax.text(0.5, -0.2, note,
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')

    def _plot_scientific_accuracy_by_difficulty(self, results: Dict[str, Any], ax):
        """Plot improved scientific accuracy by difficulty"""
        # Collect data
        all_data = []
        
        for name, result in results.items():
            if 'comparison' in result and 'accuracy_by_difficulty' in result['comparison']:
                acc = result['comparison']['accuracy_by_difficulty']
                
                # Add data for Compound AI System
                all_data.append({
                    'Configuration': name,
                    'System': 'Compound AI System',
                    'Difficulty': 'Easy',
                    'Accuracy': acc['easy']['compound'] * 100
                })
                all_data.append({
                    'Configuration': name,
                    'System': 'Compound AI System',
                    'Difficulty': 'Hard',
                    'Accuracy': acc['hard']['compound'] * 100
                })
                
                # Add data for baseline
                all_data.append({
                    'Configuration': name,
                    'System': 'Baseline',
                    'Difficulty': 'Easy',
                    'Accuracy': acc['easy']['baseline'] * 100
                })
                all_data.append({
                    'Configuration': name,
                    'System': 'Baseline',
                    'Difficulty': 'Hard',
                    'Accuracy': acc['hard']['baseline'] * 100
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Plot grouped bar chart using seaborn for better scientific look
        # First calculate the positions
        systems = df['System'].unique()
        difficulties = df['Difficulty'].unique()
        positions = []
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Get number of configurations
        configs = df['Configuration'].unique()
        n_configs = len(configs)
        
        # Define width and spacing
        bar_width = 0.18
        group_width = bar_width * 4 + 0.1
        
        # Define positions for each group
        positions = np.arange(n_configs) * (group_width + 0.3)
        
        # Define colors for each category
        colors = {
            ('Compound AI System', 'Easy'): self.palette['compound'],
            ('Compound AI System', 'Hard'): '#66b3ff',  # Light blue
            ('Baseline', 'Easy'): self.palette['baseline'],
            ('Baseline', 'Hard'): '#ff9999'  # Light red
        }
        
        # Plot each group
        bar_positions = {}
        for i, difficulty in enumerate(difficulties):
            for j, system in enumerate(systems):
                pos = positions + (j*2 + i) * bar_width
                subset = df[(df['System'] == system) & (df['Difficulty'] == difficulty)]
                
                # Save positions for annotations
                for config in configs:
                    idx = np.where(configs == config)[0][0]
                    bar_positions[(config, system, difficulty)] = pos[idx]
                
                # Create bars
                bars = ax.bar(pos, subset['Accuracy'], 
                             width=bar_width, 
                             color=colors[(system, difficulty)],
                             alpha=0.8,
                             label=f"{system} - {difficulty}")
                
                # Add value annotations to bars
                for idx, (_, row) in enumerate(subset.iterrows()):
                    ax.text(pos[idx], row['Accuracy'] + 1, 
                           f"{row['Accuracy']:.1f}%",
                           ha='center', va='bottom',
                           fontsize=self.annotation_fontsize-1,  # Smaller to avoid overlap
                           color=colors[(system, difficulty)],
                           fontweight='bold')
        
        # Set x-ticks at group centers
        ax.set_xticks(positions + group_width/2 - bar_width/2)
        ax.set_xticklabels(configs, rotation=0)
        
        # Set labels and title
        ax.set_ylabel('Accuracy (%)', fontsize=self.axis_label_fontsize)
        ax.set_title('Accuracy by Question Difficulty', fontsize=self.title_fontsize, fontweight='bold')
        
        # Create custom legend with clearer labels
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[('Compound AI System', 'Easy')], label='Compound AI System - Easy Questions', alpha=0.8),
            Patch(facecolor=colors[('Compound AI System', 'Hard')], label='Compound AI System - Hard Questions', alpha=0.8),
            Patch(facecolor=colors[('Baseline', 'Easy')], label='Baseline - Easy Questions', alpha=0.8),
            Patch(facecolor=colors[('Baseline', 'Hard')], label='Baseline - Hard Questions', alpha=0.8)
        ]
        ax.legend(handles=legend_elements, loc='upper center', 
                 bbox_to_anchor=(0.5, -0.13), ncol=2,
                 frameon=True, framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Set y-axis limits
        ax.set_ylim(0, 105)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add explanatory note
        ax.text(0.5, -0.18, "Higher accuracy on hard questions indicates better reasoning capabilities",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')

    def _plot_scientific_token_usage(self, results: Dict[str, Any], ax):
        """Plot improved scientific token usage comparison"""
        test_names = []
        compound_input = []
        compound_output = []
        baseline_input = []
        baseline_output = []
        token_savings_pct = []

        for name, result in results.items():
            if 'comparison' in result and 'api_costs' in result['comparison']:
                test_names.append(name)

                compound_tokens = result['comparison']['api_costs']['compound']
                baseline_tokens = result['comparison']['api_costs']['baseline']

                compound_input.append(compound_tokens['total_input_tokens'])
                compound_output.append(compound_tokens['total_output_tokens'])
                baseline_input.append(baseline_tokens['total_input_tokens'])
                baseline_output.append(baseline_tokens['total_output_tokens'])
                
                # Calculate total token savings percentage
                compound_total = compound_tokens['total_input_tokens'] + compound_tokens['total_output_tokens']
                baseline_total = baseline_tokens['total_input_tokens'] + baseline_tokens['total_output_tokens']
                if baseline_total > 0:
                    savings = (baseline_total - compound_total) / baseline_total * 100
                else:
                    savings = 0
                token_savings_pct.append(savings)

        # Create dataframe for easier manipulation and plotting
        df = pd.DataFrame({
            'Test Configuration': test_names,
            'Compound Input': compound_input,
            'Compound Output': compound_output,
            'Baseline Input': baseline_input,
            'Baseline Output': baseline_output,
            'Token Savings (%)': token_savings_pct
        })
        
        # Calculate total tokens for sorting
        df['Compound Total'] = df['Compound Input'] + df['Compound Output']
        df['Baseline Total'] = df['Baseline Input'] + df['Baseline Output']
        
        # Sort by token savings
        df = df.sort_values('Token Savings (%)', ascending=False)
        
        # Melt the dataframe for easier grouped bar plotting
        plot_data = pd.melt(df, 
                           id_vars=['Test Configuration', 'Token Savings (%)'],
                           value_vars=['Compound Input', 'Compound Output', 
                                      'Baseline Input', 'Baseline Output'],
                           var_name='Token Type', value_name='Count')
        
        # Add a System column based on Token Type
        plot_data['System'] = plot_data['Token Type'].apply(
            lambda x: 'Compound' if 'Compound' in x else 'Baseline')
        plot_data['Token Direction'] = plot_data['Token Type'].apply(
            lambda x: 'Input' if 'Input' in x else 'Output')
        
        # Define colors for different token types
        colors = {
            'Compound Input': '#7eb0d5',  # Light blue
            'Compound Output': '#2d7bb6',  # Dark blue
            'Baseline Input': '#ffb28f',   # Light red
            'Baseline Output': '#d65c5c'   # Dark red
        }
        
        # Define positions for grouped bars
        configs = df['Test Configuration'].unique()
        n_configs = len(configs)
        
        bar_width = 0.2
        group_width = bar_width * 4 + 0.1
        
        # Define positions for each group
        positions = np.arange(n_configs) * (group_width + 0.3)
        
        # Plot each token type
        token_types = ['Compound Input', 'Compound Output', 'Baseline Input', 'Baseline Output']
        
        for i, token_type in enumerate(token_types):
            pos = positions + i * bar_width
            subset = plot_data[plot_data['Token Type'] == token_type]
            
            # Match order of bars with positions
            subset = subset.set_index('Test Configuration').loc[configs].reset_index()
            
            # Create bars
            bars = ax.bar(pos, subset['Count'], 
                         width=bar_width, 
                         color=colors[token_type],
                         alpha=0.8,
                         label=token_type)
            
            # Add value annotations to significant bars
            for idx, count in enumerate(subset['Count']):
                if count > max(df['Baseline Total'].max(), df['Compound Total'].max()) * 0.05:  # Only label significant bars
                    ax.text(pos[idx], count + 50, 
                           f"{count:.0f}",
                           ha='center', va='bottom',
                           fontsize=self.annotation_fontsize-2,  # Smaller font for readability
                           color=colors[token_type],
                           fontweight='bold',
                           rotation=90)
        
        # Set x-ticks at group centers
        ax.set_xticks(positions + group_width/2 - bar_width/2)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        
        # Create twin axis for token savings
        ax2 = ax.twinx()
        ax2.plot(positions + group_width/2 - bar_width/2, df['Token Savings (%)'], 
                'D-', color=self.palette['gain'], linewidth=2, 
                markersize=8, label='Token Savings (%)')
        
        # Add savings percentage annotations
        for i, (pos, savings) in enumerate(zip(positions + group_width/2 - bar_width/2, df['Token Savings (%)'])):
            ax2.text(pos, savings + 2, 
                    f"{savings:.1f}%",
                    ha='center', va='bottom',
                    fontsize=self.annotation_fontsize,
                    color=self.palette['gain'],
                    fontweight='bold')
        
        # Set labels and title
        ax.set_ylabel('Token Count', fontsize=self.axis_label_fontsize)
        ax2.set_ylabel('Token Savings (%)', fontsize=self.axis_label_fontsize, color=self.palette['gain'])
        ax.set_title('Token Usage Analysis', fontsize=self.title_fontsize, fontweight='bold')
        
        # Set axis limits
        max_tokens = max(df['Baseline Total'].max(), df['Compound Total'].max())
        ax.set_ylim(0, max_tokens * 1.2)
        ax2.set_ylim(0, max(df['Token Savings (%)'].max() * 1.2, 10))  # Ensure we see at least up to 10%
        
        # Color the y-axis label and ticks on the right
        ax2.tick_params(axis='y', colors=self.palette['gain'])
        
        # Create custom legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Patch(facecolor=colors['Compound Input'], label='Compound AI System - Input', alpha=0.8),
            Patch(facecolor=colors['Compound Output'], label='Compound AI System - Output', alpha=0.8),
            Patch(facecolor=colors['Baseline Input'], label='Baseline - Input', alpha=0.8),
            Patch(facecolor=colors['Baseline Output'], label='Baseline - Output', alpha=0.8),
            Line2D([0], [0], color=self.palette['gain'], marker='D', linewidth=2,
                  markersize=8, label='Token Savings (%)')
        ]
        
        ax.legend(handles=legend_elements, loc='upper center',
                 bbox_to_anchor=(0.5, -0.18), ncol=3,
                 frameon=True, framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Remove top spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        # Add explanatory note
        ax.text(0.5, -0.25, "Higher token savings indicates greater inference efficiency",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')

    def _plot_scientific_radar_chart(self, results: Dict[str, Any], ax):
        """Create a scientific radar chart comparing key metrics between compound and baseline systems"""
        if not results:
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Select the first result for demonstration
        # In a real paper, you might want to aggregate results or select a specific configuration
        result_name = list(results.keys())[0]
        result = results[result_name]
        
        if 'comparison' not in result:
            ax.text(0.5, 0.5, "No comparison data available", transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        comp = result['comparison']
        
        # Define the metrics to compare - focusing on the requested metrics
        metrics = {
            'Accuracy': {
                'Compound': comp['accuracy']['compound'],
                'Baseline': comp['accuracy']['baseline'],
                'Max Value': 1.0  # Maximum possible value for normalization
            },
            'Latency\nEfficiency': {  # Inverse of latency (higher is better)
                'Compound': 1000 / comp['average_time_ms']['compound'] if comp['average_time_ms']['compound'] > 0 else 0,
                'Baseline': 1000 / comp['average_time_ms']['baseline'] if comp['average_time_ms']['baseline'] > 0 else 0,
                'Max Value': max(1000 / comp['average_time_ms']['compound'] if comp['average_time_ms']['compound'] > 0 else 0,
                               1000 / comp['average_time_ms']['baseline'] if comp['average_time_ms']['baseline'] > 0 else 0) * 1.2
            },
            'Cost\nEfficiency': {  # Inverse of cost (higher is better)
                'Compound': 1 / comp['api_costs']['compound']['total_cost'] if comp['api_costs']['compound']['total_cost'] > 0 else 0,
                'Baseline': 1 / comp['api_costs']['baseline']['total_cost'] if comp['api_costs']['baseline']['total_cost'] > 0 else 0,
                'Max Value': max(1 / comp['api_costs']['compound']['total_cost'] if comp['api_costs']['compound']['total_cost'] > 0 else 0,
                               1 / comp['api_costs']['baseline']['total_cost'] if comp['api_costs']['baseline']['total_cost'] > 0 else 0) * 1.2
            },
            'Token\nEfficiency': {  # Inverse of token count (higher is better)
                'Compound': 1 / (comp['api_costs']['compound']['total_input_tokens'] + comp['api_costs']['compound']['total_output_tokens']) 
                          if (comp['api_costs']['compound']['total_input_tokens'] + comp['api_costs']['compound']['total_output_tokens']) > 0 else 0,
                'Baseline': 1 / (comp['api_costs']['baseline']['total_input_tokens'] + comp['api_costs']['baseline']['total_output_tokens'])
                          if (comp['api_costs']['baseline']['total_input_tokens'] + comp['api_costs']['baseline']['total_output_tokens']) > 0 else 0,
                'Max Value': max(1 / (comp['api_costs']['compound']['total_input_tokens'] + comp['api_costs']['compound']['total_output_tokens']) 
                               if (comp['api_costs']['compound']['total_input_tokens'] + comp['api_costs']['compound']['total_output_tokens']) > 0 else 0,
                               1 / (comp['api_costs']['baseline']['total_input_tokens'] + comp['api_costs']['baseline']['total_output_tokens'])
                               if (comp['api_costs']['baseline']['total_input_tokens'] + comp['api_costs']['baseline']['total_output_tokens']) > 0 else 0) * 1.2
            },
            'Router\nAccuracy': {
                'Compound': comp['router_performance']['accuracy'],
                'Baseline': 0.5,  # Baseline has no router, so we use 0.5 as a reference (random guessing)
                'Max Value': 1.0
            }
        }
        
        # Number of metrics
        N = len(metrics)
        
        # Normalize the data between 0 and 1 for the radar chart
        normalized_data = {
            'Compound': [],
            'Baseline': []
        }
        
        # Normalize and prepare data
        metric_names = []
        for metric_name, metric_data in metrics.items():
            metric_names.append(metric_name)
            max_val = metric_data['Max Value']
            
            normalized_data['Compound'].append(metric_data['Compound'] / max_val)
            normalized_data['Baseline'].append(metric_data['Baseline'] / max_val)
        
        # Set the angles for each metric (evenly distributed)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the normalized data points
        normalized_data['Compound'] += normalized_data['Compound'][:1]
        normalized_data['Baseline'] += normalized_data['Baseline'][:1]
        
        # Set up the radar chart
        ax.set_theta_offset(np.pi / 2)  # Start at the top
        ax.set_theta_direction(-1)  # Go clockwise
        
        # Draw the background radar grid
        # Draw axis lines from center to maximum
        for angle in angles[:-1]:  # Skip the last one as it's a duplicate
            ax.plot([0, angle], [0, 1], color='gray', alpha=0.2, linewidth=1)
        
        # Draw concentric circles as grid
        for r in [0.2, 0.4, 0.6, 0.8]:
            ax.plot(angles, [r] * len(angles), color='gray', alpha=0.2, linewidth=1, linestyle='-')
        
        # Plot the data
        ax.plot(angles, normalized_data['Compound'], 'o-', 
               linewidth=2.5, color=self.palette['compound'], label='Compound AI System', zorder=10)
        ax.fill(angles, normalized_data['Compound'], color=self.palette['compound'], alpha=0.25)
        
        ax.plot(angles, normalized_data['Baseline'], 'o-', 
               linewidth=2.5, color=self.palette['baseline'], label='Baseline', zorder=10)
        ax.fill(angles, normalized_data['Baseline'], color=self.palette['baseline'], alpha=0.25)
        
        # Add metric labels at appropriate angles
        for i, angle in enumerate(angles[:-1]):  # Skip the last one as it's a duplicate
            ax.text(angle, 1.3, metric_names[i], 
                   ha='center', va='center', 
                   size=self.axis_label_fontsize,
                   fontweight='bold')
            
            # Add grid labels (0%, 20%, 40%, 60%, 80%, 100%)
            if i == 0:  # Add labels only for the first axis
                for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    ax.text(angle, r, f"{int(r*100)}%", 
                           ha='center', va='bottom',
                           fontsize=self.annotation_fontsize-2,
                           color='gray')
        
        # Remove yticks for cleaner look
        ax.set_yticks([])
        
        # Hide the axis labels which overlap with our custom labels
        ax.set_xticklabels([])
        
        # Add title
        ax.set_title('Compound AI System vs Baseline Comparison', 
                    fontsize=self.title_fontsize, 
                    fontweight='bold', 
                    y=1.1)
        
        # Add legend with better placement
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                 frameon=True, framealpha=0.9, facecolor='white', 
                 edgecolor='gray', ncol=2)
        
        # Add a note explaining the metrics
        explanation = "Higher values indicate better performance across all metrics"
        plt.figtext(0.5, 0.01, explanation, ha='center', 
                   fontsize=self.annotation_fontsize, fontstyle='italic')

    def create_summary_dashboard(self, results: Dict[str, Any], title: str = "CompoundAI System Results"):
        """
        Create a comprehensive dashboard of results with scientific quality plots.

        Args:
            results: Dictionary of test results
            title: Dashboard title
        """
        print("CREATING DASHBOARD: ", len(results))

        # Create figure with scientific layout
        fig = plt.figure(figsize=(20, 24), dpi=self.dpi)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Create a more structured grid layout
        gs = GridSpec(4, 2, figure=fig)
        
        # 1. Accuracy comparison (Top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_scientific_accuracy_comparison(results, ax1)
        
        # 2. Router performance (Top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_scientific_router_performance(results, ax2)
        
        # 3. Model usage (Second row, left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_scientific_model_usage(results, ax3)
        
        # 4. Latency comparison (Second row, right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_scientific_latency_comparison(results, ax4)
        
        # 5. Accuracy by difficulty (Third row, left)
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_scientific_accuracy_by_difficulty(results, ax5)
        
        # 6. Cost savings (Third row, right)
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_scientific_cost_savings(results, ax6)
        
        # 7. Token usage (Bottom left)
        ax7 = fig.add_subplot(gs[3, 0])
        self._plot_scientific_token_usage(results, ax7)
        
        # 8. Radar chart for overall comparison (Bottom right)
        ax8 = fig.add_subplot(gs[3, 1], polar=True)
        self._plot_scientific_radar_chart(results, ax8)
        
        # Add overall title with paper quality styling
        plt.suptitle(title, fontsize=24, fontweight='bold', y=0.98)
        
        # Add dashboard metadata
        metadata = f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')} | Tests: {len(results)}"
        plt.figtext(0.5, 0.96, metadata, ha='center', fontsize=12, fontstyle='italic')

        plt.show()
        # Save figure in both PNG and PDF (for papers)
        plt.savefig(os.path.join(self.output_dir, "scientific_dashboard.png"), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, "scientific_dashboard.pdf"), 
                   bbox_inches='tight')
        plt.close()
        
        # Also save individual visualizations for potential inclusion in papers
        self.save_individual_visualizations(results, "scientific")
        
        print(f"Scientific dashboard saved to {self.output_dir}")

    def create_detailed_report(self, results: Dict[str, Any]):
        """
        Create a detailed breakdown of each test result with scientific quality plots.

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
            fig = plt.figure(figsize=(15, 24), dpi=self.dpi)
            gs = GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 1.2, 1.5, 1.2])

            # 1. Accuracy pie chart - more scientific version
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_scientific_accuracy_pie(result, ax1)

            # Save individual visualization
            plt.figure(figsize=(8, 8), dpi=self.dpi)
            self._plot_scientific_accuracy_pie(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "scientific_accuracy_pie.png"), dpi=self.dpi, bbox_inches='tight')
            plt.savefig(os.path.join(detailed_dir, "scientific_accuracy_pie.pdf"), bbox_inches='tight')
            plt.close()

            # 2. Router confusion matrix - more scientific version
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_scientific_router_confusion(result, ax2)

            # Save individual visualization
            plt.figure(figsize=(8, 8), dpi=self.dpi)
            self._plot_scientific_router_confusion(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "scientific_router_confusion.png"), dpi=self.dpi, bbox_inches='tight')
            plt.savefig(os.path.join(detailed_dir, "scientific_router_confusion.pdf"), bbox_inches='tight')
            plt.close()

            # 3. Latency distribution - more scientific version
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_scientific_latency_distribution(result, ax3)

            # Save individual visualization
            plt.figure(figsize=(8, 8), dpi=self.dpi)
            self._plot_scientific_latency_distribution(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "scientific_latency_distribution.png"), dpi=self.dpi, bbox_inches='tight')
            plt.savefig(os.path.join(detailed_dir, "scientific_latency_distribution.pdf"), bbox_inches='tight')
            plt.close()

            # 4. Token usage breakdown - more scientific version
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_scientific_token_breakdown(result, ax4)

            # Save individual visualization
            plt.figure(figsize=(8, 8), dpi=self.dpi)
            self._plot_scientific_token_breakdown(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "scientific_token_breakdown.png"), dpi=self.dpi, bbox_inches='tight')
            plt.savefig(os.path.join(detailed_dir, "scientific_token_breakdown.pdf"), bbox_inches='tight')
            plt.close()

            # 5. Performance by question type - more scientific version
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_scientific_performance_by_question(result, ax5)

            # Save individual visualization
            plt.figure(figsize=(12, 6), dpi=self.dpi)
            self._plot_scientific_performance_by_question(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "scientific_performance_by_question.png"), dpi=self.dpi, bbox_inches='tight')
            plt.savefig(os.path.join(detailed_dir, "scientific_performance_by_question.pdf"), bbox_inches='tight')
            plt.close()

            # 6. Error analysis - more scientific version
            ax6 = fig.add_subplot(gs[3, :])
            self._plot_scientific_error_analysis(result, ax6)

            # Save individual visualization
            plt.figure(figsize=(12, 6), dpi=self.dpi)
            self._plot_scientific_error_analysis(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "scientific_error_analysis.png"), dpi=self.dpi, bbox_inches='tight')
            plt.savefig(os.path.join(detailed_dir, "scientific_error_analysis.pdf"), bbox_inches='tight')
            plt.close()

            # 7. Key metrics table - more scientific version
            ax7 = fig.add_subplot(gs[4, :])
            self._add_scientific_metrics_table(result, ax7)

            # Save individual visualization
            plt.figure(figsize=(12, 6), dpi=self.dpi)
            self._add_scientific_metrics_table(result, plt.gca())
            plt.tight_layout()
            plt.savefig(os.path.join(detailed_dir, "scientific_metrics_table.png"), dpi=self.dpi, bbox_inches='tight')
            plt.savefig(os.path.join(detailed_dir, "scientific_metrics_table.pdf"), bbox_inches='tight')
            plt.close()

            # Set title
            fig.suptitle(f"Detailed Scientific Analysis: {test_name}", 
                        fontsize=24, fontweight='bold', y=0.98)

            # Adjust layout
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            # Save figure
            plt.savefig(os.path.join(self.output_dir, f"{test_name}_scientific_detailed.png"), 
                       dpi=self.dpi, bbox_inches='tight')
            plt.savefig(os.path.join(self.output_dir, f"{test_name}_scientific_detailed.pdf"), 
                       bbox_inches='tight')
            plt.close()

            print(f"Detailed scientific report for {test_name} saved to {detailed_dir}")

    def _plot_scientific_accuracy_pie(self, result: Dict[str, Any], ax):
        """Plot scientific accuracy pie chart"""
        if 'compound_results' not in result:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return

        # Count correct and incorrect
        correct = sum(1 for r in result['compound_results'] if r.get('correct'))
        incorrect = len(result['compound_results']) - correct
        total = len(result['compound_results'])
        
        # Calculate percentages for display
        correct_pct = correct / total * 100
        incorrect_pct = incorrect / total * 100

        # Create pie chart with scientific styling
        wedges, texts = ax.pie(
            [correct, incorrect], 
            labels=None,  # We'll add custom labels
            autopct=None,  # We'll add custom percentage labels
            startangle=90, 
            colors=[self.palette['compound'], '#d3d3d3'],  # Light gray for incorrect
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            shadow=True
        )
        
        # Add a circle at the center to make it look like a donut chart (more scientific)
        centre_circle = plt.Circle((0, 0), 0.5, fc='white', edgecolor='white')
        ax.add_patch(centre_circle)
        
        # Add accuracy percentage in the center
        ax.text(0, 0, f"{correct_pct:.1f}%\nAccuracy", 
               ha='center', va='center', 
               fontsize=14, fontweight='bold',
               color=self.palette['compound'])
        
        # Add custom legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=self.palette['compound'], 
                          ec='white', linewidth=1.5, 
                          label=f'Correct ({correct}/{total}, {correct_pct:.1f}%)'),
            plt.Rectangle((0, 0), 1, 1, fc='#d3d3d3', 
                          ec='white', linewidth=1.5, 
                          label=f'Incorrect ({incorrect}/{total}, {incorrect_pct:.1f}%)')
        ]
        ax.legend(handles=legend_elements, loc='lower center', 
                 bbox_to_anchor=(0.5, -0.1), frameon=True, ncol=2,
                 framealpha=0.9, facecolor='white', edgecolor='gray')
        
        ax.set_title('Accuracy Distribution', fontsize=self.title_fontsize, fontweight='bold')
        
        # Equal aspect ratio
        ax.set_aspect('equal')

    def _plot_scientific_router_confusion(self, result: Dict[str, Any], ax):
        """Plot scientific router confusion matrix"""
        if 'compound_results' not in result:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
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
            else:  # hard
                if r.get('predicted_difficulty') == 'hard':
                    true_hard += 1
                else:
                    false_easy += 1

        # Create confusion matrix
        cm = np.array([[true_easy, false_hard], [false_easy, true_hard]])
        
        # Calculate derived metrics
        total = cm.sum()
        accuracy = (true_easy + true_hard) / total if total > 0 else 0
        
        # Precision and recall for "hard" class
        precision_hard = true_hard / (true_hard + false_hard) if (true_hard + false_hard) > 0 else 0
        recall_hard = true_hard / (true_hard + false_easy) if (true_hard + false_easy) > 0 else 0
        f1_hard = 2 * precision_hard * recall_hard / (precision_hard + recall_hard) if (precision_hard + recall_hard) > 0 else 0
        
        # Create normalized confusion matrix for visualization
        cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
        
        # Create more scientific heatmap
        custom_cmap = LinearSegmentedColormap.from_list(
            'custom_blues', [(0.9, 0.9, 0.9), (0.1, 0.3, 0.6)], N=100)
        
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=custom_cmap, vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Frequency', rotation=270, va='bottom', fontsize=10)
        
        # Add value annotations
        thresh = cm_norm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Add normalized percentage
                pct = cm_norm[i, j] * 100
                # Add absolute count
                count = cm[i, j]
                
                # Use white text for dark background, black for light
                color = 'white' if cm_norm[i, j] > thresh else 'black'
                
                ax.text(j, i, f"{count}\n({pct:.1f}%)", 
                       ha="center", va="center", 
                       color=color, fontweight='bold',
                       fontsize=self.annotation_fontsize)

        # Configure axis
        class_labels = ['Easy', 'Hard']
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", va="center",
                rotation_mode="anchor")
        
        # Add axis labels
        ax.set_ylabel('True Difficulty', fontsize=self.axis_label_fontsize)
        ax.set_xlabel('Predicted Difficulty', fontsize=self.axis_label_fontsize)
        
        # Add title with accuracy information
        ax.set_title(f'Router Confusion Matrix\nAccuracy: {accuracy:.2f}', 
                    fontsize=self.title_fontsize, fontweight='bold')
        
        # Add F1 score and other metrics as a subtitle
        metrics_text = f"Precision (Hard): {precision_hard:.2f}, Recall (Hard): {recall_hard:.2f}, F1 (Hard): {f1_hard:.2f}"
        ax.text(0.5, -0.15, metrics_text, 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontweight='bold')
        
        # Ensure proper aspect
        ax.set_aspect('equal')

    def _plot_scientific_latency_distribution(self, result: Dict[str, Any], ax):
        """Plot scientific latency distribution"""
        if 'compound_results' not in result:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return

        # Extract latencies
        latencies = [r.get('total_time_ms', 0) for r in result['compound_results']]
        
        # Create histogram with KDE overlay for scientific look
        sns.histplot(latencies, bins=20, kde=True, color=self.palette['compound'], 
                    alpha=0.7, ax=ax, stat='density', edgecolor='white', linewidth=1)
        
        # Add mean, median and standard deviation lines
        mean_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        std_latency = np.std(latencies)
        
        # Add vertical lines for mean and median
        ax.axvline(mean_latency, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_latency:.2f}ms')
        ax.axvline(median_latency, color='green', linestyle='-.', linewidth=2, 
                  label=f'Median: {median_latency:.2f}ms')
        
        # Add annotation for standard deviation
        ax.text(0.95, 0.95, f"Std. Dev: {std_latency:.2f}ms", 
               transform=ax.transAxes, ha='right', va='top',
               fontsize=self.annotation_fontsize, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Add percentile lines
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        ax.axvline(p95, color='orange', linestyle=':', linewidth=2, 
                  label=f'95th: {p95:.2f}ms')
        ax.axvline(p99, color='purple', linestyle=':', linewidth=2, 
                  label=f'99th: {p99:.2f}ms')
        
        # Configure axes
        ax.set_xlabel('Latency (ms)', fontsize=self.axis_label_fontsize)
        ax.set_ylabel('Density', fontsize=self.axis_label_fontsize)
        ax.set_title('Latency Distribution', fontsize=self.title_fontsize, fontweight='bold')
        
        # Add grid for better readability
        ax.grid(linestyle='--', alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper right', frameon=True, 
                 framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_scientific_token_breakdown(self, result: Dict[str, Any], ax):
        """Plot scientific token usage breakdown"""
        if 'comparison' not in result or 'api_costs' not in result['comparison']:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return

        # Extract token usage
        compound = result['comparison']['api_costs']['compound']
        baseline = result['comparison']['api_costs']['baseline']
        
        # Calculate percentages
        compound_input_pct = compound['total_input_tokens'] / (compound['total_input_tokens'] + compound['total_output_tokens']) * 100 if (compound['total_input_tokens'] + compound['total_output_tokens']) > 0 else 0
        compound_output_pct = 100 - compound_input_pct
        
        baseline_input_pct = baseline['total_input_tokens'] / (baseline['total_input_tokens'] + baseline['total_output_tokens']) * 100 if (baseline['total_input_tokens'] + baseline['total_output_tokens']) > 0 else 0
        baseline_output_pct = 100 - baseline_input_pct
        
        # Calculate token savings
        total_compound = compound['total_input_tokens'] + compound['total_output_tokens']
        total_baseline = baseline['total_input_tokens'] + baseline['total_output_tokens']
        token_savings = (total_baseline - total_compound) / total_baseline * 100 if total_baseline > 0 else 0
        
        # Create data for grouped bar chart
        labels = ['Input Tokens', 'Output Tokens']
        compound_values = [compound['total_input_tokens'], compound['total_output_tokens']]
        baseline_values = [baseline['total_input_tokens'], baseline['total_output_tokens']]
        
        # Setup for grouped bars
        x = np.arange(len(labels))
        width = 0.35
        
        # Plot bars
        ax.bar(x - width/2, baseline_values, width, 
               label='Baseline', color=self.palette['baseline'], alpha=0.8, edgecolor='white', linewidth=1)
        ax.bar(x + width/2, compound_values, width, 
               label='Compound AI System', color=self.palette['compound'], alpha=0.8, edgecolor='white', linewidth=1)
        
        # Add value annotations
        for i, v in enumerate(baseline_values):
            ax.text(i - width/2, v, f"{v:,}", 
                   ha='center', va='bottom', 
                   fontsize=self.annotation_fontsize,
                   color=self.palette['baseline'],
                   fontweight='bold')
                   
        for i, v in enumerate(compound_values):
            ax.text(i + width/2, v, f"{v:,}", 
                   ha='center', va='bottom', 
                   fontsize=self.annotation_fontsize,
                   color=self.palette['compound'],
                   fontweight='bold')
        
        # Add percentage breakdown annotations
        ax.text(0 - width/2, baseline_values[0]/2, f"{baseline_input_pct:.1f}%", 
               ha='center', va='center', 
               fontsize=self.annotation_fontsize,
               color='white', fontweight='bold')
               
        ax.text(1 - width/2, baseline_values[1]/2, f"{baseline_output_pct:.1f}%", 
               ha='center', va='center', 
               fontsize=self.annotation_fontsize,
               color='white', fontweight='bold')
               
        ax.text(0 + width/2, compound_values[0]/2, f"{compound_input_pct:.1f}%", 
               ha='center', va='center', 
               fontsize=self.annotation_fontsize,
               color='white', fontweight='bold')
               
        ax.text(1 + width/2, compound_values[1]/2, f"{compound_output_pct:.1f}%", 
               ha='center', va='center', 
               fontsize=self.annotation_fontsize,
               color='white', fontweight='bold')
        
        # Configure axes
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Token Count', fontsize=self.axis_label_fontsize)
        ax.set_title('Token Usage Breakdown', fontsize=self.title_fontsize, fontweight='bold')
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Improve legend
        ax.legend(loc='upper right', frameon=True, 
                 framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add token savings information
        savings_text = f"Token Savings: {token_savings:.1f}%"
        ax.text(0.5, -0.1, savings_text, 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize + 2, fontweight='bold',
               color=self.palette['gain'])

    def _plot_scientific_performance_by_question(self, result: Dict[str, Any], ax):
        """Plot scientific performance by question type"""
        if 'compound_results' not in result or not result['compound_results']:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return

        # Group results by query length as a proxy for complexity
        query_lengths = [len(r.get('query', '')) for r in result['compound_results']]
        
        # Define bins for query length
        bins = [0, 50, 100, 150, 200, float('inf')]
        bin_labels = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
        
        # Assign bin indices
        bin_indices = np.digitize(query_lengths, bins[1:])
        
        # Calculate accuracy by bin
        bin_results = []
        for i in range(len(bin_labels)):
            bin_data = [r for r, idx in zip(result['compound_results'], bin_indices) if idx == i]
            
            if bin_data:
                correct = sum(1 for r in bin_data if r.get('correct'))
                accuracy = correct / len(bin_data) * 100
                
                # Get model distribution within this bin
                small_count = sum(1 for r in bin_data if r.get('chosen_llm') == 'small')
                large_count = len(bin_data) - small_count
                small_pct = small_count / len(bin_data) * 100
                
                bin_results.append({
                    'bin': bin_labels[i],
                    'count': len(bin_data),
                    'accuracy': accuracy,
                    'small_model_pct': small_pct,
                    'small_count': small_count,
                    'large_count': large_count
                })
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(bin_results)
        
        if len(df) == 0:
            ax.text(0.5, 0.5, "Not enough data for analysis", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        # Create a twin axis plot for accuracy and model distribution
        x = np.arange(len(df))
        width = 0.4
        
        # Plot model distribution as stacked bars
        ax.bar(x, df['small_count'], width, 
               label='Small LLM', color=self.palette['small'], alpha=0.7)
        ax.bar(x, df['large_count'], width, 
               bottom=df['small_count'], label='Large LLM', 
               color=self.palette['large'], alpha=0.7)
        
        # Add count labels to the stacked bars
        for i, row in enumerate(df.itertuples()):
            # Only add label if value is significant
            if row.small_count > 0:
                ax.text(i, row.small_count/2, f"{row.small_count}", 
                       ha='center', va='center', 
                       fontsize=self.annotation_fontsize,
                       color='black', fontweight='bold')
            
            if row.large_count > 0:
                ax.text(i, row.small_count + row.large_count/2, f"{row.large_count}", 
                       ha='center', va='center', 
                       fontsize=self.annotation_fontsize,
                       color='black', fontweight='bold')
        
        # Create twin axis for accuracy
        ax2 = ax.twinx()
        
        # Plot accuracy as line with markers
        ax2.plot(x, df['accuracy'], 'o-', color='darkgreen', linewidth=2, 
                markersize=8, label='Accuracy')
        
        # Add accuracy value annotations
        for i, acc in enumerate(df['accuracy']):
            ax2.text(i, acc + 2, f"{acc:.1f}%", 
                    ha='center', va='bottom', 
                    fontsize=self.annotation_fontsize,
                    color='darkgreen', fontweight='bold')
        
        # Configure axes
        ax.set_xlabel('Query Length', fontsize=self.axis_label_fontsize)
        ax.set_ylabel('Sample Count', fontsize=self.axis_label_fontsize)
        ax2.set_ylabel('Accuracy (%)', fontsize=self.axis_label_fontsize, color='darkgreen')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(df['bin'])
        
        # Set y-limits
        ax2.set_ylim(0, 105)
        
        # Color the y-axis label and ticks on the right
        ax2.tick_params(axis='y', colors='darkgreen')
        
        # Add title
        ax.set_title('Performance by Query Length', fontsize=self.title_fontsize, fontweight='bold')
        
        # Create a unified legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                 bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True,
                 framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Remove top spine for cleaner look
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        # Add a note about the relationship between query length and complexity
        ax.text(0.5, -0.2, "Longer queries typically represent more complex questions",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')

    def _plot_scientific_error_analysis(self, result: Dict[str, Any], ax):
        """Plot scientific error analysis"""
        if 'compound_results' not in result:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
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
        
        # Calculate accuracy for each category
        accuracy = []
        for c, ic in zip(correct, incorrect):
            total = c + ic
            accuracy.append(c / total * 100 if total > 0 else 0)
        
        # Set up the plot
        x = np.arange(len(categories))
        width = 0.6
        
        # Create stacked bars
        ax.bar(x, correct, width, label='Correct', color='#4CAF50', alpha=0.8)
        ax.bar(x, incorrect, width, bottom=correct, label='Incorrect', color='#F44336', alpha=0.8)
        
        # Add count labels to the segments
        for i, (c, ic) in enumerate(zip(correct, incorrect)):
            total = c + ic
            
            # Only add labels if the values are significant
            if c > 0:
                ax.text(i, c/2, f"{c}", 
                       ha='center', va='center', 
                       fontsize=self.annotation_fontsize,
                       color='white', fontweight='bold')
            
            if ic > 0:
                ax.text(i, c + ic/2, f"{ic}", 
                       ha='center', va='center', 
                       fontsize=self.annotation_fontsize,
                       color='white', fontweight='bold')
            
            # Add accuracy annotation at the top
            if total > 0:
                ax.text(i, total + 1, f"{accuracy[i]:.1f}%", 
                       ha='center', va='bottom', 
                       fontsize=self.annotation_fontsize,
                       color='#4CAF50', fontweight='bold')
        
        # Configure axes
        ax.set_ylabel('Sample Count', fontsize=self.axis_label_fontsize)
        ax.set_title('Error Analysis by Model and Question Difficulty', 
                    fontsize=self.title_fontsize, fontweight='bold')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Improve legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                 frameon=True, framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add an interpretive note
        note = "Higher accuracy for Small LLM on easy questions indicates good router efficiency"
        ax.text(0.5, -0.2, note,
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')

    def _add_scientific_metrics_table(self, result: Dict[str, Any], ax):
        """Add scientific key metrics table"""
        if 'comparison' not in result:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return

        # Hide axes
        ax.axis('off')
        
        # Extract key metrics
        comp = result['comparison']
        
        # Prepare data for table
        # Create a more detailed scientific table
        metrics = [
            ['Metric', 'Compound AI System', 'Baseline', 'Difference/Ratio'],
            ['Overall Accuracy', f"{comp['accuracy']['compound']:.2%}", 
             f"{comp['accuracy']['baseline']:.2%}", 
             f"{comp['accuracy']['difference']:.2%} ({comp['accuracy']['difference']/comp['accuracy']['baseline']*100 if comp['accuracy']['baseline'] > 0 else 0:.1f}%)"],
            
            ['Easy Questions Accuracy', f"{comp['accuracy_by_difficulty']['easy']['compound']:.2%}", 
             f"{comp['accuracy_by_difficulty']['easy']['baseline']:.2%}", 
             f"{comp['accuracy_by_difficulty']['easy']['compound'] - comp['accuracy_by_difficulty']['easy']['baseline']:.2%}"],
            
            ['Hard Questions Accuracy', f"{comp['accuracy_by_difficulty']['hard']['compound']:.2%}", 
             f"{comp['accuracy_by_difficulty']['hard']['baseline']:.2%}", 
             f"{comp['accuracy_by_difficulty']['hard']['compound'] - comp['accuracy_by_difficulty']['hard']['baseline']:.2%}"],
            
            ['Average Latency', f"{comp['average_time_ms']['compound']:.2f}ms", 
             f"{comp['average_time_ms']['baseline']:.2f}ms", 
             f"{comp['average_time_ms']['speedup']:.2f}x"],
            
            ['Total API Cost', f"${comp['api_costs']['compound']['total_cost']:.4f}", 
             f"${comp['api_costs']['baseline']['total_cost']:.4f}", 
             f"${comp['api_costs']['savings']['amount']:.4f} ({comp['api_costs']['savings']['percentage']:.2f}%)"],
            
            ['Total Input Tokens', f"{comp['api_costs']['compound']['total_input_tokens']:,}", 
             f"{comp['api_costs']['baseline']['total_input_tokens']:,}", 
             f"{comp['api_costs']['baseline']['total_input_tokens'] - comp['api_costs']['compound']['total_input_tokens']:,}"],
            
            ['Total Output Tokens', f"{comp['api_costs']['compound']['total_output_tokens']:,}", 
             f"{comp['api_costs']['baseline']['total_output_tokens']:,}", 
             f"{comp['api_costs']['baseline']['total_output_tokens'] - comp['api_costs']['compound']['total_output_tokens']:,}"],
            
            ['Router Accuracy', f"{comp['router_performance']['accuracy']:.2%}", 
             "N/A", "N/A"],
            
            ['Small LLM Usage', f"{comp['resource_utilization']['small_llm_usage']:.2%}", 
             "0%", "N/A"]
        ]
        
        # Create scientific table with color coding
        table = ax.table(
            cellText=metrics,
            cellLoc='center',
            loc='center',
            colWidths=[0.25, 0.25, 0.25, 0.25]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        
        # Color code cells based on their meaning
        for i in range(len(metrics)):
            for j in range(4):
                cell = table[(i, j)]
                
                # Header row
                if i == 0:
                    cell.set_facecolor('#E6E6E6')
                    cell.set_text_props(weight='bold')
                
                # Metric names column
                elif j == 0:
                    cell.set_facecolor('#F5F5F5')
                    cell.set_text_props(weight='bold')
                
                # Compound AI System results
                elif j == 1:
                    cell.set_facecolor('#E3F2FD')  # Light blue
                
                # Baseline results
                elif j == 2:
                    cell.set_facecolor('#FFEBEE')  # Light red
                
                # Difference/ratio column - color code based on improvement
                elif j == 3 and i > 0:
                    text = cell.get_text().get_text()
                    
                    # If the text contains a percentage or 'x', check if it's positive
                    if '%' in text or 'x' in text:
                        if '-' in text or text.startswith('('):
                            cell.set_facecolor('#FFCDD2')  # Darker red for negative change
                        else:
                            cell.set_facecolor('#C8E6C9')  # Green for positive change
        
        # Add a title
        ax.set_title('Key Performance Metrics', fontsize=self.title_fontsize, fontweight='bold', pad=20)
        
        # Add interpretation note below the table
        note = "Positive values in the difference column indicate improvements by the Compound AI system."
        ax.text(0.5, 0.02, note,
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')

    def create_comparison_visualization(self, results: Dict[str, Any], focus_metrics: List[str] = None):
        """
        Create scientific visualization comparing different test configurations.

        Args:
            results: Dictionary of test results
            focus_metrics: List of metrics to focus on (default: accuracy, latency, cost_savings)
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
        fig, axes = plt.subplots(len(focus_metrics), 1, figsize=(12, 6 * len(focus_metrics)), dpi=self.dpi)

        # If only one metric, axes is not a list
        if len(focus_metrics) == 1:
            axes = [axes]

        # Process each metric
        for i, metric in enumerate(focus_metrics):
            if metric == 'accuracy':
                self._plot_scientific_comparison_accuracy(results, axes[i])
            elif metric == 'latency':
                self._plot_scientific_comparison_latency(results, axes[i])
            elif metric == 'cost_savings':
                self._plot_scientific_comparison_cost_savings(results, axes[i])
            elif metric == 'router_accuracy':
                self._plot_scientific_comparison_router_accuracy(results, axes[i])
            elif metric == 'model_usage':
                self._plot_scientific_comparison_model_usage(results, axes[i])

            # Create and save individual plots too
            plt.figure(figsize=(10, 6), dpi=self.dpi)
            ax = plt.gca()

            if metric == 'accuracy':
                self._plot_scientific_comparison_accuracy(results, ax)
            elif metric == 'latency':
                self._plot_scientific_comparison_latency(results, ax)
            elif metric == 'cost_savings':
                self._plot_scientific_comparison_cost_savings(results, ax)
            elif metric == 'router_accuracy':
                self._plot_scientific_comparison_router_accuracy(results, ax)
            elif metric == 'model_usage':
                self._plot_scientific_comparison_model_usage(results, ax)

            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f"scientific_comparison_{metric}.png"), dpi=self.dpi, bbox_inches='tight')
            plt.savefig(os.path.join(comparison_dir, f"scientific_comparison_{metric}.pdf"), bbox_inches='tight')
            plt.close()

        # Set title
        fig.suptitle(f"Scientific Comparison of Different Configurations", 
                    fontsize=self.title_fontsize + 4, fontweight='bold', y=0.98)

        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        plt.savefig(os.path.join(self.output_dir, "scientific_configuration_comparison.png"), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, "scientific_configuration_comparison.pdf"), 
                   bbox_inches='tight')
        plt.close()
        
        print(f"Comparison visualizations saved to {comparison_dir}")

    def _plot_scientific_comparison_accuracy(self, results: Dict[str, Any], ax):
        """Plot scientific accuracy comparison across configs"""
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

        # Create DataFrame for better plotting
        df = pd.DataFrame({
            'Configuration': test_names,
            'Compound AI System': compound_acc,
            'Baseline': baseline_acc,
            'Improvement': difference
        })
        
        # Sort by improvement for better visualization
        df = df.sort_values('Improvement', ascending=False)
        
        # Set up grouped bar plot
        x = np.arange(len(df))
        width = 0.35
        
        # Plot with more scientific styling
        compound_bars = ax.bar(x - width/2, df['Compound AI System'], width, 
                              color=self.palette['compound'], label='Compound AI System',
                              edgecolor='white', linewidth=1, alpha=0.8)
        baseline_bars = ax.bar(x + width/2, df['Baseline'], width, 
                              color=self.palette['baseline'], label='Baseline',
                              edgecolor='white', linewidth=1, alpha=0.8)
        
        # Add labels with improved styling
        for i, bar in enumerate(compound_bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f"{df['Compound AI System'].iloc[i]:.1f}%", 
                   ha='center', va='bottom', 
                   color=self.palette['compound'],
                   fontsize=self.annotation_fontsize,
                   fontweight='bold')
                   
        for i, bar in enumerate(baseline_bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f"{df['Baseline'].iloc[i]:.1f}%", 
                   ha='center', va='bottom', 
                   color=self.palette['baseline'],
                   fontsize=self.annotation_fontsize,
                   fontweight='bold')
        
        # Add improvement annotations
        for i, (imp, comp, base) in enumerate(zip(df['Improvement'], df['Compound AI System'], df['Baseline'])):
            y_pos = max(comp, base) + 5
            # Use green for positive, red for negative
            color = self.palette['gain'] if imp >= 0 else self.palette['loss']
            
            text = f"+{imp:.1f}%" if imp >= 0 else f"{imp:.1f}%"
            ax.text(i, y_pos, text,
                   ha='center', va='bottom',
                   color=color, fontweight='bold',
                   fontsize=self.annotation_fontsize)
        
        # Configure axes
        ax.set_ylabel('Accuracy (%)', fontsize=self.axis_label_fontsize)
        ax.set_title('Accuracy Comparison Across Configurations', 
                    fontsize=self.title_fontsize, fontweight='bold')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(df['Configuration'], rotation=45, ha='right')
        
        # Set y-axis to start at 0
        ax.set_ylim(0, max(df['Compound AI System'].max(), df['Baseline'].max()) * 1.2)
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Improve legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
                 frameon=True, framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add an interpretive note
        ax.text(0.5, -0.2, "Higher improvements indicate better performance of the Compound AI system",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')

    def _plot_scientific_comparison_latency(self, results: Dict[str, Any], ax):
        """Plot scientific latency comparison across configs"""
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

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Configuration': test_names,
            'Compound AI System': compound_latency,
            'Baseline': baseline_latency,
            'Speedup': speedup
        })
        
        # Sort by speedup for better visualization
        df = df.sort_values('Speedup', ascending=False)
        
        # Set up for grouped bar plot
        x = np.arange(len(df))
        width = 0.35
        
        # Plot latency with scientific styling
        compound_bars = ax.bar(x - width/2, df['Compound AI System'], width, 
                              color=self.palette['compound'], label='Compound AI System',
                              edgecolor='white', linewidth=1, alpha=0.8)
        baseline_bars = ax.bar(x + width/2, df['Baseline'], width, 
                              color=self.palette['baseline'], label='Baseline',
                              edgecolor='white', linewidth=1, alpha=0.8)
        
        # Add a twin axis for speedup
        ax2 = ax.twinx()
        
        # Plot speedup as a line with markers
        speedup_line = ax2.plot(x, df['Speedup'], 'o-', 
                               color=self.palette['gain'], linewidth=2.5,
                               label='Speedup Factor', marker='D', markersize=8,
                               zorder=10)
        
        # Add reference line at speedup = 1
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        # Add latency labels
        for i, bar in enumerate(compound_bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f"{df['Compound AI System'].iloc[i]:.0f}ms", 
                   ha='center', va='bottom', 
                   color=self.palette['compound'],
                   fontsize=self.annotation_fontsize-1,
                   fontweight='bold')
                   
        for i, bar in enumerate(baseline_bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f"{df['Baseline'].iloc[i]:.0f}ms", 
                   ha='center', va='bottom', 
                   color=self.palette['baseline'],
                   fontsize=self.annotation_fontsize-1,
                   fontweight='bold')
        
        # Add speedup labels
        for i, speed in enumerate(df['Speedup']):
            ax2.text(i, speed + 0.1, f"{speed:.1f}×",
                    ha='center', va='bottom',
                    color=self.palette['gain'],
                    fontsize=self.annotation_fontsize,
                    fontweight='bold')
        
        # Configure axes
        ax.set_ylabel('Latency (ms)', fontsize=self.axis_label_fontsize)
        ax2.set_ylabel('Speedup Factor', fontsize=self.axis_label_fontsize, color=self.palette['gain'])
        ax.set_title('Latency Comparison Across Configurations', 
                    fontsize=self.title_fontsize, fontweight='bold')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(df['Configuration'], rotation=45, ha='right')
        
        # Color the y-axis label and ticks on the right
        ax2.tick_params(axis='y', colors=self.palette['gain'])
        
        # Set axis limits
        max_latency = max(df['Baseline'].max(), df['Compound AI System'].max())
        ax.set_ylim(0, max_latency * 1.3)
        ax2.set_ylim(0, max(df['Speedup'].max() * 1.2, 3.0))  # Ensure we at least see up to 3x
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                 bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True,
                 framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Remove top spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        # Add an interpretive note
        ax.text(0.5, -0.2, "Higher speedup factor indicates better performance of the Compound AI system",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')

    def _plot_scientific_comparison_cost_savings(self, results: Dict[str, Any], ax):
        """Plot scientific cost savings comparison across configs"""
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

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Configuration': test_names,
            'Compound AI System': compound_cost,
            'Baseline': baseline_cost,
            'Savings (%)': savings_pct,
            'Savings ($)': savings_amount
        })
        
        # Sort by savings percentage for better visualization
        df = df.sort_values('Savings (%)', ascending=False)
        
        # Set up for grouped bar plot
        x = np.arange(len(df))
        width = 0.35
        
        # Plot with scientific styling
        compound_bars = ax.bar(x - width/2, df['Compound AI System'], width, 
                              color=self.palette['compound'], label='Compound AI System',
                              edgecolor='white', linewidth=1, alpha=0.8)
        baseline_bars = ax.bar(x + width/2, df['Baseline'], width, 
                              color=self.palette['baseline'], label='Baseline',
                              edgecolor='white', linewidth=1, alpha=0.8)
        
        # Add a twin axis for savings percentage
        ax2 = ax.twinx()
        
        # Plot savings percentage as a line with markers
        savings_line = ax2.plot(x, df['Savings (%)'], 'o-', 
                               color=self.palette['gain'], linewidth=2.5,
                               label='Cost Savings (%)', marker='s', markersize=8,
                               zorder=10)
        
        # Add cost labels
        for i, bar in enumerate(compound_bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"${df['Compound AI System'].iloc[i]:.4f}", 
                   ha='center', va='bottom', 
                   color=self.palette['compound'],
                   fontsize=self.annotation_fontsize-1,
                   fontweight='bold')
                   
        for i, bar in enumerate(baseline_bars):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f"${df['Baseline'].iloc[i]:.4f}", 
                   ha='center', va='bottom', 
                   color=self.palette['baseline'],
                   fontsize=self.annotation_fontsize-1,
                   fontweight='bold')
        
        # Add savings percentage labels
        for i, (pct, amt) in enumerate(zip(df['Savings (%)'], df['Savings ($)'])):
            ax2.text(i, pct + 2, f"{pct:.1f}%\n(${amt:.4f})",
                    ha='center', va='bottom',
                    color=self.palette['gain'],
                    fontsize=self.annotation_fontsize-1,
                    fontweight='bold')
        
        # Configure axes
        ax.set_ylabel('Cost ($)', fontsize=self.axis_label_fontsize)
        ax2.set_ylabel('Savings (%)', fontsize=self.axis_label_fontsize, color=self.palette['gain'])
        ax.set_title('API Cost Comparison Across Configurations', 
                    fontsize=self.title_fontsize, fontweight='bold')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(df['Configuration'], rotation=45, ha='right')
        
        # Color the y-axis label and ticks on the right
        ax2.tick_params(axis='y', colors=self.palette['gain'])
        
        # Set axis limits
        max_cost = max(df['Baseline'].max(), df['Compound AI System'].max())
        ax.set_ylim(0, max_cost * 1.3)
        ax2.set_ylim(0, max(df['Savings (%)'].max() * 1.2, 10))  # Ensure we at least see up to 10%
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                 bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True,
                 framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Remove top spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        # Add an interpretive note
        ax.text(0.5, -0.2, "Higher savings indicates better cost efficiency of the Compound AI system",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')

    def _plot_scientific_comparison_router_accuracy(self, results: Dict[str, Any], ax):
        """Plot scientific router accuracy comparison across configs"""
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

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Configuration': test_names,
            'Router Accuracy': router_acc,
            'False Positives': false_pos,
            'False Negatives': false_neg
        })
        
        # Sort by router accuracy for better visualization
        df = df.sort_values('Router Accuracy', ascending=False)
        
        # Set up for the plot
        x = np.arange(len(df))
        
        # Create a more scientific bar and line combination
        # Bar for accuracy
        accuracy_bars = ax.bar(x, df['Router Accuracy'], width=0.6, 
                              color=self.palette['compound'], alpha=0.8,
                              edgecolor='white', linewidth=1,
                              label='Router Accuracy (%)')
        
        # Add a twin axis for error counts
        ax2 = ax.twinx()
        
        # Plot errors as line with markers
        width = 0.2
        ax2.plot(x-width, df['False Positives'], 'o-', 
                color=self.palette['hard'], linewidth=2, 
                label='False Positives (Hard as Easy)', 
                marker='o', markersize=8)
        ax2.plot(x+width, df['False Negatives'], 's-', 
                color=self.palette['easy'], linewidth=2, 
                label='False Negatives (Easy as Hard)', 
                marker='s', markersize=8)
        
        # Add accuracy labels
        for i, acc in enumerate(df['Router Accuracy']):
            ax.text(i, acc + 1, f"{acc:.1f}%",
                   ha='center', va='bottom',
                   color=self.palette['compound'],
                   fontsize=self.annotation_fontsize,
                   fontweight='bold')
        
        # Add error count labels
        for i, (fp, fn) in enumerate(zip(df['False Positives'], df['False Negatives'])):
            if fp > 0:
                ax2.text(i-width, fp, str(fp),
                        ha='center', va='bottom',
                        color=self.palette['hard'],
                        fontsize=self.annotation_fontsize,
                        fontweight='bold')
            
            if fn > 0:
                ax2.text(i+width, fn, str(fn),
                        ha='center', va='bottom',
                        color=self.palette['easy'],
                        fontsize=self.annotation_fontsize,
                        fontweight='bold')
        
        # Configure axes
        ax.set_ylabel('Accuracy (%)', fontsize=self.axis_label_fontsize)
        ax2.set_ylabel('Error Count', fontsize=self.axis_label_fontsize)
        ax.set_title('Router Performance Across Configurations', 
                    fontsize=self.title_fontsize, fontweight='bold')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(df['Configuration'], rotation=45, ha='right')
        
        # Set axis limits
        ax.set_ylim(0, 105)
        max_error = max(df['False Positives'].max(), df['False Negatives'].max())
        ax2.set_ylim(0, max_error * 1.3)
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                 bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True,
                 framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Remove top spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        # Add an interpretive note
        ax.text(0.5, -0.2, "Higher router accuracy with fewer errors indicates better routing efficiency",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')

    def _plot_scientific_comparison_model_usage(self, results: Dict[str, Any], ax):
        """Plot scientific model usage comparison across configs"""
        test_names = []
        small_usage = []
        accuracy = []

        for name, result in results.items():
            if 'comparison' in result and 'resource_utilization' in result['comparison']:
                test_names.append(name)
                util = result['comparison']['resource_utilization']
                small_usage.append(util['small_llm_usage'] * 100)
                accuracy.append(result['comparison']['accuracy']['compound'] * 100)

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Configuration': test_names,
            'Small LLM Usage (%)': small_usage,
            'Accuracy (%)': accuracy
        })
        
        # Sort by small LLM usage for better visualization
        df = df.sort_values('Small LLM Usage (%)', ascending=False)
        
        # Set up for plot
        x = np.arange(len(df))
        
        # Create scientific bar and scatter combination
        # Bar for small LLM usage
        usage_bars = ax.bar(x, df['Small LLM Usage (%)'], width=0.6, 
                           color=self.palette['small'], alpha=0.8,
                           edgecolor='white', linewidth=1,
                           label='Small LLM Usage (%)')
        
        # Add a twin axis for accuracy
        ax2 = ax.twinx()
        
        # Plot accuracy as scatter points with line
        accuracy_scatter = ax2.plot(x, df['Accuracy (%)'], 'o-', 
                                   color='darkgreen', linewidth=2.5,
                                   label='Accuracy (%)', marker='D', markersize=10,
                                   zorder=10)
        
        # Add usage labels
        for i, usage in enumerate(df['Small LLM Usage (%)']):
            ax.text(i, usage + 1, f"{usage:.1f}%",
                   ha='center', va='bottom',
                   color=self.palette['small'],
                   fontsize=self.annotation_fontsize,
                   fontweight='bold')
        
        # Add accuracy labels
        for i, acc in enumerate(df['Accuracy (%)']):
            ax2.text(i, acc + 1, f"{acc:.1f}%",
                    ha='center', va='bottom',
                    color='darkgreen',
                    fontsize=self.annotation_fontsize,
                    fontweight='bold')
        
        # Configure axes
        ax.set_ylabel('Small LLM Usage (%)', fontsize=self.axis_label_fontsize)
        ax2.set_ylabel('Accuracy (%)', fontsize=self.axis_label_fontsize, color='darkgreen')
        ax.set_title('Model Usage and Accuracy Across Configurations', 
                    fontsize=self.title_fontsize, fontweight='bold')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(df['Configuration'], rotation=45, ha='right')
        
        # Color the y-axis label and ticks on the right
        ax2.tick_params(axis='y', colors='darkgreen')
        
        # Set axis limits
        ax.set_ylim(0, 105)
        ax2.set_ylim(0, 105)
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                 bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True,
                 framealpha=0.9, facecolor='white', edgecolor='gray')
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Remove top spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        # Add an interpretive note
        ax.text(0.5, -0.2, "Higher small LLM usage with high accuracy indicates optimal resource efficiency",
               transform=ax.transAxes, ha='center', va='center',
               fontsize=self.annotation_fontsize, fontstyle='italic')