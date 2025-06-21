# generate_visualizations.py

import os
from visualization.results_visualizer import ResultsVisualizer

def main():
    """
    Generates all visualizations for the CompoundAI test results, focusing
    on a primary, high-quality dashboard.
    """
    print("Starting scientific visualization generation...")

    # Define paths
    results_dir = "results"
    output_dir = os.path.join(results_dir, "visualizations_output")

    # Initialize the visualizer
    visualizer = ResultsVisualizer(results_dir, output_dir)
    print(f"Visualizations will be saved to: {os.path.abspath(output_dir)}")

    # Load results from the specified JSON file
    # You can change the pattern to load multiple files for comparison,
    # e.g., "*_results_full.json"
    results = visualizer.load_results(pattern="*_full.json")

    if not results:
        print(f"No results found matching '*_full.json' in '{results_dir}'. Exiting.")
        return

    print(f"Loaded {len(results)} result file(s): {', '.join(results.keys())}")

    # --- Primary Output: The Summary Dashboard ---
    try:
        print("\nCreating the main summary dashboard...")
        visualizer.create_summary_dashboard(
            results,
            title="CompoundAI System Performance Analysis"
        )
        print("-> Main dashboard created successfully.")
    except Exception as e:
        print(f"Could not create the main summary dashboard. Error: {e}")
        # Optionally, re-raise the exception if this is a critical failure
        # raise

    # --- Secondary, Optional Outputs ---

    # Generate individual plots for closer inspection
    try:
        print("\nGenerating individual plots for detailed analysis...")
        visualizer.save_individual_visualizations(results)
        print("-> Individual plots saved successfully.")
    except Exception as e:
        print(f"Could not save individual plots. Error: {e}")


    # Generate detailed breakdown reports (confusion matrices, etc.)
    try:
        print("\nCreating detailed breakdown reports...")
        visualizer.create_detailed_report(results)
        print("-> Detailed reports created successfully.")
    except Exception as e:
        print(f"Could not create detailed reports. Error: {e}")

    # Generate comparison plots only if more than one result file is loaded
    if len(results) > 1:
        try:
            print("\nCreating comparison visualizations across all loaded configurations...")
            focus_metrics = ['accuracy', 'latency', 'cost_savings', 'router_accuracy', 'model_usage']
            visualizer.create_comparison_visualization(results, focus_metrics=focus_metrics)
            print("-> Comparison visualizations created successfully.")
        except Exception as e:
            print(f"Could not create comparison visualizations. Error: {e}")
    else:
        print("\nSkipping multi-configuration comparison as only one result file was loaded.")

    print("\n\nVisualization generation complete!")
    print(f"All outputs are in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main() 