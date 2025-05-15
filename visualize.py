# scripts/visualize_results.py

import os
import argparse
import glob
from visualization import ResultsVisualizerV2
from src.utils.logging import setup_logging


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Visualize CompoundAI system test results")

    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing test results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save visualizations (defaults to results_dir/visualizations)')
    parser.add_argument('--pattern', type=str, default="*_full.json",
                        help='File pattern for results files')
    parser.add_argument('--comparison', action='store_true',
                        help='Create comparison visualization across configurations')
    parser.add_argument('--metrics', type=str, default="accuracy,latency,cost_savings",
                        help='Comma-separated list of metrics to include in comparison')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    # Parse arguments
    args = parse_arguments()

    # Configure logging
    logger = setup_logging(name="visualize_results",
                           level="DEBUG" if args.debug else "INFO")

    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        logger.error(f"Results directory not found: {args.results_dir}")
        return 1

    # Check if there are any results files
    results_files = glob.glob(os.path.join(args.results_dir, args.pattern))
    if not results_files:
        logger.error(f"No results files found matching pattern '{args.pattern}' in {args.results_dir}")
        return 1

    logger.info(f"Found {len(results_files)} results files")

    try:
        # Create visualizer
        visualizer = ResultsVisualizerV2(
            results_dir=args.results_dir,
            output_dir=args.output_dir
        )

        # Load results
        results = visualizer.load_results(args.pattern)
        logger.info(f"Loaded {len(results)} test results")

        # Create summary dashboard
        logger.info("Creating summary dashboard...")
        visualizer.create_summary_dashboard(results)

        # Create detailed reports for each test
        logger.info("Creating detailed reports...")
        visualizer.create_detailed_report(results)

        # Create comparison visualization if requested
        if args.comparison and len(results) >= 2:
            logger.info("Creating comparison visualization...")
            focus_metrics = args.metrics.split(',')
            visualizer.create_comparison_visualization(results, focus_metrics)

        logger.info(f"Visualizations saved to {visualizer.output_dir}")

    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())