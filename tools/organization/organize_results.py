#!/usr/bin/env python3
"""
Script to organize experimental results into a cleaner structure.
"""
import os
import shutil
import json
from datetime import datetime
from pathlib import Path

def create_organized_structure():
    """Create the organized results directory structure."""
    base_dir = "results_organized"
    
    # Main categories
    structure = {
        "experiments": {
            "router_types": ["transformer", "random", "llm", "oracle"],
            "model_combinations": ["llama3_claude", "llama3_gpt4", "phi2_claude", "phi2_gpt4"],
            "threshold_analysis": []
        },
        "baselines": ["large_llm_only", "small_llm_only"],
        "visualizations": ["dashboards", "individual_plots", "comparisons"],
        "analysis": ["calibration", "error_analysis", "cost_efficiency"]
    }
    
    # Create directories
    for main_cat, subcats in structure.items():
        main_path = Path(base_dir) / main_cat
        main_path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(subcats, dict):
            for subcat, items in subcats.items():
                subcat_path = main_path / subcat
                subcat_path.mkdir(exist_ok=True)
                for item in items:
                    item_path = subcat_path / item
                    item_path.mkdir(exist_ok=True)
        elif isinstance(subcats, list):
            for subcat in subcats:
                subcat_path = main_path / subcat
                subcat_path.mkdir(exist_ok=True)
    
    print(f"Created organized structure in {base_dir}/")

def get_experiment_metadata(result_file):
    """Extract metadata from a result file to categorize it."""
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract key information
        metadata = {}
        
        # Check if it's a compound system result
        if 'compound_results' in data and 'test_config' in data:
            config = data['test_config']
            metadata['type'] = 'compound'
            metadata['router_type'] = 'transformer'  # Default, could be extracted from config
            metadata['small_model'] = config.get('small_model', 'unknown')
            metadata['large_model'] = config.get('large_model', 'unknown')
            metadata['large_model_type'] = config.get('large_model_type', 'unknown')
            metadata['num_samples'] = config.get('num_samples', 0)
            
        elif 'comparison' in data:
            metadata['type'] = 'comparison'
            # Could extract more details from comparison data
            
        else:
            metadata['type'] = 'baseline'
            
        return metadata
        
    except Exception as e:
        print(f"Error reading {result_file}: {e}")
        return None

def organize_existing_results():
    """Organize existing results into the new structure."""
    results_dir = "results"
    organized_dir = "results_organized"
    
    if not os.path.exists(results_dir):
        print("No existing results directory found.")
        return
    
    # Process files in results directory
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json'):
                source_path = os.path.join(root, file)
                
                # Get relative path from results
                rel_path = os.path.relpath(root, results_dir)
                
                # Determine destination based on file content and path
                metadata = get_experiment_metadata(source_path)
                
                if metadata:
                    if metadata['type'] == 'compound':
                        # Organize by router type and model combination
                        router_type = metadata.get('router_type', 'transformer')
                        small_model = metadata.get('small_model', 'unknown')
                        large_model_type = metadata.get('large_model_type', 'unknown')
                        
                        dest_dir = f"{organized_dir}/experiments/router_types/{router_type}"
                        
                    elif metadata['type'] == 'baseline':
                        dest_dir = f"{organized_dir}/baselines/large_llm_only"
                        
                    else:
                        dest_dir = f"{organized_dir}/experiments/router_types/transformer"
                else:
                    # Default categorization based on path
                    if 'baseline' in rel_path.lower():
                        dest_dir = f"{organized_dir}/baselines/large_llm_only"
                    elif 'random' in rel_path.lower():
                        dest_dir = f"{organized_dir}/experiments/router_types/random"
                    elif 'oracle' in rel_path.lower():
                        dest_dir = f"{organized_dir}/experiments/router_types/oracle"
                    else:
                        dest_dir = f"{organized_dir}/experiments/router_types/transformer"
                
                # Create destination directory and copy file
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, file)
                
                # Add timestamp if file exists
                if os.path.exists(dest_path):
                    name, ext = os.path.splitext(file)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dest_path = os.path.join(dest_dir, f"{name}_{timestamp}{ext}")
                
                shutil.copy2(source_path, dest_path)
                print(f"Moved: {source_path} -> {dest_path}")
    
    # Handle visualizations
    viz_dirs = ["visualizations", "visualizations_output"]
    for viz_dir in viz_dirs:
        viz_path = os.path.join(results_dir, viz_dir)
        if os.path.exists(viz_path):
            dest_viz_path = os.path.join(organized_dir, "visualizations", viz_dir)
            if os.path.exists(dest_viz_path):
                shutil.rmtree(dest_viz_path)
            shutil.copytree(viz_path, dest_viz_path)
            print(f"Copied visualization directory: {viz_path} -> {dest_viz_path}")

def create_experiment_naming_convention():
    """Create a naming convention guide for future experiments."""
    guide = """
# Experiment Naming Convention

## File naming format:
{router_type}_{small_model}_{large_model}_{date}_{config}.json

Examples:
- transformer_llama3.2_claude_20241218_default.json
- random_llama3.2_gpt4_20241218_baseline.json
- llm_phi2_claude_20241218_thresh08.json

## Directory structure:
results_organized/
├── experiments/
│   ├── router_types/
│   │   ├── transformer/     # DistilBERT-based router
│   │   ├── random/          # Random baseline router
│   │   ├── llm/             # LLM-based router (Phi-2, etc.)
│   │   └── oracle/          # Perfect router (upper bound)
│   ├── model_combinations/
│   │   ├── llama3_claude/   # Llama3.2 + Claude
│   │   ├── llama3_gpt4/     # Llama3.2 + GPT-4
│   │   ├── phi2_claude/     # Phi-2 + Claude
│   │   └── phi2_gpt4/       # Phi-2 + GPT-4
│   └── threshold_analysis/  # Different confidence thresholds
├── baselines/
│   ├── large_llm_only/      # Single large model baselines
│   └── small_llm_only/      # Single small model baselines
├── visualizations/
│   ├── dashboards/          # Summary dashboards
│   ├── individual_plots/    # Individual metric plots
│   └── comparisons/         # Cross-experiment comparisons
└── analysis/
    ├── calibration/         # Router calibration analysis
    ├── error_analysis/      # Error pattern analysis
    └── cost_efficiency/     # Cost-benefit analysis
"""
    
    with open("results_organized/README.md", "w") as f:
        f.write(guide)
    
    print("Created naming convention guide in results_organized/README.md")

if __name__ == "__main__":
    print("Organizing experimental results...")
    
    # Create organized structure
    create_organized_structure()
    
    # Organize existing results
    organize_existing_results()
    
    # Create naming convention guide
    create_experiment_naming_convention()
    
    print("\nResults organization complete!")
    print("Check results_organized/README.md for naming conventions.")