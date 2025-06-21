#!/usr/bin/env python3
"""
Comprehensive codebase reorganization script.
Consolidates all analysis tools and organizes results properly.
"""
import os
import shutil
from pathlib import Path

def reorganize_codebase():
    """Reorganize the entire codebase structure."""
    
    base_dir = Path(".")
    
    print("🔄 REORGANIZING COMPOUND AI ROUTER CODEBASE")
    print("=" * 50)
    
    # 1. Create new directory structure
    new_dirs = [
        "tools/analysis",
        "tools/experiments", 
        "tools/organization",
        "tools/visualization",
        "results/experiments/router_types/llm",
        "results/experiments/router_types/transformer",
        "results/experiments/router_types/oracle",
        "results/experiments/router_types/random",
        "results/experiments/model_combinations/llama3_claude",
        "results/experiments/model_combinations/phi2_claude",
        "results/experiments/model_combinations/stress_1b_claude",
        "results/baselines/large_llm_only",
        "results/baselines/small_llm_only",
        "results/analysis/calibration",
        "results/analysis/threshold_sensitivity",
        "results/analysis/cost_efficiency",
        "results/visualizations/comparisons",
        "results/visualizations/dashboards",
        "results/visualizations/individual_plots"
    ]
    
    for dir_path in new_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    # 2. Move analysis tools (already done, but document here)
    tool_moves = {
        "analyze_router_performance.py": "tools/analysis/router_performance_analyzer.py",
        "router_comparison_summary.py": "tools/analysis/router_comparison_summary.py", 
        "run_phi2_experiment.py": "tools/experiments/",
        "organize_results.py": "tools/organization/"
    }
    
    print(f"\n📁 TOOLS REORGANIZATION:")
    for old_path, new_path in tool_moves.items():
        if Path(old_path).exists():
            print(f"  {old_path} → {new_path}")
        else:
            print(f"  ✅ {old_path} already moved to {new_path}")
    
    # 3. Move visualization tools
    viz_tools = [
        "generate_visualizations.py",
        "visualize.py"
    ]
    
    for tool in viz_tools:
        if Path(tool).exists():
            shutil.move(tool, f"tools/visualization/{tool}")
            print(f"  📊 {tool} → tools/visualization/")
    
    # 4. Consolidate baseline results
    baseline_sources = [
        "results/baseline_large_llm/baseline_large_llm_results_full.json",
        "results/compound_oracle_router/compound_oracle_router_results_full.json",
        "results/compound_random_router/compound_random_router_results_full.json"
    ]
    
    print(f"\n📊 CONSOLIDATING BASELINE RESULTS:")
    for source in baseline_sources:
        if Path(source).exists():
            filename = Path(source).name
            dest = f"results/baselines/large_llm_only/{filename}"
            shutil.copy2(source, dest)
            print(f"  ✅ {source} → {dest}")
    
    # 5. Move experimental results to proper categories
    print(f"\n🧪 ORGANIZING EXPERIMENTAL RESULTS:")
    
    # Router type results
    router_results = {
        "compound_system": "transformer",
        "compound_oracle_router": "oracle", 
        "compound_random_router": "random"
    }
    
    for source_dir, router_type in router_results.items():
        source_path = Path(f"results/{source_dir}")
        if source_path.exists():
            dest_path = Path(f"results/experiments/router_types/{router_type}")
            
            for file in source_path.glob("*.json"):
                shutil.copy2(file, dest_path / file.name)
                print(f"  📋 {file} → {dest_path}/")
    
    # 6. Create README files for each directory
    readme_content = {
        "tools/README.md": """# Tools Directory

This directory contains analysis, experimentation, and utility tools for the Compound AI Router project.

## Structure:
- `analysis/`: Router performance analysis and comparison tools
- `experiments/`: Experiment running scripts and configurations
- `organization/`: Codebase organization and maintenance tools
- `visualization/`: Plotting and dashboard generation tools
""",
        "results/README.md": """# Results Directory

Organized experimental results and analysis outputs.

## Structure:
- `experiments/`: All experimental results organized by router type and model combinations
- `baselines/`: Baseline performance results (large LLM only, small LLM only)
- `analysis/`: Analysis outputs (calibration, threshold sensitivity, cost efficiency)
- `visualizations/`: Generated plots, dashboards, and visual analyses
""",
        "tools/analysis/README.md": """# Analysis Tools

## Scripts:
- `router_performance_analyzer.py`: Analyzes router effectiveness and impact
- `router_comparison_summary.py`: Compares different router approaches
- `experiment_dashboard.py`: Interactive dashboard for experiment results
""",
        "tools/experiments/README.md": """# Experiment Tools

## Scripts:
- `run_phi2_experiment.py`: Runs Phi-2 router experiments with various configurations
- `stress_test_runner.py`: Runs stress tests with different model combinations
"""
    }
    
    print(f"\n📚 CREATING DOCUMENTATION:")
    for readme_path, content in readme_content.items():
        with open(readme_path, 'w') as f:
            f.write(content)
        print(f"  📝 Created: {readme_path}")
    
    # 7. Update import paths in moved files
    print(f"\n🔧 UPDATING IMPORT PATHS:")
    update_import_paths()
    
    print(f"\n✅ REORGANIZATION COMPLETE!")
    print(f"📁 New structure created with organized tools and results")
    print(f"🧹 Legacy directories can be cleaned up manually if needed")

def update_import_paths():
    """Update import paths in moved files."""
    
    # Update router_performance_analyzer.py
    analyzer_path = "tools/analysis/router_performance_analyzer.py"
    if Path(analyzer_path).exists():
        with open(analyzer_path, 'r') as f:
            content = f.read()
        
        # Update results path
        content = content.replace(
            'results_file = "results_organized/experiments/router_types/llm/phi2_router_llama3.2_claude_results_full.json"',
            'results_file = "results/experiments/router_types/llm/phi2_router_llama3.2_claude_results_full.json"'
        )
        
        with open(analyzer_path, 'w') as f:
            f.write(content)
        print(f"  🔧 Updated paths in {analyzer_path}")
    
    # Update router_comparison_summary.py
    comparison_path = "tools/analysis/router_comparison_summary.py"
    if Path(comparison_path).exists():
        with open(comparison_path, 'r') as f:
            content = f.read()
        
        # Update results directory path
        content = content.replace(
            'results_dir = Path("results_organized/experiments/router_types/llm")',
            'results_dir = Path("results/experiments/router_types/llm")'
        )
        
        with open(comparison_path, 'w') as f:
            f.write(content)
        print(f"  🔧 Updated paths in {comparison_path}")

if __name__ == "__main__":
    reorganize_codebase()