#!/usr/bin/env python3
"""
Router Threshold Sensitivity Experiments for GPT + Qwen
========================================================
This script runs a comprehensive threshold sensitivity analysis for the compound AI system
using GPT-4o-mini as the large model and Qwen2.5:1.5B as the small model.

Runs 8 experiments with different confidence thresholds: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_environment():
    """Load environment variables from .env file"""
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ Loaded environment variables from .env file")
        return True
    else:
        print("⚠️  No .env file found")
        return False

def check_ollama_models():
    """Check if required Ollama models are available"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            models = result.stdout.lower()
            if 'qwen2.5:1.5b' in models:
                print("✅ Qwen2.5:1.5B model available")
                return True
            else:
                print("❌ Qwen2.5:1.5B model not found. Please run: ollama pull qwen2.5:1.5b")
                return False
        else:
            print("❌ Could not connect to Ollama")
            return False
    except FileNotFoundError:
        print("❌ Ollama not found")
        return False

def check_api_keys():
    """Check if required API keys are available"""
    openai_key = os.environ.get('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    print("✅ OpenAI API key found")
    return True

def check_prerequisites():
    """Check all prerequisites for running experiments"""
    print("🔍 Checking prerequisites...")
    
    # Load environment
    if not load_environment():
        return False
    
    # Check Ollama models
    print("🔍 Checking Ollama models...")
    if not check_ollama_models():
        return False
    
    # Check API keys
    if not check_api_keys():
        return False
    
    return True

def create_threshold_config(base_config_path: str, threshold: float, output_dir: str) -> str:
    """Create a config file for a specific threshold"""
    
    # Read base config
    with open(base_config_path, 'r') as f:
        config_content = f.read()
    
    # Replace model to use Qwen instead of Llama
    config_content = config_content.replace('llama3.2:1b', 'qwen2.5:1.5b')
    
    # Replace threshold (from 0.7 to the target threshold)
    config_content = config_content.replace(
        'confidence_threshold: 0.7', 
        f'confidence_threshold: {threshold}'
    )
    
    # Replace output directory
    config_content = config_content.replace(
        'results/experiments/threshold_sensitivity/gpt_llama1b',
        output_dir
    )
    
    # Create new config file
    config_filename = f"threshold_{threshold:.2f}_gpt_qwen.yaml"
    config_path = f"configs/evaluation/{config_filename}"
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created config: {config_path}")
    return config_path

def run_evaluation_experiment(config_path: str, threshold: float) -> dict:
    """Run a single evaluation experiment"""
    
    print(f"\n{'='*60}")
    print(f"🚀 Running experiment with threshold {threshold:.2f}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the evaluation using hydra
        cmd = [
            "python", "scripts/run_evaluation.py", 
            "--config-name", Path(config_path).stem
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            end_time = time.time()
            duration = end_time - start_time
            print(f"✅ Experiment completed successfully in {duration:.1f} seconds")
            
            # Load results
            output_file = f"results/experiments/threshold_sensitivity_gpt_qwen/threshold_{threshold:.2f}/evaluation_results_full.json"
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    results = json.load(f)
                return {
                    'success': True,
                    'threshold': threshold,
                    'duration': duration,
                    'results': results
                }
            else:
                print(f"❌ Results file not found: {output_file}")
                return {
                    'success': False,
                    'threshold': threshold,
                    'duration': duration,
                    'error': 'Results file not found'
                }
        else:
            print(f"❌ Experiment failed with return code {result.returncode}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            return {
                'success': False,
                'threshold': threshold,
                'duration': time.time() - start_time,
                'error': f"Process failed: {result.stderr}"
            }
            
    except subprocess.TimeoutExpired:
        print(f"❌ Experiment timed out after 1 hour")
        return {
            'success': False,
            'threshold': threshold,
            'duration': 3600,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        return {
            'success': False,
            'threshold': threshold,
            'duration': time.time() - start_time,
            'error': str(e)
        }

def run_threshold_sensitivity_experiments():
    """Run the complete threshold sensitivity analysis"""
    
    print("🎯 Router Threshold Sensitivity Experiments - GPT + Qwen")
    print("=" * 60)
    print("System: GPT-4o-mini + Qwen2.5-1.5B with Transformer Router")
    print("=" * 60)
    
    # Configuration
    base_config = "configs/evaluation/compound_transformer_gpt_qwen.yaml"
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 8 experiments
    
    # Create output directory
    base_output_dir = "results/experiments/threshold_sensitivity_gpt_qwen"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Results storage
    all_results = []
    successful_runs = 0
    failed_runs = 0
    
    start_time = datetime.now()
    
    print(f"📋 Running {len(thresholds)} experiments")
    print(f"📊 Thresholds: {thresholds}")
    print(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for i, threshold in enumerate(thresholds, 1):
        print(f"\n🔄 Experiment {i}/{len(thresholds)}: Threshold {threshold:.2f}")
        
        # Create output directory for this threshold
        threshold_output_dir = f"{base_output_dir}/threshold_{threshold:.2f}"
        os.makedirs(threshold_output_dir, exist_ok=True)
        
        # Create config for this threshold
        config_path = create_threshold_config(base_config, threshold, threshold_output_dir)
        
        # Run experiment
        result = run_evaluation_experiment(config_path, threshold)
        all_results.append(result)
        
        if result['success']:
            successful_runs += 1
            print(f"✅ Success! ({successful_runs}/{i})")
        else:
            failed_runs += 1
            print(f"❌ Failed! ({failed_runs}/{i})")
        
        # Pause between experiments to avoid overwhelming APIs
        if i < len(thresholds):
            print("⏸️  Pausing 10 seconds between experiments...")
            time.sleep(10)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Save summary
    summary = {
        'experiment_type': 'threshold_sensitivity_gpt_qwen',
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'total_duration': str(duration),
        'total_experiments': len(thresholds),
        'successful_experiments': successful_runs,
        'failed_experiments': failed_runs,
        'thresholds': thresholds,
        'results': all_results
    }
    
    summary_file = f"{base_output_dir}/threshold_experiments_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(thresholds)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Duration: {duration}")
    print(f"Average per experiment: {duration / len(thresholds)}")
    print(f"\n📄 Summary saved to: {summary_file}")
    
    # Print successful experiment paths
    if successful_runs > 0:
        print(f"\n✅ Successful experiments (ready for plotting):")
        for result in all_results:
            if result['success']:
                threshold = result['threshold']
                result_file = f"{base_output_dir}/threshold_{threshold:.2f}/evaluation_results.json"
                print(f"  Threshold {threshold:.2f}: {result_file}")
    
    return all_results

def main():
    """Main function"""
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n💥 Prerequisites not met. Please fix the issues above.")
        return
    
    # Auto-proceed for batch experiments
    print(f"\n⚠️  Starting 8 full evaluation experiments...")
    print(f"⏰ Expected duration: ~2-4 hours (depending on API speed)")
    print(f"💰 Estimated cost: ~$5-15 (API calls)")
    print(f"🚀 Proceeding automatically...")
    
    # Run experiments
    results = run_threshold_sensitivity_experiments()
    
    print(f"\n🎉 Threshold sensitivity experiments completed!")
    print(f"📊 Ready to generate analysis plots with real data.")

if __name__ == "__main__":
    main()