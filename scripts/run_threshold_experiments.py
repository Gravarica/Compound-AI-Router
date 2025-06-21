#!/usr/bin/env python3
"""
Router Threshold Sensitivity Experiments

Runs actual evaluation experiments with different routing thresholds
for the GPT-4o-mini + Qwen2.5-1.5B system to generate real data
for the threshold sensitivity analysis plot.
"""

import os
import yaml
import subprocess
import json
from datetime import datetime
from pathlib import Path
import time
from dotenv import load_dotenv

def create_threshold_config(base_config_path: str, threshold: float, output_dir: str) -> str:
    """Create a new config file with specified threshold"""
    
    # Load base configuration
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update threshold, model, and output paths
    config['router']['confidence_threshold'] = threshold
    config['small_llm']['model_name'] = "qwen2.5:1.5b"  # Change to Qwen
    config['experiment_name'] = f"threshold_{threshold:.2f}_gpt_qwen"
    config['evaluation']['output_dir'] = output_dir
    config['evaluation']['output_file'] = f"{output_dir}/evaluation_results.json"
    
    # Create new config file name
    config_filename = f"threshold_{threshold:.2f}_gpt_qwen.yaml"
    config_path = f"configs/evaluation/{config_filename}"
    
    # Save new config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
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
            output_file = f"results/experiments/threshold_sensitivity/threshold_{threshold:.2f}/evaluation_results_full.json"
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
                print(f"⚠️  Results file not found: {output_file}")
                return {'success': False, 'threshold': threshold, 'error': 'Results file not found'}
        else:
            print(f"❌ Experiment failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return {'success': False, 'threshold': threshold, 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment timed out after 1 hour")
        return {'success': False, 'threshold': threshold, 'error': 'Timeout'}
    except Exception as e:
        print(f"💥 Experiment crashed: {e}")
        return {'success': False, 'threshold': threshold, 'error': str(e)}

def run_threshold_sensitivity_experiments():
    """Run experiments across multiple thresholds"""
    
    print("🎯 Router Threshold Sensitivity Experiments")
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
        
        # Clean up config file
        try:
            os.remove(config_path)
        except:
            pass
        
        # Brief pause between experiments
        if i < len(thresholds):
            print("⏸️  Pausing 10 seconds between experiments...")
            time.sleep(10)
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(thresholds)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Duration: {total_duration}")
    print(f"Average per experiment: {total_duration / len(thresholds)}")
    
    # Save summary
    summary = {
        'experiment_info': {
            'system': 'GPT-4o-mini + Qwen2.5-1.5B',
            'router_type': 'transformer',
            'thresholds_tested': thresholds,
            'total_experiments': len(thresholds),
            'successful_runs': successful_runs,
            'failed_runs': failed_runs,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_seconds': total_duration.total_seconds()
        },
        'results': all_results
    }
    
    summary_file = f"{base_output_dir}/threshold_experiments_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Summary saved to: {summary_file}")
    
    # List successful results for plotting
    successful_results = [r for r in all_results if r['success']]
    if successful_results:
        print(f"\n✅ Successful experiments (ready for plotting):")
        for result in successful_results:
            threshold = result['threshold']
            output_dir = f"{base_output_dir}/threshold_{threshold:.2f}"
            print(f"  Threshold {threshold:.2f}: {output_dir}/evaluation_results.json")
    
    return all_results

def check_ollama_model(model_name: str) -> bool:
    """Check if Ollama model is available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            available_models = [m['name'] for m in models.get('models', [])]
            return model_name in available_models
        return False
    except:
        return False

def check_prerequisites():
    """Check if everything is ready for experiments"""
    
    print("🔍 Checking prerequisites...")
    
    # Check if base config exists
    base_config = "configs/evaluation/compound_transformer_gpt_qwen.yaml"
    if not os.path.exists(base_config):
        print(f"❌ Base config not found: {base_config}")
        return False
    
    # Check if router model exists
    router_model_path = "model-store/distilbert_improved"
    if not os.path.exists(router_model_path):
        print(f"❌ Router model not found: {router_model_path}")
        print("Available models:")
        for model_dir in ["model-store/distilbert_hybrid", "model-store/minilm_enhanced", "model-store/roberta_enhanced"]:
            if os.path.exists(model_dir):
                print(f"  ✅ {model_dir}")
        return False
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check environment variables
    if not os.environ.get('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("💡 Make sure your .env file contains: OPENAI_API_KEY=your_key_here")
        return False
    
    # Check if evaluation script exists
    if not os.path.exists("scripts/run_evaluation.py"):
        print("❌ Evaluation script not found: scripts/run_evaluation.py")
        return False
    
    # Check if Ollama is running and Llama3.2:1B is available
    print("🔍 Checking Ollama models...")
    if not check_ollama_model("qwen2.5:1.5b"):
        print("❌ Qwen2.5:1.5B not found in Ollama")
        print("💡 Available models:")
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                for model in models.get('models', []):
                    print(f"  ✅ {model['name']}")
                    
                print(f"\n💡 To pull Qwen2.5:1.5B, run:")
                print(f"   ollama pull qwen2.5:1.5b")
                return False
            else:
                print("❌ Cannot connect to Ollama server")
                print("💡 Make sure Ollama is running: ollama serve")
                return False
        except:
            print("❌ Cannot connect to Ollama server")
            print("💡 Make sure Ollama is running: ollama serve")
            return False
    else:
        print("✅ Qwen2.5:1.5B model available")
    
    print("✅ All prerequisites satisfied!")
    return True

def main():
    """Main function"""
    
    # Load environment variables from .env file first
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
    
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