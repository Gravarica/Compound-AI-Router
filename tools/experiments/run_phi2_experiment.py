#!/usr/bin/env python3
"""
Script to run the Phi-2 router experiment with OpenAI API.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    # Load environment variables from .env file
    load_dotenv()
    """Check if all required environment variables and dependencies are set."""
    print("🔍 Checking environment setup...")
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY environment variable not found!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False
    else:
        print(f"✅ OpenAI API key found (ends with: ...{openai_key[-8:]})")
    
    # Check if OpenAI package is installed
    try:
        import openai
        print("✅ OpenAI package is installed")
    except ImportError:
        print("❌ OpenAI package not found!")
        print("Please install it with: pip install openai")
        return False
    
    # Check Ollama connection
    try:
        import ollama
        models = ollama.list()
        print("✅ Ollama connection successful")
        
        # Check for required models by parsing CLI output
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        model_list = result.stdout
        
        if 'phi:latest' in model_list or 'phi' in model_list:
            print("✅ Phi model found")
        else:
            print("❌ Phi model not found! Run: ollama pull phi")
            return False
            
        if 'llama3.2:3b' in model_list:
            print("✅ Llama3.2:3b model found")
        else:
            print("❌ Llama3.2:3b model not found! Run: ollama pull llama3.2:3b")
            return False
            
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("Make sure Ollama is running with: ollama serve")
        return False
    
    # Check if results directory exists
    results_dir = "results_organized/experiments/router_types/llm"
    if not os.path.exists(results_dir):
        print(f"📁 Creating results directory: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)
    else:
        print(f"✅ Results directory exists: {results_dir}")
    
    return True

def run_experiment():
    """Run the Phi-2 router experiment."""
    print("\n🚀 Starting Phi-2 router experiment...")
    print("Configuration:")
    print("  - Router: LLM-based (Phi-2)")
    print("  - Small LLM: Llama3.2:3b (via Ollama)")
    print("  - Large LLM: GPT-4o-mini (via OpenAI API)")
    print("  - Samples: 500")
    print("  - Confidence threshold: 0.7")
    
    # Run the evaluation
    cmd = "python scripts/run_evaluation.py --config-name=phi2_router_openai"
    print(f"\n📋 Running command: {cmd}")
    
    return os.system(cmd)

def main():
    """Main function."""
    print("🤖 Phi-2 Router Experiment Setup")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above and try again.")
        sys.exit(1)
    
    print("\n✅ Environment check passed!")
    
    # Ask user confirmation
    response = input("\n🔄 Run the Phi-2 router experiment now? (y/N): ").lower().strip()
    
    if response in ['y', 'yes']:
        exit_code = run_experiment()
        
        if exit_code == 0:
            print("\n🎉 Experiment completed successfully!")
            print("📊 Results saved to: results_organized/experiments/router_types/llm/")
            print("\nNext steps:")
            print("1. Check the results JSON file")
            print("2. Run visualization: python generate_visualizations.py")
            print("3. Compare with other router types")
        else:
            print(f"\n❌ Experiment failed with exit code: {exit_code}")
            print("Check the logs above for error details.")
    else:
        print("\n👋 Experiment cancelled. Run this script again when ready!")

if __name__ == "__main__":
    main()