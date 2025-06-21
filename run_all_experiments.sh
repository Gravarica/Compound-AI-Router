#!/bin/bash

# Comprehensive Experiment Runner
# Runs all baseline and compound AI experiments with new small models

echo "=== COMPOUND AI ROUTER EXPERIMENT SUITE ==="
echo "Testing all combinations of small and large models"
echo ""

# Function to run experiment with error handling
run_experiment() {
    local config_name=$1
    local description=$2
    
    echo "----------------------------------------"
    echo "Running: $description"
    echo "Config: $config_name"
    echo "----------------------------------------"
    
    python scripts/run_evaluation.py --config-name="$config_name"
    
    if [ $? -eq 0 ]; then
        echo "✅ SUCCESS: $description"
    else
        echo "❌ FAILED: $description"
    fi
    echo ""
}

# BASELINE EVALUATIONS
echo "🔍 PHASE 1: BASELINE EVALUATIONS"
echo ""

run_experiment "baseline_claude_haiku" "Claude Haiku Baseline"

# COMPOUND AI EVALUATIONS - GPT-4o-mini Large Model
echo "🚀 PHASE 2: COMPOUND AI WITH GPT-4o-mini"
echo ""

run_experiment "compound_transformer_gpt_3b" "Llama3.2 3B → GPT-4o-mini"
run_experiment "compound_transformer_gpt_qwen" "Qwen2.5 1.5B → GPT-4o-mini"

# COMPOUND AI EVALUATIONS - Claude Haiku Large Model
echo "🧠 PHASE 3: COMPOUND AI WITH CLAUDE HAIKU"
echo ""

run_experiment "compound_transformer_claude_3b" "Llama3.2 3B → Claude Haiku"
run_experiment "compound_transformer_claude_qwen" "Qwen2.5 1.5B → Claude Haiku"

echo "=== EXPERIMENT SUITE COMPLETE ==="
echo ""
echo "📊 RESULTS SUMMARY:"
echo "• Baseline results: results/baselines/"
echo "• Compound results: results/experiments/compound/transformer_router/"
echo ""
echo "📈 ANALYSIS COMMANDS:"
echo "python tools/analysis/router_performance_analyzer.py"
echo "python tools/analysis/router_comparison_summary.py"