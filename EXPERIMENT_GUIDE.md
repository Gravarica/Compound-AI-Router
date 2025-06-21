# Experiment Configuration Guide

## Available Experiments

### Baseline Evaluations
Test individual models on ARC dataset to establish performance benchmarks.

```bash
# Small Models
python scripts/run_evaluation.py --config-name=baseline_gemma2_2b
python scripts/run_evaluation.py --config-name=baseline_qwen2_5_1_5b  
python scripts/run_evaluation.py --config-name=baseline_phi

# Previously completed baselines:
# - baseline_llama3_2_3b (75.4% accuracy, $0.000015/query)
# - baseline_openai_gpt4o_mini (93.2% accuracy, $0.000015/query)
```

### Compound AI Evaluations
Test transformer router with different small/large model combinations.

#### GPT-4o-mini as Large Model
```bash
python scripts/run_evaluation.py --config-name=compound_transformer_gpt_3b        # Llama3.2 3B → GPT
python scripts/run_evaluation.py --config-name=compound_transformer_gpt_gemma2   # Gemma2 2B → GPT  
python scripts/run_evaluation.py --config-name=compound_transformer_gpt_qwen     # Qwen2.5 1.5B → GPT
python scripts/run_evaluation.py --config-name=compound_transformer_gpt_phi      # Phi-2 → GPT
```

#### Claude Haiku as Large Model  
```bash
python scripts/run_evaluation.py --config-name=compound_transformer_claude_3b      # Llama3.2 3B → Claude
python scripts/run_evaluation.py --config-name=compound_transformer_claude_gemma2 # Gemma2 2B → Claude
python scripts/run_evaluation.py --config-name=compound_transformer_claude_qwen   # Qwen2.5 1.5B → Claude  
python scripts/run_evaluation.py --config-name=compound_transformer_claude_phi    # Phi-2 → Claude
```

## Run All Experiments
```bash
./run_all_experiments.sh
```

## Model Specifications

| Model | Size | API Cost | Expected Performance |
|-------|------|----------|---------------------|
| Qwen2.5 1.5B | 986MB | $0.03/M tokens | Most cost-effective |
| Gemma2 2B | 1.6GB | $0.04/M tokens | Google efficiency |
| Phi-2 | 1.6GB | $0.05/M tokens | Microsoft reasoning |
| Llama3.2 3B | 2.0GB | $0.06/M tokens | Meta performance |
| GPT-4o-mini | API | $0.15/M tokens | OpenAI accuracy |
| Claude Haiku | API | $0.25/M tokens | Anthropic speed |

## Expected Results Structure
```
results/
├── baselines/
│   ├── gemma2_2b/
│   ├── qwen2_5_1_5b/
│   └── phi/
└── experiments/compound/transformer_router/
    ├── gpt_gemma2/
    ├── gpt_qwen/
    ├── gpt_phi/
    ├── claude_gemma2/
    ├── claude_qwen/
    └── claude_phi/
```

## Analysis Tools
```bash
# Router performance analysis
python tools/analysis/router_performance_analyzer.py

# Compare different router approaches  
python tools/analysis/router_comparison_summary.py

# Interactive dashboard
python tools/analysis/experiment_dashboard.py --command report
```

## Key Metrics Tracked
- **Accuracy**: Percentage of correct answers
- **Latency**: Average response time in milliseconds  
- **Cost**: Total cost and cost per query/correct answer
- **Router Performance**: Small vs large model usage ratios