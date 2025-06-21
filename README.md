# Compound AI Router System

A comprehensive system for intelligently routing queries between small and large Language Models (LLMs) based on predicted difficulty, optimizing for cost-efficiency while maintaining accuracy.

## 🎯 System Overview

This project implements a **Compound AI Router** that dynamically selects between different LLMs based on query difficulty prediction. The system aims to reduce computational costs and API expenses by using smaller, faster models for easier queries while routing complex queries to larger, more capable models.

### Core Hypothesis
- **Small models** (1B-3B parameters) can handle "easy" queries effectively at lower cost
- **Large models** (Claude, GPT-4) are necessary for "hard" queries but expensive
- **Intelligent routing** can achieve near-large-model accuracy at significantly reduced cost
  
## How to Run

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export ANTHROPIC_API_KEY="your_claude_api_key"
export OPENAI_API_KEY="your_openai_api_key"  # Optional
export OLLAMA_HOST="http://localhost:11434"  # Default

# Start Ollama service with required models
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull phi:latest
```

### Running Individual Model Baselines

```bash
# Create results directories
mkdir -p results/baselines/{llama3_2_1b,llama3_2_3b,claude_haiku,phi}

# Run baseline evaluations with cost tracking
python scripts/run_evaluation.py --config-name=baseline_llama3_2_1b
python scripts/run_evaluation.py --config-name=baseline_llama3_2_3b
python scripts/run_evaluation.py --config-name=baseline_claude_haiku
python scripts/run_evaluation.py --config-name=baseline_phi

# Results saved to: results/baselines/{model_name}/baseline_{model_name}_results_full.json
```

### Running Compound AI System Evaluations

```bash
python scripts/run_evaluation.py --config-name=[config]
python scripts/run_evaluation.py --config-name=compound_transformer_gpt_gwen
```

## 🔧 Configuration System

The project uses **Hydra** for configuration management:

- **Base config**: `configs/default.yaml`
- **Evaluation configs**: `configs/evaluation/`
  - Individual baselines: `baseline_{model_name}.yaml`
  - Compound systems: `phi2_router_claude.yaml`, `cot_router_test.yaml`, etc.
  - Comparisons: `random_router.yaml`, `oracle.yaml`

### Key Configuration Parameters
```yaml
run_mode: "baseline" | "compound"           # Evaluation mode
router:
  type: "transformer"                       # Router approach
  confidence_threshold: 0.8                 # Routing decision threshold
small_llm:
  type: "ollama"                           # Local model provider
  model_name: "llama3.2:3b"               # Specific model
large_llm:
  type: "claude"                           # API provider
  model_name: "claude-3-haiku-20240307"   # Specific model
evaluation:
  num_samples: 500                         # Evaluation dataset size
  seed: 42                                 # Reproducibility seed
``
