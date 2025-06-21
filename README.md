# Compound AI Router System

A comprehensive system for intelligently routing queries between small and large Language Models (LLMs) based on predicted difficulty, optimizing for cost-efficiency while maintaining accuracy.

## 🎯 System Overview

This project implements a **Compound AI Router** that dynamically selects between different LLMs based on query difficulty prediction. The system aims to reduce computational costs and API expenses by using smaller, faster models for easier queries while routing complex queries to larger, more capable models.

### Core Hypothesis
- **Small models** (1B-3B parameters) can handle "easy" queries effectively at lower cost
- **Large models** (Claude, GPT-4) are necessary for "hard" queries but expensive
- **Intelligent routing** can achieve near-large-model accuracy at significantly reduced cost

## 🏗️ Architecture Overview

### 1. **Orchestration Layer** (`src/orchestration/`)
The main coordination layer that manages the entire query processing pipeline:

- **`CompoundAIOrchestrator`**: Main orchestrator coordinating router + LLMs
- **`QueryProcessor`**: Prepares prompts for both router classification and LLM inference
- **`ResponseParser`**: Parses LLM responses and evaluates correctness against ground truth
- **`MetricsCollector`**: Tracks timing, resource usage, and performance metrics
- **`routing_strategies.py`**: Implements routing logic (threshold-based, confidence-based)

### 2. **Routing Layer** (`src/routing/`)
Different approaches to difficulty classification and routing decisions:

- **`BaseRouter`**: Abstract interface for all routing implementations
- **`TransformerRouter`**: ML-based router using fine-tuned DistilBERT for difficulty classification
- **`LLMRouter`**: Uses smaller LLMs (Phi-2) to classify query difficulty via prompting
- **`ChainOfThoughtLLMRouter`**: CoT-enhanced router with step-by-step reasoning
- **`RandomRouter`**: Random routing baseline for comparison
- **`OracleRouter`**: Perfect routing using ground truth (theoretical upper bound)
- **`RouterFactory`**: Factory pattern for creating router instances from configuration

### 3. **Model Layer** (`src/models/`)
Unified interfaces for different LLM providers and local models:

- **`LLMInterface`**: Abstract base class defining common LLM operations
- **`OllamaLLM`**: Interface for local models (Llama3.2:1B/3B, Phi, etc.)
- **`ClaudeLLM`**: Interface for Anthropic Claude API (Haiku, Sonnet, Opus)
- **`OpenAILLM`**: Interface for OpenAI API (GPT-4o-mini, GPT-4o, etc.)
- **`LLMFactory`**: Factory for creating LLM instances from configuration

### 4. **Cost Calculation System** (`src/utils/model_pricing.py`)
Comprehensive cost tracking and analysis framework:

- **`ModelPricingCatalog`**: Centralized pricing database for all supported models
- **`CostCalculator`**: Utilities for calculating usage costs across different scenarios
- **Real-time cost tracking** during evaluations with detailed breakdowns
- **Cost comparison** between baseline and compound system approaches

### 5. **Data Layer** (`src/data/`)
Dataset management and processing:

- **`ARCDataManager`**: Handles loading and processing of ARC (AI2 Reasoning Challenge) dataset
- **`RouterDataset`**: PyTorch dataset implementation for router training

## 💰 Cost Calculation Framework

### Pricing Catalog
The system maintains up-to-date pricing for major LLM providers:

```python
PRICING_DATA = {
    "claude": {
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},  # per 1M tokens
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        # ... more models
    },
    "openai": {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        # ... more models
    },
    "ollama": {
        "llama3.2:1b": {"input": 0.0, "output": 0.0, "compute_cost_per_hour": 0.10},
        # Local models track compute costs instead of API costs
    }
}
```

### Cost Calculation Features
- **Per-query cost tracking**: Individual query costs with token usage breakdown
- **Aggregate cost analysis**: Total costs across evaluation runs
- **Cost comparison**: Baseline vs compound system cost efficiency
- **Multi-provider support**: API costs (Claude, OpenAI) + compute costs (local models)
- **Real-time reporting**: Cost summaries logged during evaluation runs

### Cost Metrics Calculated
- **Input/Output token costs**: Based on actual API pricing
- **Compute costs**: For local models based on runtime
- **Cost per correct answer**: Efficiency metric combining accuracy and cost
- **Cost savings**: Absolute and percentage savings from routing
- **Cost-efficiency ratio**: Compound system vs baseline cost comparison

## 🧪 Experimental Flow

### Phase 1: Individual Model Baselines
**Goal**: Establish performance ceiling and floor for each model independently

```bash
# Run individual model evaluations
python scripts/run_evaluation.py --config-name=baseline_llama3_2_1b
python scripts/run_evaluation.py --config-name=baseline_llama3_2_3b  
python scripts/run_evaluation.py --config-name=baseline_claude_haiku
python scripts/run_evaluation.py --config-name=baseline_phi
```

**Metrics Collected**:
- Accuracy on ARC dataset (500 samples)
- Average response time per query
- Token usage (input/output)
- Total cost per evaluation run
- Cost per correct answer

### Phase 2: Router Effectiveness Analysis
**Goal**: Test different routing approaches and measure routing accuracy

```bash
# Test different router types
python scripts/run_evaluation.py --config-name=phi2_router_claude      # LLM-based router
python scripts/run_evaluation.py --config-name=cot_router_test         # Chain-of-thought router
python scripts/run_evaluation.py --config-name=compound                # DistilBERT router
python scripts/run_evaluation.py --config-name=random_router           # Random baseline
```

**Metrics Collected**:
- Router accuracy (difficulty prediction correctness)
- System accuracy (final answer correctness)
- Small/large LLM usage distribution
- Routing overhead (time/cost)
- False positive/negative rates

### Phase 3: Cost-Performance Trade-off Analysis
**Goal**: Optimize the balance between accuracy and cost efficiency

**Key Questions**:
1. At what performance gap does routing become valuable?
2. What's the optimal confidence threshold for each model pair?
3. How do routing mistakes impact overall cost-effectiveness?

## 📊 What We Calculate

### Performance Metrics
- **System Accuracy**: Overall correctness on evaluation dataset
- **Router Accuracy**: Percentage of correct difficulty classifications
- **Model Usage Distribution**: Ratio of queries routed to small vs large models
- **Latency**: End-to-end query processing time including routing overhead

### Cost Metrics
- **Total Cost**: Complete evaluation run cost including all API calls
- **Cost per Query**: Average cost per processed query
- **Cost per Correct Answer**: Efficiency metric (total cost / correct answers)
- **Cost Savings**: Absolute and percentage savings vs baseline approaches
- **ROI Analysis**: Cost-benefit analysis of routing vs always using large models

### Router Analysis
- **Confusion Matrix**: True/false positives and negatives for difficulty prediction
- **Precision/Recall**: Router performance on identifying hard queries
- **Calibration**: How well router confidence scores align with actual difficulty
- **Error Impact**: Cost of routing mistakes (easy→large, hard→small)

## 🚀 How to Run Everything

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

### Running Compound System Evaluations

```bash
# Router-based evaluations
python scripts/run_evaluation.py --config-name=phi2_router_claude
python scripts/run_evaluation.py --config-name=cot_router_test
python scripts/run_evaluation.py --config-name=compound

# Baseline comparisons
python scripts/run_evaluation.py --config-name=random_router
python scripts/run_evaluation.py --config-name=baseline_large
```

### Analyzing Results

```bash
# Router performance analysis
python tools/analysis/router_performance_analyzer.py

# Cost-effectiveness comparison
python tools/analysis/router_comparison_summary.py

# Interactive experiment dashboard
python tools/analysis/experiment_dashboard.py --command report

# Generate publication-ready visualizations
python tools/visualization/generate_visualizations.py
```

## 📁 Results Structure

```
results/
├── baselines/                    # Individual model performance
│   ├── llama3_2_1b/             # Llama3.2 1B baseline results + costs
│   ├── llama3_2_3b/             # Llama3.2 3B baseline results + costs
│   ├── claude_haiku/            # Claude Haiku baseline results + costs
│   └── phi/                     # Phi model baseline results + costs
├── experiments/                  # Compound system experiments
│   ├── router_types/            # Results by router approach
│   │   ├── llm/                 # LLM-based routers (Phi-2, CoT)
│   │   ├── transformer/         # DistilBERT router results
│   │   ├── oracle/              # Perfect router upper bound
│   │   └── random/              # Random routing baseline
│   └── model_combinations/      # Results by small/large model pairs
└── analysis/                    # Analysis and comparison reports
    ├── cost_efficiency/         # Cost-benefit analysis
    ├── calibration/             # Router calibration studies
    └── threshold_sensitivity/   # Threshold optimization results
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
  type: "transformer" | "llm" | "cot_llm"   # Router approach
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
```

## 📈 Current Experimental Status

### Completed Experiments
- ✅ Phi-2 router with Llama3.2 3B + Claude: 84% accuracy, 50% router accuracy
- ✅ Chain-of-thought router: 90% accuracy, 36% router accuracy (conservative)
- ✅ Stress test (1B vs Claude): 52% accuracy showing clear router impact

### Key Insights Discovered
1. **Router effectiveness increases with model performance gaps**
2. **Conservative routing** (CoT) trades router accuracy for system accuracy
3. **Cost calculation** reveals true efficiency beyond just accuracy metrics
4. **~50% router accuracy** leaves significant improvement potential

### Next Priority Experiments
1. **Individual model baselines** to establish performance gaps
2. **Performance gap analysis** to identify optimal model pairings
3. **Cost-optimized routing** strategies based on cost/accuracy trade-offs
4. **Router calibration** to improve confidence threshold selection

## 🎓 Academic Applications

This system is designed for research and publication in:
- **Cost-efficient AI systems**
- **Multi-model orchestration**
- **Difficulty-based query routing**
- **LLM cost optimization strategies**

The comprehensive cost tracking and experimental framework provide the quantitative foundation needed for academic analysis and comparison with other approaches.

---

**Repository Structure**: Modular, factory-pattern based architecture for easy extension and experimentation
**Cost Tracking**: Real-time cost calculation with detailed breakdowns across all supported models
**Evaluation Framework**: Standardized evaluation pipeline with consistent metrics and reproducible results
**Analysis Tools**: Comprehensive analysis and visualization tools for research publication