# src/utils/model_pricing.py
"""
Model pricing catalog and cost calculation utilities.
All prices are per 1M tokens (input/output) in USD.
Updated as of December 2024.
"""

from typing import Dict, Any, Optional
import re

class ModelPricingCatalog:
    """Centralized pricing information for various LLM providers."""
    
    # All prices in USD per 1M tokens
    PRICING_DATA = {
        # Anthropic Models
        "claude": {
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
            "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
            "claude-3-5-sonnet-.20241022": {"input": 3.00, "output": 15.00},
            "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
        },
        
        # OpenAI Models
        "openai": {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
        },
        
        # Google Models (approximate pricing)
        "google": {
            "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-pro": {"input": 0.50, "output": 1.50},
        },
        
        # Local models (Ollama) - using realistic API pricing for fair comparison
        "ollama": {
            "llama3.2:3b": {"input": 0.06, "output": 0.06, "compute_cost_per_hour": 0.0},
            "llama3.1:8b": {"input": 0.0, "output": 0.0, "compute_cost_per_hour": 0.0},
            "llama3": {"input": 0.0, "output": 0.0, "compute_cost_per_hour": 0.15},
            "phi:latest": {"input": 0.05, "output": 0.05, "compute_cost_per_hour": 0.0},  # Phi-2 pricing estimate
            "phi": {"input": 0.05, "output": 0.05, "compute_cost_per_hour": 0.0},
            "phi3": {"input": 0.05, "output": 0.05, "compute_cost_per_hour": 0.0},
            "gemma2:2b": {"input": 0.04, "output": 0.04, "compute_cost_per_hour": 0.0},  # Gemma 2B pricing estimate
            "gemma:2b": {"input": 0.04, "output": 0.04, "compute_cost_per_hour": 0.0},
            "gemma:7b": {"input": 0.0, "output": 0.0, "compute_cost_per_hour": 0.20},
            "qwen2.5:1.5b": {"input": 0.03, "output": 0.03, "compute_cost_per_hour": 0.0},  # Qwen pricing estimate
            "mistral": {"input": 0.0, "output": 0.0, "compute_cost_per_hour": 0.18},
        }
    }
    
    @classmethod
    def get_model_pricing(cls, model_name: str, provider: str = None) -> Optional[Dict[str, float]]:
        """
        Get pricing information for a specific model.
        
        Args:
            model_name: Name of the model (e.g., 'gpt-4o-mini', 'claude-3-haiku-20240307')
            provider: Provider name (e.g., 'openai', 'claude'). If None, will try to infer.
            
        Returns:
            Dictionary with 'input' and 'output' prices per 1M tokens, or None if not found
        """
        if provider is None:
            provider = cls._infer_provider(model_name)
        
        if provider in cls.PRICING_DATA and model_name in cls.PRICING_DATA[provider]:
            return cls.PRICING_DATA[provider][model_name]
        
        return None
    
    @classmethod
    def _infer_provider(cls, model_name: str) -> str:
        """Infer provider from model name."""
        if any(claude_model in model_name.lower() for claude_model in ["claude", "haiku", "sonnet", "opus"]):
            return "claude"
        elif any(gpt_model in model_name.lower() for gpt_model in ["gpt", "turbo"]):
            return "openai"
        elif any(gemini_model in model_name.lower() for gemini_model in ["gemini"]):
            return "google"
        elif any(local_model in model_name.lower() for local_model in ["llama", "phi", "gemma", "mistral"]):
            return "ollama"
        else:
            return "unknown"
    
    @classmethod
    def get_all_models(cls) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get all available models and their pricing."""
        return cls.PRICING_DATA
    
    @classmethod
    def add_custom_pricing(cls, provider: str, model_name: str, input_price: float, output_price: float):
        """Add custom pricing for a model."""
        if provider not in cls.PRICING_DATA:
            cls.PRICING_DATA[provider] = {}
        cls.PRICING_DATA[provider][model_name] = {"input": input_price, "output": output_price}


class CostCalculator:
    """Utility class for calculating LLM usage costs."""
    
    def __init__(self, pricing_catalog: ModelPricingCatalog = None):
        self.catalog = pricing_catalog or ModelPricingCatalog()
    
    def calculate_cost(self, 
                      model_name: str, 
                      input_tokens: int, 
                      output_tokens: int,
                      provider: str = None,
                      runtime_hours: float = None) -> Dict[str, Any]:
        """
        Calculate cost for a single model usage.
        
        Args:
            model_name: Name of the model (can be "provider/model" or just "model")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: Provider name (optional)
            runtime_hours: Runtime in hours for local models (optional)
            
        Returns:
            Dictionary with cost breakdown
        """
        original_model_name = model_name
        
        # Parse provider/model_name format
        if "/" in model_name and provider is None:
            provider, model_name = model_name.split("/", 1)

        pricing = self.catalog.get_model_pricing(model_name, provider)
        
        if not pricing:
            return {
                "model_name": original_model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "compute_cost": 0.0,
                "total_cost": 0.0,
                "error": f"Pricing not available for model: {model_name} (provider: {provider})"
            }
        
        # Calculate token costs
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        # Calculate compute cost for local models
        compute_cost = 0.0
        if runtime_hours and 'compute_cost_per_hour' in pricing:
            compute_cost = runtime_hours * pricing['compute_cost_per_hour']
        
        total_cost = input_cost + output_cost
        
        return {
            "model_name": original_model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "compute_cost": round(compute_cost, 6),
            "total_cost": round(total_cost, 6),
            "pricing_per_1m": pricing
        }
    
    def calculate_baseline_cost(self, results: list, model_name: str = None) -> Dict[str, Any]:
        """
        Calculate total cost for baseline evaluation results.
        
        Args:
            results: List of evaluation results
            model_name: Override model name if not in results
            
        Returns:
            Cost summary dictionary with accuracy, latency, and cost metrics
        """
        total_input_tokens = 0
        total_output_tokens = 0
        total_runtime_ms = 0
        correct_answers = 0
        
        for result in results:
            # Extract model name from first result if not provided
            if model_name is None:
                model_name = result.get('model_name', result.get('model', 'unknown'))
            
            # Extract token usage
            usage = result.get('resource_usage', {})
            total_input_tokens += usage.get('prompt_tokens', 0)
            total_output_tokens += usage.get('completion_tokens', 0)
            
            # Extract timing - try multiple possible field names
            result_time = (result.get('total_time_ms', 0) or 
                          result.get('latency_ms', 0) or
                          usage.get('latency_ms', 0) or 0)
            total_runtime_ms += result_time
            
            # Count correct answers
            if result.get('correct', False):
                correct_answers += 1
        
        # Convert runtime to hours for compute cost calculation
        runtime_hours = total_runtime_ms / (1000 * 3600)
        
        # Calculate cost
        cost_breakdown = self.calculate_cost(
            model_name=model_name,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            runtime_hours=runtime_hours
        )
        
        # Add summary metrics
        total_queries = len(results)
        accuracy = correct_answers / total_queries if total_queries > 0 else 0.0
        avg_latency_ms = total_runtime_ms / total_queries if total_queries > 0 else 0.0
        cost_per_query = cost_breakdown['total_cost'] / total_queries if total_queries > 0 else 0.0
        cost_per_correct = cost_breakdown['total_cost'] / correct_answers if correct_answers > 0 else float('inf')
        
        # Enhanced return with all three key metrics
        return {
            **cost_breakdown,  # Include all cost details
            'summary_metrics': {
                'accuracy': round(accuracy, 4),
                'avg_latency_ms': round(avg_latency_ms, 2),
                'total_cost': cost_breakdown['total_cost'],
                'cost_per_query': round(cost_per_query, 6),
                'cost_per_correct_answer': round(cost_per_correct, 6) if cost_per_correct != float('inf') else None
            },
            'evaluation_stats': {
                'total_queries': total_queries,
                'correct_answers': correct_answers,
                'total_runtime_ms': total_runtime_ms,
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens
            }
        }
    
    def calculate_compound_cost(self, results: list) -> Dict[str, Any]:
        """
        Calculate total cost for compound system results.
        
        Args:
            results: List of compound evaluation results
            
        Returns:
            Cost breakdown by model and total
        """
        model_costs = {}
        total_cost = 0.0
        
        for result in results:
            resource_usage = result.get('resource_usage', {})
            
            # Router cost (if applicable)
            if 'router' in resource_usage:
                router_usage = resource_usage['router']
                router_cost = self.calculate_cost(
                    model_name='router_overhead',
                    input_tokens=router_usage.get('prompt_tokens', 0),
                    output_tokens=router_usage.get('completion_tokens', 0)
                )
                if 'router' not in model_costs:
                    model_costs['router'] = {'input_tokens': 0, 'output_tokens': 0, 'total_cost': 0.0}
                model_costs['router']['input_tokens'] += router_cost['input_tokens']
                model_costs['router']['output_tokens'] += router_cost['output_tokens']
                model_costs['router']['total_cost'] += router_cost['total_cost']
            
            # LLM cost
            if 'llm' in resource_usage:
                llm_usage = resource_usage['llm']
                model_name = result.get('chosen_llm_name', 'unknown')
                
                llm_cost = self.calculate_cost(
                    model_name=model_name,
                    input_tokens=llm_usage.get('prompt_tokens', 0),
                    output_tokens=llm_usage.get('completion_tokens', 0),
                    runtime_hours=result.get('total_time_ms', 0) / (1000 * 3600)
                )
                
                if model_name not in model_costs:
                    model_costs[model_name] = {'input_tokens': 0, 'output_tokens': 0, 'total_cost': 0.0}
                
                model_costs[model_name]['input_tokens'] += llm_cost['input_tokens']
                model_costs[model_name]['output_tokens'] += llm_cost['output_tokens']
                model_costs[model_name]['total_cost'] += llm_cost['total_cost']
                
                total_cost += llm_cost['total_cost']
        
        # Calculate summary metrics similar to baseline
        total_queries = len(results)
        correct_answers = sum(1 for r in results if r.get('correct', False))
        total_runtime_ms = sum(r.get('total_time_ms', 0) for r in results)
        
        accuracy = correct_answers / total_queries if total_queries > 0 else 0.0
        avg_latency_ms = total_runtime_ms / total_queries if total_queries > 0 else 0.0
        cost_per_query = total_cost / total_queries if total_queries > 0 else 0.0
        cost_per_correct = total_cost / correct_answers if correct_answers > 0 else float('inf')
        
        return {
            'model_breakdown': model_costs,
            'total_cost': round(total_cost, 6),
            'total_evaluations': len(results),
            'summary_metrics': {
                'accuracy': round(accuracy, 4),
                'avg_latency_ms': round(avg_latency_ms, 2),
                'total_cost': round(total_cost, 6),
                'cost_per_query': round(cost_per_query, 6),
                'cost_per_correct_answer': round(cost_per_correct, 6) if cost_per_correct != float('inf') else None
            },
            'evaluation_stats': {
                'total_queries': total_queries,
                'correct_answers': correct_answers,
                'total_runtime_ms': total_runtime_ms,
                'small_llm_usage': sum(1 for r in results if r.get('chosen_llm') == 'small'),
                'large_llm_usage': sum(1 for r in results if r.get('chosen_llm') == 'large')
            }
        }
    
    def compare_costs(self, baseline_cost: Dict, compound_cost: Dict) -> Dict[str, Any]:
        """Compare baseline vs compound costs and calculate savings."""
        baseline_total = baseline_cost.get('total_cost', 0.0)
        compound_total = compound_cost.get('total_cost', 0.0)
        
        savings = baseline_total - compound_total
        savings_percentage = (savings / baseline_total * 100) if baseline_total > 0 else 0.0
        
        return {
            'baseline_cost': baseline_total,
            'compound_cost': compound_total,
            'absolute_savings': round(savings, 6),
            'savings_percentage': round(savings_percentage, 2),
            'cost_efficiency': compound_total / baseline_total if baseline_total > 0 else 1.0
        }


# Example usage and testing
if __name__ == "__main__":
    catalog = ModelPricingCatalog()
    calculator = CostCalculator()
    
    # Test pricing lookup
    print("=== Model Pricing Examples ===")
    models_to_test = [
        "gpt-4o-mini",
        "claude-3-haiku-20240307", 
        "llama3.2:3b",
        "gemini-1.5-flash"
    ]
    
    for model in models_to_test:
        pricing = catalog.get_model_pricing(model)
        print(f"{model}: {pricing}")
    
    # Test cost calculation
    print("\n=== Cost Calculation Example ===")
    cost = calculator.calculate_cost(
        model_name="gpt-4o-mini",
        input_tokens=1000,
        output_tokens=500
    )
    print(f"GPT-4o-mini cost: {cost}")