# tests/test_model_pricing.py
import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.model_pricing import ModelPricingCatalog, CostCalculator


class TestModelPricing(unittest.TestCase):
    """Test model pricing catalog and cost calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.catalog = ModelPricingCatalog()
        self.calculator = CostCalculator()
    
    def test_claude_model_pricing(self):
        """Test Claude model pricing lookup."""
        # Test with full model name as returned by LLM interface
        claude_model_name = "claude/claude-3-haiku-20240307"
        pricing = self.catalog.get_model_pricing("claude-3-haiku-20240307", "claude")
        
        self.assertIsNotNone(pricing, f"Pricing not found for Claude Haiku")
        self.assertEqual(pricing["input"], 0.25)
        self.assertEqual(pricing["output"], 1.25)
        
        # Test cost calculation
        cost = self.calculator.calculate_cost(
            model_name="claude/claude-3-haiku-20240307",
            input_tokens=1000,
            output_tokens=500
        )
        
        self.assertGreater(cost["total_cost"], 0, "Claude cost should be > 0")
        self.assertEqual(cost["input_tokens"], 1000)
        self.assertEqual(cost["output_tokens"], 500)
        print(f"Claude Haiku cost test: {cost}")
    
    def test_openai_model_pricing(self):
        """Test OpenAI model pricing lookup."""
        # Test GPT-4o-mini as returned by LLM interface
        openai_model_name = "openai/gpt-4o-mini"
        pricing = self.catalog.get_model_pricing("gpt-4o-mini", "openai")
        
        self.assertIsNotNone(pricing, f"Pricing not found for GPT-4o-mini")
        self.assertEqual(pricing["input"], 0.15)
        self.assertEqual(pricing["output"], 0.60)
        
        # Test cost calculation
        cost = self.calculator.calculate_cost(
            model_name="openai/gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500
        )
        
        self.assertGreater(cost["total_cost"], 0, "OpenAI cost should be > 0")
        print(f"GPT-4o-mini cost test: {cost}")
    
    def test_ollama_model_pricing(self):
        """Test Ollama (local) model pricing lookup."""
        # Test Llama models as returned by LLM interface
        test_cases = [
            ("ollama/llama3.2:1b", "llama3.2:1b"),
            ("ollama/llama3.2:3b", "llama3.2:3b"),
            ("ollama/phi:latest", "phi:latest"),
        ]
        
        for full_name, model_name in test_cases:
            with self.subTest(model=model_name):
                pricing = self.catalog.get_model_pricing(model_name, "ollama")
                
                self.assertIsNotNone(pricing, f"Pricing not found for {model_name}")
                self.assertEqual(pricing["input"], 0.0, f"{model_name} should have 0 input cost")
                self.assertEqual(pricing["output"], 0.0, f"{model_name} should have 0 output cost")
                self.assertIn("compute_cost_per_hour", pricing, f"{model_name} should have compute cost")
                
                # Test cost calculation with runtime
                cost = self.calculator.calculate_cost(
                    model_name=full_name,
                    input_tokens=1000,
                    output_tokens=500,
                    runtime_hours=0.1  # 6 minutes
                )
                
                self.assertEqual(cost["input_cost"], 0.0)
                self.assertEqual(cost["output_cost"], 0.0)
                self.assertGreater(cost["compute_cost"], 0, f"{model_name} should have compute cost")
                print(f"{model_name} cost test: {cost}")
    
    def test_model_name_inference(self):
        """Test automatic provider inference from model names."""
        test_cases = [
            ("claude-3-haiku-20240307", "claude"),
            ("gpt-4o-mini", "openai"),
            ("llama3.2:1b", "ollama"),
            ("phi:latest", "ollama"),
            ("gemini-1.5-pro", "google"),
        ]
        
        for model_name, expected_provider in test_cases:
            with self.subTest(model=model_name):
                inferred_provider = self.catalog._infer_provider(model_name)
                self.assertEqual(inferred_provider, expected_provider, 
                               f"Expected {expected_provider} for {model_name}, got {inferred_provider}")
    
    def test_baseline_cost_calculation_simulation(self):
        """Test baseline cost calculation as it would be called from evaluation."""
        # Simulate baseline results from different models
        
        # Claude Haiku results simulation
        claude_results = [
            {
                'model_name': 'claude/claude-3-haiku-20240307',
                'resource_usage': {
                    'prompt_tokens': 150,
                    'completion_tokens': 50,
                },
                'total_time_ms': 2500,
                'correct': True
            },
            {
                'model_name': 'claude/claude-3-haiku-20240307', 
                'resource_usage': {
                    'prompt_tokens': 200,
                    'completion_tokens': 75,
                },
                'total_time_ms': 3000,
                'correct': False
            }
        ]
        
        claude_cost = self.calculator.calculate_baseline_cost(claude_results)
        self.assertGreater(claude_cost['total_cost'], 0, "Claude baseline cost should be > 0")
        self.assertEqual(claude_cost['input_tokens'], 350)
        self.assertEqual(claude_cost['output_tokens'], 125)
        print(f"Claude baseline cost simulation: {claude_cost}")
        
        # Llama 3B results simulation  
        llama_results = [
            {
                'model_name': 'ollama/llama3.2:3b',
                'resource_usage': {
                    'prompt_tokens': 150,
                    'completion_tokens': 50,
                },
                'total_time_ms': 5000,
                'correct': True
            },
            {
                'model_name': 'ollama/llama3.2:3b',
                'resource_usage': {
                    'prompt_tokens': 200, 
                    'completion_tokens': 75,
                },
                'total_time_ms': 6000,
                'correct': True
            }
        ]
        
        llama_cost = self.calculator.calculate_baseline_cost(llama_results)
        self.assertEqual(llama_cost['input_cost'], 0.0, "Llama should have 0 input cost")
        self.assertEqual(llama_cost['output_cost'], 0.0, "Llama should have 0 output cost") 
        self.assertGreater(llama_cost['compute_cost'], 0, "Llama should have compute cost")
        print(f"Llama 3B baseline cost simulation: {llama_cost}")
    
    def test_model_name_parsing_edge_cases(self):
        """Test edge cases in model name parsing."""
        # Test various formats that might come from LLM interfaces
        test_cases = [
            # (input_model_name, expected_parsed_name, expected_provider)
            ("claude/claude-3-haiku-20240307", "claude-3-haiku-20240307", "claude"),
            ("openai/gpt-4o-mini", "gpt-4o-mini", "openai"),
            ("ollama/llama3.2:3b", "llama3.2:3b", "ollama"),
            ("claude-3-haiku-20240307", "claude-3-haiku-20240307", "claude"),  # No prefix
            ("gpt-4o-mini", "gpt-4o-mini", "openai"),  # No prefix
            ("llama3.2:1b", "llama3.2:1b", "ollama"),  # No prefix
        ]
        
        for full_name, expected_name, expected_provider in test_cases:
            with self.subTest(model=full_name):
                # Test if we can find pricing for this model
                if "/" in full_name:
                    provider, model_name = full_name.split("/", 1)
                else:
                    provider = None
                    model_name = full_name
                
                pricing = self.catalog.get_model_pricing(model_name, provider)
                if pricing is None:
                    pricing = self.catalog.get_model_pricing(model_name)  # Try auto-inference
                
                self.assertIsNotNone(pricing, f"Could not find pricing for {full_name}")
                print(f"Successfully found pricing for {full_name}: {pricing}")
    
    def test_cost_calculation_with_zero_tokens(self):
        """Test cost calculation edge cases."""
        # Test with zero tokens
        cost = self.calculator.calculate_cost(
            model_name="claude/claude-3-haiku-20240307",
            input_tokens=0,
            output_tokens=0
        )
        
        self.assertEqual(cost['total_cost'], 0.0)
        self.assertEqual(cost['input_cost'], 0.0)
        self.assertEqual(cost['output_cost'], 0.0)
        
        # Test with unknown model
        cost = self.calculator.calculate_cost(
            model_name="unknown/unknown-model",
            input_tokens=1000,
            output_tokens=500
        )
        
        self.assertEqual(cost['total_cost'], 0.0)
        self.assertIn('error', cost)
        print(f"Unknown model cost (should be 0): {cost}")
    
    def test_real_config_model_names(self):
        """Test model names exactly as they appear in config files."""
        # These are the exact model names from our config files
        config_models = [
            # From baseline configs
            ("claude", "claude-3-haiku-20240307"),
            ("openai", "gpt-4o-mini"), 
            ("ollama", "llama3.2:1b"),
            ("ollama", "llama3.2:3b"),
            ("ollama", "phi:latest"),
        ]
        
        for provider, model_name in config_models:
            with self.subTest(provider=provider, model=model_name):
                pricing = self.catalog.get_model_pricing(model_name, provider)
                self.assertIsNotNone(pricing, f"Config model {provider}/{model_name} not found in pricing")
                
                # Test cost calculation
                cost = self.calculator.calculate_cost(
                    model_name=f"{provider}/{model_name}",
                    input_tokens=100,
                    output_tokens=50,
                    runtime_hours=0.01 if provider == "ollama" else None
                )
                
                if provider == "ollama":
                    self.assertGreaterEqual(cost['total_cost'], 0)  # Should have compute cost
                else:
                    self.assertGreater(cost['total_cost'], 0)  # Should have API cost
                
                print(f"Config model {provider}/{model_name} cost: ${cost['total_cost']:.6f}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)