# tests/test_model_pricing_fixed.py
import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.model_pricing import ModelPricingCatalog, CostCalculator


class TestModelPricingFixed(unittest.TestCase):
    """Test model pricing catalog and cost calculations with real config model names."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.catalog = ModelPricingCatalog()
        self.calculator = CostCalculator()
    
    def test_real_config_model_names(self):
        """Test model names exactly as they appear in config files and LLM interfaces."""
        # These are the exact model names from our config files and how they're returned by LLMs
        config_test_cases = [
            # (config_provider, config_model_name, llm_interface_returned_name, expected_found)
            ("claude", "claude-3-haiku-20240307", "claude/claude-3-haiku-20240307", True),
            ("openai", "gpt-4o-mini", "openai/gpt-4o-mini", True), 
            ("ollama", "llama3.2:1b", "ollama/llama3.2:1b", True),
            ("ollama", "llama3.2:3b", "ollama/llama3.2:3b", True),
            ("ollama", "phi:latest", "ollama/phi:latest", True),
        ]
        
        print("\n=== Testing Real Config Model Names ===")
        
        for provider, config_name, llm_returned_name, should_find in config_test_cases:
            with self.subTest(provider=provider, model=config_name):
                print(f"\nTesting: {provider} / {config_name}")
                print(f"LLM returns: {llm_returned_name}")
                
                # Test 1: Direct lookup with provider and model name
                pricing_direct = self.catalog.get_model_pricing(config_name, provider)
                print(f"Direct lookup result: {pricing_direct}")
                
                if should_find:
                    self.assertIsNotNone(pricing_direct, f"Direct lookup failed for {provider}/{config_name}")
                
                # Test 2: Cost calculation as it would happen in baseline evaluation
                if pricing_direct:
                    cost = self.calculator.calculate_cost(
                        model_name=llm_returned_name,  # This is what the evaluation script receives
                        input_tokens=100,
                        output_tokens=50,
                        runtime_hours=0.01 if provider == "ollama" else None
                    )
                    
                    print(f"Cost calculation result: {cost}")
                    
                    if 'error' in cost:
                        print(f"ERROR in cost calculation: {cost['error']}")
                        self.fail(f"Cost calculation failed for {llm_returned_name}: {cost['error']}")
                    
                    if provider == "ollama":
                        self.assertGreaterEqual(cost['total_cost'], 0, f"Ollama models should have >= 0 cost")
                        if cost['total_cost'] == 0 and cost['input_cost'] == 0 and cost['output_cost'] == 0:
                            print("WARNING: Ollama model has 0 total cost (no compute cost calculated)")
                    else:
                        self.assertGreater(cost['total_cost'], 0, f"API models should have > 0 cost")
                    
                    print(f"✅ {provider}/{config_name} cost: ${cost['total_cost']:.6f}")
                else:
                    print(f"❌ Could not find pricing for {provider}/{config_name}")
    
    def test_model_name_parsing_in_cost_calculator(self):
        """Test how the cost calculator handles model names with provider prefixes."""
        print("\n=== Testing Model Name Parsing ===")
        
        test_cases = [
            "claude/claude-3-haiku-20240307",
            "openai/gpt-4o-mini",
            "ollama/llama3.2:1b",
            "ollama/llama3.2:3b",
            "ollama/phi:latest",
        ]
        
        for full_model_name in test_cases:
            with self.subTest(model=full_model_name):
                print(f"\nTesting cost calculation for: {full_model_name}")
                
                cost = self.calculator.calculate_cost(
                    model_name=full_model_name,
                    input_tokens=1000,
                    output_tokens=500,
                    runtime_hours=0.1  # 6 minutes
                )
                
                print(f"Result: {cost}")
                
                if 'error' in cost:
                    print(f"❌ ERROR: {cost['error']}")
                    # Let's debug the parsing
                    if "/" in full_model_name:
                        provider, model_name = full_model_name.split("/", 1)
                        print(f"Parsed - Provider: '{provider}', Model: '{model_name}'")
                        
                        # Try direct lookup
                        pricing = self.catalog.get_model_pricing(model_name, provider)
                        print(f"Direct pricing lookup: {pricing}")
                        
                        if pricing is None:
                            # Try inference
                            inferred_provider = self.catalog._infer_provider(model_name)
                            print(f"Inferred provider: {inferred_provider}")
                            pricing_inferred = self.catalog.get_model_pricing(model_name, inferred_provider)
                            print(f"Inferred pricing lookup: {pricing_inferred}")
                else:
                    print(f"✅ Success: ${cost['total_cost']:.6f}")
    
    def test_baseline_evaluation_simulation(self):
        """Simulate exactly what happens in baseline evaluation."""
        print("\n=== Simulating Baseline Evaluation ===")
        
        # Simulate what baseline evaluation receives
        simulation_cases = [
            {
                'model_type': 'claude',
                'config_model_name': 'claude-3-haiku-20240307',
                'llm_interface_response': {
                    'model_name': 'claude/claude-3-haiku-20240307',  # What get_model_name() returns
                    'resource_usage': {
                        'prompt_tokens': 150,
                        'completion_tokens': 75,
                        'latency_ms': 2500
                    }
                }
            },
            {
                'model_type': 'ollama',
                'config_model_name': 'llama3.2:3b',
                'llm_interface_response': {
                    'model_name': 'ollama/llama3.2:3b',  # What get_model_name() returns
                    'resource_usage': {
                        'prompt_tokens': 150,
                        'completion_tokens': 75,
                        'latency_ms': 5000
                    }
                }
            }
        ]
        
        for case in simulation_cases:
            with self.subTest(model=case['config_model_name']):
                print(f"\nSimulating: {case['config_model_name']}")
                
                # Create fake baseline results as they would appear in evaluation
                fake_results = [
                    {
                        'model_name': case['llm_interface_response']['model_name'],
                        'resource_usage': case['llm_interface_response']['resource_usage'],
                        'total_time_ms': case['llm_interface_response']['resource_usage']['latency_ms'],
                        'correct': True
                    }
                ]
                
                print(f"Fake results: {fake_results[0]}")
                
                # Test baseline cost calculation
                cost_summary = self.calculator.calculate_baseline_cost(fake_results)
                print(f"Baseline cost summary: {cost_summary}")
                
                if 'error' in cost_summary:
                    print(f"❌ Baseline cost calculation failed: {cost_summary['error']}")
                    self.fail(f"Baseline cost calculation failed for {case['config_model_name']}")
                else:
                    print(f"✅ Baseline cost calculated: ${cost_summary['total_cost']:.6f}")
                    
                    # Check that we have the basic fields
                    required_fields = ['input_tokens', 'output_tokens', 'total_cost']
                    for field in required_fields:
                        self.assertIn(field, cost_summary, f"Missing {field} in cost summary")
    
    def test_pricing_catalog_completeness(self):
        """Test that all our config models are in the pricing catalog."""
        print("\n=== Testing Pricing Catalog Completeness ===")
        
        required_models = {
            'claude': ['claude-3-haiku-20240307'],
            'openai': ['gpt-4o-mini'],
            'ollama': ['llama3.2:1b', 'llama3.2:3b', 'phi:latest']
        }
        
        for provider, models in required_models.items():
            print(f"\nChecking {provider} models:")
            for model in models:
                pricing = self.catalog.get_model_pricing(model, provider)
                print(f"  {model}: {pricing}")
                self.assertIsNotNone(pricing, f"Missing pricing for {provider}/{model}")
                
                # Check required pricing fields
                self.assertIn('input', pricing, f"Missing input price for {provider}/{model}")
                self.assertIn('output', pricing, f"Missing output price for {provider}/{model}")
    
    def test_debug_cost_calculation_step_by_step(self):
        """Debug the cost calculation process step by step."""
        print("\n=== Debugging Cost Calculation Process ===")
        
        test_model = "claude/claude-3-haiku-20240307"
        print(f"Testing model: {test_model}")
        
        # Step 1: Parse model name
        if "/" in test_model:
            provider, model_name = test_model.split("/", 1)
        else:
            provider = None
            model_name = test_model
        
        print(f"Step 1 - Parsed provider: '{provider}', model_name: '{model_name}'")
        
        # Step 2: Get pricing
        pricing = self.catalog.get_model_pricing(model_name, provider)
        print(f"Step 2 - Pricing lookup result: {pricing}")
        
        # Step 3: Check if we have the model in catalog
        print(f"Step 3 - Available models in catalog:")
        for cat_provider, models in self.catalog.PRICING_DATA.items():
            print(f"  {cat_provider}: {list(models.keys())}")
        
        # Step 4: Try cost calculation
        if pricing:
            cost = self.calculator.calculate_cost(
                model_name=test_model,
                input_tokens=100,
                output_tokens=50
            )
            print(f"Step 4 - Cost calculation: {cost}")
        else:
            print("Step 4 - Skipped (no pricing found)")


if __name__ == '__main__':
    # Run the tests with maximum verbosity
    unittest.main(verbosity=2)