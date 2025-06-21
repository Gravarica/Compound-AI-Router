#!/usr/bin/env python3
"""
Compare all router approaches tested so far.
"""
import json
from pathlib import Path

def compare_all_routers():
    """Compare performance across all router types."""
    
    results_dir = Path("results/experiments/router_types/llm")
    
    routers = {
        "Improved Phi-2 Router": {
            "file": results_dir / "phi2_router_llama3.2_claude_results_full.json",
            "description": "Phi-2 with improved prompt examples (3B vs Claude)"
        },
        "Chain-of-Thought Router": {
            "file": results_dir / "cot_router_results_full.json", 
            "description": "Phi-2 with step-by-step reasoning (3B vs Claude)"
        },
        "Stress Test Router": {
            "file": results_dir / "stress_test_1b_vs_claude_results_full.json",
            "description": "Phi-2 with bigger LLM gap (1B vs Claude)"
        }
    }
    
    print("🎯 ROUTER COMPARISON SUMMARY")
    print("=" * 60)
    
    comparison_data = []
    
    for router_name, info in routers.items():
        if info["file"].exists():
            with open(info["file"], 'r') as f:
                data = json.load(f)
            
            summary = data["summary"]
            
            router_data = {
                "name": router_name,
                "description": info["description"],
                "overall_accuracy": summary["overall_accuracy"],
                "router_accuracy": summary["router_performance"]["accuracy"],
                "small_llm_usage": summary["routing_distribution"]["small_llm"],
                "large_llm_usage": summary["routing_distribution"]["large_llm"],
                "small_llm_accuracy": summary["accuracy_by_llm"]["small_llm"],
                "large_llm_accuracy": summary["accuracy_by_llm"]["large_llm"],
                "avg_routing_time": summary["average_times_ms"]["routing"],
                "total_tokens": summary["large_llm_token_usage"]["total_tokens"]
            }
            
            comparison_data.append(router_data)
            
            print(f"\n📊 {router_name}")
            print(f"   Description: {info['description']}")
            print(f"   Overall Accuracy: {summary['overall_accuracy']:.1%}")
            print(f"   Router Accuracy: {summary['router_performance']['accuracy']:.1%}")
            print(f"   Small LLM Usage: {summary['routing_distribution']['small_llm']}/100")
            print(f"   Large LLM Usage: {summary['routing_distribution']['large_llm']}/100") 
            print(f"   Small LLM Accuracy: {summary['accuracy_by_llm']['small_llm']:.1%}")
            print(f"   Large LLM Accuracy: {summary['accuracy_by_llm']['large_llm']:.1%}")
            print(f"   Avg Routing Time: {summary['average_times_ms']['routing']:.0f}ms")
            print(f"   Total Tokens Used: {summary['large_llm_token_usage']['total_tokens']}")
    
    # Analysis and insights
    print(f"\n🔍 KEY INSIGHTS:")
    print("-" * 40)
    
    if len(comparison_data) >= 2:
        improved = comparison_data[0] if "Improved" in comparison_data[0]["name"] else comparison_data[1]
        cot = comparison_data[1] if "Chain" in comparison_data[1]["name"] else comparison_data[0]
        
        print(f"1. Router Accuracy:")
        print(f"   • Improved Prompt: {improved['router_accuracy']:.1%}")
        print(f"   • Chain-of-Thought: {cot['router_accuracy']:.1%}")
        print(f"   • Difference: {cot['router_accuracy'] - improved['router_accuracy']:+.1%}")
        
        print(f"\n2. Overall System Performance:")
        print(f"   • Improved Prompt: {improved['overall_accuracy']:.1%}")
        print(f"   • Chain-of-Thought: {cot['overall_accuracy']:.1%}")
        print(f"   • Difference: {cot['overall_accuracy'] - improved['overall_accuracy']:+.1%}")
        
        print(f"\n3. Resource Usage Strategy:")
        print(f"   • Improved uses small LLM {improved['small_llm_usage']}/100 times")
        print(f"   • CoT uses small LLM {cot['small_llm_usage']}/100 times")
        print(f"   • CoT is {'more' if cot['small_llm_usage'] < improved['small_llm_usage'] else 'less'} conservative")
        
        print(f"\n4. Token Efficiency:")
        print(f"   • Improved uses {improved['total_tokens']} tokens")
        print(f"   • CoT uses {cot['total_tokens']} tokens")
        print(f"   • CoT uses {cot['total_tokens'] - improved['total_tokens']:+} more tokens")
        
    print(f"\n💡 CONCLUSIONS:")
    print("-" * 40)
    print("• Chain-of-Thought router trades router accuracy for system accuracy")
    print("• CoT is more conservative (higher large LLM usage)")
    print("• Both routers still struggle with 1B parameter classification")
    print("• Need bigger LLM performance gaps to see router value clearly")
    print("\n🎯 NEXT: Run stress test with 1B vs Claude for bigger gaps!")

if __name__ == "__main__":
    compare_all_routers()