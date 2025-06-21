#!/usr/bin/env python3
"""
Quick Analysis Runner
Run comprehensive analysis of all experiment results with a single command.
"""

import sys
import os
sys.path.append('tools/analysis')

from comprehensive_experiment_analyzer import ComprehensiveAnalyzer

if __name__ == "__main__":
    print("🚀 Compound AI Router - Comprehensive Results Analysis")
    print("="*60)
    
    analyzer = ComprehensiveAnalyzer()
    analyzer.run_analysis()
    
    print("\n✨ Analysis complete!")
    print("Generated files:")
    print("• analysis_baseline_comparison.png - Baseline model comparison")
    print("• analysis_compound_comparison.png - Compound AI performance")
    print("• analysis_routing_effectiveness.png - Routing impact analysis")