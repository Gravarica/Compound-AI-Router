#!/usr/bin/env python3
"""
Hyperparameter Tuning for Router Models

Systematically search for better hyperparameters to boost performance.
"""

import os
import json
import itertools
from datetime import datetime

from src.routing.query_router import QueryRouter
from src.data.dataloader import ARCDataManager


def hyperparameter_search():
    """Grid search over key hyperparameters"""
    
    # Load data once
    manager = ARCDataManager()
    train_data, val_data, test_data = manager.create_router_training_data(
        balance_classes=True, 
        balance_strategy="hybrid"
    )
    
    # Hyperparameter grid
    param_grid = {
        'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
        'batch_size': [8, 16, 32],
        'epochs': [8, 12, 15],
        'weight_decay': [0.01, 0.05, 0.1],
        'warmup_ratio': [0.05, 0.1, 0.15]
    }
    
    # Models to test
    models_to_test = [
        "microsoft/MiniLM-L12-H384-uncased",
        "roberta-base",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    results = []
    
    for model_name in models_to_test:
        print(f"\n=== Tuning {model_name} ===")
        
        # Sample promising combinations (full grid would take too long)
        promising_combinations = [
            {'learning_rate': 2e-5, 'batch_size': 16, 'epochs': 12, 'weight_decay': 0.01, 'warmup_ratio': 0.1},
            {'learning_rate': 1e-5, 'batch_size': 32, 'epochs': 15, 'weight_decay': 0.05, 'warmup_ratio': 0.05},
            {'learning_rate': 3e-5, 'batch_size': 8, 'epochs': 10, 'weight_decay': 0.01, 'warmup_ratio': 0.15},
            {'learning_rate': 2e-5, 'batch_size': 16, 'epochs': 15, 'weight_decay': 0.05, 'warmup_ratio': 0.1},
        ]
        
        for i, params in enumerate(promising_combinations):
            print(f"\nTrying combination {i+1}/{len(promising_combinations)}: {params}")
            
            output_dir = f"./model-store/tuned_{model_name.replace('/', '_')}_{i+1}"
            
            try:
                # Initialize router
                router = QueryRouter(
                    model_name_or_path=model_name,
                    max_length=512
                )
                
                # Train with these parameters
                training_result = router.fine_tune(
                    train_data=train_data,
                    val_data=val_data,
                    output_dir=output_dir,
                    enhanced_features=True,  # Use enhanced features
                    **params
                )
                
                # Evaluate
                eval_results = router.evaluate_router(test_data)
                
                result = {
                    'model_name': model_name,
                    'params': params,
                    'output_dir': output_dir,
                    'train_results': training_result,
                    'eval_results': eval_results,
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
                
                print(f"Accuracy: {eval_results['accuracy']:.4f}")
                print(f"F1: {eval_results['f1']:.4f}")
                
            except Exception as e:
                print(f"Failed with params {params}: {e}")
                continue
    
    # Save all results
    with open("./results/hyperparameter_tuning_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Find best result
    if results:
        best_result = max(results, key=lambda x: x['eval_results']['accuracy'])
        print(f"\n=== Best Result ===")
        print(f"Model: {best_result['model_name']}")
        print(f"Params: {best_result['params']}")
        print(f"Accuracy: {best_result['eval_results']['accuracy']:.4f}")
        print(f"F1: {best_result['eval_results']['f1']:.4f}")
        print(f"Model saved to: {best_result['output_dir']}")
    
    return results


if __name__ == "__main__":
    hyperparameter_search()