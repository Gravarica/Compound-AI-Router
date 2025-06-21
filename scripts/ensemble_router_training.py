#!/usr/bin/env python3
"""
Ensemble Router Training

Combines predictions from multiple models to improve performance.
Often gives 3-7% accuracy boost over single models.
"""

import numpy as np
import json
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from src.routing import TransformerRouter
from src.data.dataloader import ARCDataManager


class EnsembleRouter:
    def __init__(self, model_paths: List[str], weights: List[float] = None):
        """
        Initialize ensemble of router models
        
        Args:
            model_paths: List of paths to trained router models
            weights: Optional weights for each model (default: equal weights)
        """
        self.routers = []
        self.weights = weights or [1.0] * len(model_paths)
        
        print("Loading ensemble models...")
        for i, path in enumerate(model_paths):
            try:
                router = TransformerRouter(model_name_or_path=path)
                self.routers.append(router)
                print(f"✓ Loaded {path}")
            except Exception as e:
                print(f"✗ Failed to load {path}: {e}")
        
        # Normalize weights
        total_weight = sum(self.weights[:len(self.routers)])
        self.weights = [w/total_weight for w in self.weights[:len(self.routers)]]
        
        print(f"Ensemble ready with {len(self.routers)} models")
        print(f"Weights: {[f'{w:.3f}' for w in self.weights]}")
    
    def predict_difficulty(self, query_text: str, confidence_threshold: float = 0.7) -> Tuple[str, float]:
        """Ensemble prediction by averaging probabilities"""
        
        if not self.routers:
            return 'hard', 0.5
        
        # Collect predictions from all models
        hard_probs = []
        
        for router in self.routers:
            pred_difficulty, confidence = router.predict_difficulty(
                query_text=query_text, 
                confidence_threshold=0.5  # Use 0.5 to get raw probabilities
            )
            
            # Convert to hard probability
            if pred_difficulty == 'hard':
                hard_probs.append(confidence)
            else:
                hard_probs.append(1.0 - confidence)
        
        # Weighted average
        ensemble_hard_prob = np.average(hard_probs, weights=self.weights)
        
        # Make final prediction
        if ensemble_hard_prob >= confidence_threshold:
            return 'hard', ensemble_hard_prob
        else:
            return 'easy', 1.0 - ensemble_hard_prob


def evaluate_ensemble():
    """Evaluate ensemble router performance"""
    
    # Load test data
    manager = ARCDataManager()
    eval_data = manager.get_arc_evaluation_set(use_router_test_split=False)
    
    # Define ensemble models (add paths of your best models)
    ensemble_paths = [
        "./model-store/distilbert_hybrid",
        "./model-store/minilm_enhanced", 
        "./model-store/roberta_enhanced"
    ]
    
    # Test different weighting strategies
    weight_strategies = {
        "equal": [1.0, 1.0, 1.0],
        "favor_roberta": [1.0, 1.0, 2.0],  # If RoBERTa performs best
        "favor_minilm": [1.0, 2.0, 1.0],   # If MiniLM performs best
    }
    
    results = {}
    
    for strategy_name, weights in weight_strategies.items():
        print(f"\n=== Testing {strategy_name} weighting ===")
        
        ensemble = EnsembleRouter(ensemble_paths, weights)
        
        y_true = []
        y_pred = []
        y_probs = []
        
        print("Evaluating ensemble...")
        for item in eval_data[:500]:  # Test on subset first
            text = item['text']
            true_label = item['label']
            
            pred_difficulty, confidence = ensemble.predict_difficulty(text, confidence_threshold=0.7)
            pred_label = 1 if pred_difficulty == 'hard' else 0
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            # Get probability for ROC
            pred_difficulty_raw, confidence_raw = ensemble.predict_difficulty(text, confidence_threshold=0.5)
            if pred_difficulty_raw == 'hard':
                y_probs.append(confidence_raw)
            else:
                y_probs.append(1.0 - confidence_raw)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_probs)
        
        results[strategy_name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(classification_report(y_true, y_pred, target_names=['easy', 'hard']))
    
    # Save results
    with open("./results/ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to ./results/ensemble_results.json")
    return results


if __name__ == "__main__":
    evaluate_ensemble()