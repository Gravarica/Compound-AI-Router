from datasets import load_dataset
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader import ARCDataManager
from mmlu_data_manager import MMLUDataManager
from gsm8k_data_manager import GSM8KDataManager

class MultiDatasetManager:
    """Unified manager for multiple datasets (ARC, MMLU, GSM8K)."""
    
    def __init__(self, datasets: List[str] = ['arc', 'mmlu', 'gsm8k']):
        self.datasets = datasets
        self.managers = {}
        
        # Initialize individual managers
        if 'arc' in datasets:
            self.managers['arc'] = ARCDataManager()
        if 'mmlu' in datasets:
            self.managers['mmlu'] = MMLUDataManager()
        if 'gsm8k' in datasets:
            self.managers['gsm8k'] = GSM8KDataManager()
    
    def load_all_data(self) -> None:
        """Load data for all specified datasets."""
        for dataset_name, manager in self.managers.items():
            print(f"\n=== Loading {dataset_name.upper()} ===")
            manager.load_data()
    
    def create_combined_router_training_data(self,
                                           val_split_ratio: float = 0.1,
                                           test_split_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create combined router training data from all datasets."""
        
        combined_train_data = []
        combined_val_data = []
        combined_test_data = []
        
        for dataset_name, manager in self.managers.items():
            print(f"\n--- Processing {dataset_name.upper()} for router training ---")
            train_data, val_data, test_data = manager.create_router_training_data(
                val_split_ratio, test_split_ratio
            )
            
            # Add dataset identifier to each item
            for item in train_data:
                item['dataset'] = dataset_name
            for item in val_data:
                item['dataset'] = dataset_name
            for item in test_data:
                item['dataset'] = dataset_name
                
            combined_train_data.extend(train_data)
            combined_val_data.extend(val_data)
            combined_test_data.extend(test_data)
        
        # Shuffle combined data
        random.shuffle(combined_train_data)
        random.shuffle(combined_val_data)
        random.shuffle(combined_test_data)
        
        print(f"\n=== COMBINED ROUTER TRAINING DATA ===")
        print(f"Total training examples: {len(combined_train_data)}")
        print(f"Total validation examples: {len(combined_val_data)}")
        print(f"Total test examples: {len(combined_test_data)}")
        
        # Print dataset distribution
        train_dist = {}
        for item in combined_train_data:
            dataset = item['dataset']
            train_dist[dataset] = train_dist.get(dataset, 0) + 1
        
        print(f"Training distribution: {train_dist}")
        
        return combined_train_data, combined_val_data, combined_test_data
    
    def get_evaluation_set(self, 
                          dataset_name: str,
                          max_samples: int = 500,
                          use_router_test_split: bool = False,
                          router_test_data: Optional[List[Dict]] = None) -> List[Dict]:
        """Get evaluation set for a specific dataset."""
        
        if dataset_name not in self.managers:
            raise ValueError(f"Dataset {dataset_name} not available. Available: {list(self.managers.keys())}")
        
        manager = self.managers[dataset_name]
        
        if dataset_name == 'arc':
            return manager.get_arc_evaluation_set(use_router_test_split, router_test_data)
        elif dataset_name == 'mmlu':
            return manager.get_mmlu_evaluation_set(use_router_test_split, router_test_data, max_samples)
        elif dataset_name == 'gsm8k':
            return manager.get_gsm8k_evaluation_set(use_router_test_split, router_test_data, max_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def create_prompt(self, dataset_name: str, question_data: Dict) -> str:
        """Create prompt for a specific dataset."""
        if dataset_name not in self.managers:
            raise ValueError(f"Dataset {dataset_name} not available.")
        
        return self.managers[dataset_name].create_prompt(question_data)
    
    def validate_answer(self, dataset_name: str, predicted_answer: str, question_data: Dict) -> bool:
        """Validate answer for a specific dataset."""
        if dataset_name == 'arc':
            correct_answer = question_data['correct_answer']
            return predicted_answer.strip().upper() == correct_answer.strip().upper()
        
        elif dataset_name == 'mmlu':
            correct_answer_idx = question_data['correct_answer']
            correct_letter = self.managers['mmlu'].get_answer_letter(correct_answer_idx)
            return predicted_answer.strip().upper() == correct_letter.strip().upper()
        
        elif dataset_name == 'gsm8k':
            correct_answer = question_data['correct_answer']
            return self.managers['gsm8k'].validate_answer(predicted_answer, correct_answer)
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def get_dataset_stats(self) -> Dict[str, Dict]:
        """Get statistics for all loaded datasets."""
        stats = {}
        
        for dataset_name, manager in self.managers.items():
            if dataset_name == 'arc':
                train_data, val_data, test_data = manager.create_router_training_data()
                eval_set = manager.get_arc_evaluation_set()
            elif dataset_name == 'mmlu':
                train_data, val_data, test_data = manager.create_router_training_data()
                eval_set = manager.get_mmlu_evaluation_set(max_samples=500)
            elif dataset_name == 'gsm8k':
                train_data, val_data, test_data = manager.create_router_training_data()
                eval_set = manager.get_gsm8k_evaluation_set(max_samples=500)
            
            # Calculate difficulty distribution
            easy_count = sum(1 for item in train_data + val_data + test_data if item.get('difficulty') == 'easy' or item.get('label') == 0)
            hard_count = sum(1 for item in train_data + val_data + test_data if item.get('difficulty') == 'hard' or item.get('label') == 1)
            
            stats[dataset_name] = {
                'router_train_size': len(train_data),
                'router_val_size': len(val_data),
                'router_test_size': len(test_data),
                'eval_size': len(eval_set),
                'easy_examples': easy_count,
                'hard_examples': hard_count,
                'difficulty_ratio': hard_count / (easy_count + hard_count) if (easy_count + hard_count) > 0 else 0
            }
        
        return stats

def test_multi_dataset_manager():
    """Test the multi-dataset manager."""
    print("🚀 Testing Multi-Dataset Manager")
    print("=" * 50)
    
    # Test with all datasets
    manager = MultiDatasetManager(['arc', 'mmlu', 'gsm8k'])
    manager.load_all_data()
    
    # Create combined router training data
    train_data, val_data, test_data = manager.create_combined_router_training_data()
    
    # Get evaluation sets for each dataset
    print(f"\n=== EVALUATION SETS ===")
    for dataset_name in ['arc', 'mmlu', 'gsm8k']:
        eval_set = manager.get_evaluation_set(dataset_name, max_samples=100)
        print(f"{dataset_name.upper()}: {len(eval_set)} evaluation examples")
        
        # Test prompt creation
        sample = eval_set[0]
        prompt = manager.create_prompt(dataset_name, sample)
        print(f"\nSample {dataset_name.upper()} prompt:")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    # Get dataset statistics
    stats = manager.get_dataset_stats()
    print(f"\n=== DATASET STATISTICS ===")
    for dataset_name, stat in stats.items():
        print(f"\n{dataset_name.upper()}:")
        for key, value in stat.items():
            if key == 'difficulty_ratio':
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")
    
    return {
        'total_train': len(train_data),
        'total_val': len(val_data),
        'total_test': len(test_data),
        'datasets': list(manager.managers.keys()),
        'stats': stats
    }

if __name__ == "__main__":
    results = test_multi_dataset_manager()
    print(f"\n=== FINAL RESULTS ===")
    for key, value in results.items():
        if key != 'stats':
            print(f"{key}: {value}")