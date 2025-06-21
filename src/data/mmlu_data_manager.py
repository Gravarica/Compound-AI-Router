from datasets import load_dataset
import random
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split

class MMLUDataManager:
    def __init__(self, dataset_path: str = 'cais/mmlu'):
        self.dataset_path = dataset_path
        self.mmlu_data = None
        
        # MMLU subjects categorized by difficulty (based on typical performance)
        self.easy_subjects = [
            'high_school_geography', 'high_school_psychology', 'marketing',
            'public_relations', 'sociology', 'high_school_macroeconomics',
            'human_sexuality', 'nutrition', 'professional_psychology',
            'high_school_government_and_politics', 'high_school_world_history',
            'high_school_us_history', 'high_school_european_history',
            'moral_scenarios', 'miscellaneous', 'global_facts', 'us_foreign_policy',
            'security_studies', 'philosophy', 'prehistory', 'world_religions',
            'business_ethics', 'jurisprudence', 'moral_disputes'
        ]
        
        self.hard_subjects = [
            'abstract_algebra', 'college_mathematics', 'college_physics',
            'electrical_engineering', 'machine_learning', 'formal_logic',
            'college_chemistry', 'anatomy', 'professional_medicine',
            'college_computer_science', 'computer_security', 'high_school_chemistry',
            'high_school_physics', 'high_school_mathematics', 'high_school_biology',
            'college_biology', 'conceptual_physics', 'astronomy', 'econometrics',
            'clinical_knowledge', 'medical_genetics', 'virology', 'logical_fallacies',
            'international_law', 'professional_law'
        ]

    def load_data(self) -> None:
        print("Loading MMLU dataset...")
        self.mmlu_data = load_dataset(self.dataset_path, 'all')
        
        print(f"Loaded MMLU dataset with: ")
        print(f"\t {len(self.mmlu_data['auxiliary_train'])} auxiliary training examples")
        print(f"\t {len(self.mmlu_data['dev'])} dev examples")
        print(f"\t {len(self.mmlu_data['validation'])} validation examples") 
        print(f"\t {len(self.mmlu_data['test'])} test examples")
        
        # Print subject distribution
        subjects = set([item['subject'] for item in self.mmlu_data['test']])
        print(f"\t {len(subjects)} subjects total")

    def _preprocess_question(self, question_data: Dict) -> str:
        question = question_data['question']
        choices = question_data['choices']
        
        formatted_choices = ""
        choice_labels = ['A', 'B', 'C', 'D']
        for i, choice in enumerate(choices):
            formatted_choices += f"{choice_labels[i]}. {choice}\n"
        
        formatted_question = f"Question: {question}\n\nChoices:\n{formatted_choices}"
        return formatted_question

    def _determine_difficulty(self, item: Dict) -> str:
        """Determine difficulty based on subject categorization."""
        subject = item['subject']
        if subject in self.easy_subjects:
            return 'easy'
        elif subject in self.hard_subjects:
            return 'hard'
        else:
            # For subjects not explicitly categorized, use a heuristic
            # You could also compute this based on model performance
            return 'medium'  # We'll handle this in the router training

    def create_router_training_data(self,
                                  val_split_ratio: float = 0.1,
                                  test_split_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        
        if self.mmlu_data is None:
            self.load_data()
        
        training_data = []
        
        # Use auxiliary_train and dev splits for router training
        for split in ['auxiliary_train', 'dev']:
            for item in self.mmlu_data[split]:
                difficulty = self._determine_difficulty(item)
                
                # Skip medium difficulty for binary classification
                if difficulty in ['easy', 'hard']:
                    formatted_text = self._preprocess_question(item)
                    training_data.append({
                        'text': formatted_text,
                        'label': 0 if difficulty == 'easy' else 1,
                        'id': f"mmlu_{item['subject']}_{len(training_data)}",
                        'subject': item['subject'],
                        'difficulty': difficulty,
                        'original_data': item
                    })
        
        random.shuffle(training_data)
        
        test_size = val_split_ratio + test_split_ratio
        train_data, temp_data = train_test_split(training_data, test_size=test_size, random_state=42)
        
        relative_val_size = val_split_ratio / test_size
        val_data, test_data = train_test_split(temp_data, test_size=(1-relative_val_size), random_state=42)
        
        print(f"Created MMLU router training data: ")
        print(f"\t {len(train_data)} training examples")
        print(f"\t {len(val_data)} validation examples")
        print(f"\t {len(test_data)} test examples")
        
        return train_data, val_data, test_data

    def get_mmlu_evaluation_set(self,
                               use_router_test_split: bool = False,
                               router_test_data: Optional[List[Dict]] = None,
                               max_samples: int = 1000) -> List[Dict]:
        
        if use_router_test_split and router_test_data:
            eval_set = []
            for item in router_test_data:
                eval_set.append({
                    'question': item['original_data']['question'],
                    'choices': item['original_data']['choices'],
                    'correct_answer': item['original_data']['answer'],  # MMLU uses integer answer
                    'subject': item['subject'],
                    'id': item['id'],
                    'difficulty': item['difficulty'],
                    'text': item['text']
                })
            return eval_set
        else:
            if self.mmlu_data is None:
                self.load_data()
            
            eval_set = []
            
            # Sample from test set
            test_items = list(self.mmlu_data['test'])
            if len(test_items) > max_samples:
                test_items = random.sample(test_items, max_samples)
            
            for item in test_items:
                difficulty = self._determine_difficulty(item)
                eval_set.append({
                    'question': item['question'],
                    'choices': item['choices'],
                    'correct_answer': item['answer'],
                    'subject': item['subject'], 
                    'id': f"mmlu_eval_{item['subject']}_{len(eval_set)}",
                    'difficulty': difficulty,
                    'text': self._preprocess_question(item)
                })
            
            print(f"Created MMLU evaluation set with {len(eval_set)} examples")
            return eval_set

    def create_prompt(self, question_data: Dict) -> str:
        formatted_question = self._preprocess_question(question_data)
        prompt = f"{formatted_question}\n\nPlease select the correct answer (A, B, C, or D) and only answer with the choice letter."
        return prompt

    def get_answer_letter(self, answer_index: int) -> str:
        """Convert MMLU integer answer to letter."""
        choice_letters = ['A', 'B', 'C', 'D']
        return choice_letters[answer_index] if 0 <= answer_index < 4 else 'A'

def test_mmlu_data_manager():
    manager = MMLUDataManager()
    manager.load_data()
    
    train_data, val_data, test_data = manager.create_router_training_data()
    
    eval_set = manager.get_mmlu_evaluation_set(max_samples=500)
    
    router_eval_set = manager.get_mmlu_evaluation_set(use_router_test_split=True, router_test_data=test_data)
    
    sample = eval_set[0]
    prompt = manager.create_prompt(sample)
    
    print(f"\nSample MMLU prompt:")
    print(prompt)
    print(f"Correct answer: {manager.get_answer_letter(sample['correct_answer'])}")
    
    return {
        'train_size': len(train_data),
        'val_size': len(val_data), 
        'test_size': len(test_data),
        'eval_size': len(eval_set),
        'router_eval_size': len(router_eval_set)
    }

if __name__ == "__main__":
    results = test_mmlu_data_manager()
    print(f"\nMMLU Test Results: ")
    for key, value in results.items():
        print(f"\t{key}: {value}")