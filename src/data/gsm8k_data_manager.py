from datasets import load_dataset
import random
import re
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split

class GSM8KDataManager:
    def __init__(self, dataset_path: str = 'gsm8k'):
        self.dataset_path = dataset_path
        self.gsm8k_data = None

    def load_data(self) -> None:
        print("Loading GSM8K dataset...")
        self.gsm8k_data = load_dataset(self.dataset_path, 'main')
        
        print(f"Loaded GSM8K dataset with: ")
        print(f"\t {len(self.gsm8k_data['train'])} training examples")
        print(f"\t {len(self.gsm8k_data['test'])} test examples")

    def _extract_answer(self, answer_text: str) -> str:
        """Extract numerical answer from GSM8K answer format."""
        # GSM8K answers end with #### followed by the numerical answer
        match = re.search(r'#### (.+)', answer_text)
        if match:
            return match.group(1).strip()
        return answer_text.strip()

    def _count_reasoning_steps(self, answer_text: str) -> int:
        """Count reasoning steps in the solution (sentences or calculations)."""
        # Count sentences and calculation steps
        sentences = len([s for s in answer_text.split('.') if s.strip()])
        calculations = len(re.findall(r'<<.*?>>', answer_text))
        return max(sentences, calculations)

    def _determine_difficulty(self, item: Dict) -> str:
        """Determine difficulty based on problem characteristics."""
        question = item['question']
        answer = item['answer']
        
        # Count reasoning steps
        steps = self._count_reasoning_steps(answer)
        
        # Count numbers in question (proxy for complexity)
        numbers = len(re.findall(r'\b\d+(?:\.\d+)?\b', question))
        
        # Count words (longer problems tend to be harder)
        words = len(question.split())
        
        # Simple heuristic for difficulty
        difficulty_score = 0
        if steps >= 4:
            difficulty_score += 2
        elif steps >= 2:
            difficulty_score += 1
            
        if numbers >= 4:
            difficulty_score += 1
            
        if words >= 50:
            difficulty_score += 1
            
        return 'hard' if difficulty_score >= 3 else 'easy'

    def _preprocess_question(self, question_data: Dict) -> str:
        """Format GSM8K question for the model."""
        question = question_data['question']
        return f"Question: {question}"

    def create_router_training_data(self,
                                  val_split_ratio: float = 0.1,
                                  test_split_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        
        if self.gsm8k_data is None:
            self.load_data()
        
        training_data = []
        
        # Use train split for router training
        for item in self.gsm8k_data['train']:
            difficulty = self._determine_difficulty(item)
            formatted_text = self._preprocess_question(item)
            
            training_data.append({
                'text': formatted_text,
                'label': 0 if difficulty == 'easy' else 1,
                'id': f"gsm8k_{len(training_data)}",
                'difficulty': difficulty,
                'steps': self._count_reasoning_steps(item['answer']),
                'original_data': item
            })
        
        random.shuffle(training_data)
        
        test_size = val_split_ratio + test_split_ratio
        train_data, temp_data = train_test_split(training_data, test_size=test_size, random_state=42)
        
        relative_val_size = val_split_ratio / test_size
        val_data, test_data = train_test_split(temp_data, test_size=(1-relative_val_size), random_state=42)
        
        print(f"Created GSM8K router training data: ")
        print(f"\t {len(train_data)} training examples")
        print(f"\t {len(val_data)} validation examples")
        print(f"\t {len(test_data)} test examples")
        
        # Print difficulty distribution
        easy_count = sum(1 for item in training_data if item['difficulty'] == 'easy')
        hard_count = len(training_data) - easy_count
        print(f"\t Difficulty distribution: {easy_count} easy, {hard_count} hard")
        
        return train_data, val_data, test_data

    def get_gsm8k_evaluation_set(self,
                                use_router_test_split: bool = False,
                                router_test_data: Optional[List[Dict]] = None,
                                max_samples: int = 1000) -> List[Dict]:
        
        if use_router_test_split and router_test_data:
            eval_set = []
            for item in router_test_data:
                eval_set.append({
                    'question': item['original_data']['question'],
                    'correct_answer': self._extract_answer(item['original_data']['answer']),
                    'full_answer': item['original_data']['answer'],
                    'id': item['id'],
                    'difficulty': item['difficulty'],
                    'steps': item['steps'],
                    'text': item['text']
                })
            return eval_set
        else:
            if self.gsm8k_data is None:
                self.load_data()
            
            eval_set = []
            
            # Sample from test set
            test_items = list(self.gsm8k_data['test'])
            if len(test_items) > max_samples:
                test_items = random.sample(test_items, max_samples)
            
            for item in test_items:
                difficulty = self._determine_difficulty(item)
                eval_set.append({
                    'question': item['question'],
                    'correct_answer': self._extract_answer(item['answer']),
                    'full_answer': item['answer'],
                    'id': f"gsm8k_eval_{len(eval_set)}",
                    'difficulty': difficulty,
                    'steps': self._count_reasoning_steps(item['answer']),
                    'text': self._preprocess_question(item)
                })
            
            print(f"Created GSM8K evaluation set with {len(eval_set)} examples")
            return eval_set

    def create_prompt(self, question_data: Dict) -> str:
        formatted_question = self._preprocess_question(question_data)
        prompt = f"{formatted_question}\n\nPlease solve this step by step and provide the final numerical answer."
        return prompt

    def validate_answer(self, predicted_answer: str, correct_answer: str) -> bool:
        """Check if predicted answer matches correct answer (handling different formats)."""
        # Extract numerical value from both answers
        pred_nums = re.findall(r'-?\d+(?:\.\d+)?', predicted_answer.replace(',', ''))
        correct_nums = re.findall(r'-?\d+(?:\.\d+)?', correct_answer.replace(',', ''))
        
        if pred_nums and correct_nums:
            try:
                pred_val = float(pred_nums[-1])  # Take last number (usually the final answer)
                correct_val = float(correct_nums[-1])
                return abs(pred_val - correct_val) < 0.01  # Allow small floating point differences
            except ValueError:
                pass
        
        # Fallback to string comparison
        return predicted_answer.strip().lower() == correct_answer.strip().lower()

def test_gsm8k_data_manager():
    manager = GSM8KDataManager()
    manager.load_data()
    
    train_data, val_data, test_data = manager.create_router_training_data()
    
    eval_set = manager.get_gsm8k_evaluation_set(max_samples=500)
    
    router_eval_set = manager.get_gsm8k_evaluation_set(use_router_test_split=True, router_test_data=test_data)
    
    sample = eval_set[0]
    prompt = manager.create_prompt(sample)
    
    print(f"\nSample GSM8K prompt:")
    print(prompt)
    print(f"Correct answer: {sample['correct_answer']}")
    print(f"Difficulty: {sample['difficulty']} (steps: {sample['steps']})")
    
    return {
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'eval_size': len(eval_set),
        'router_eval_size': len(router_eval_set)
    }

if __name__ == "__main__":
    results = test_gsm8k_data_manager()
    print(f"\nGSM8K Test Results: ")
    for key, value in results.items():
        print(f"\t{key}: {value}")