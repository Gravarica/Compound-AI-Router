'''
from datasets import load_dataset, DatasetDict, concatenate_datasets

def load_hf_dataset(path, name, split = None):
    arc_easy = load_dataset(path, name, split=split)
    return arc_easy


def test():
    easy = load_hf_dataset("ai2_arc", "ARC-Easy")
    challenge = load_hf_dataset("ai2_arc", "ARC-Challenge")

    easy = easy.map(lambda x: {**x, "difficulty": "easy"})
    challenge = challenge.map(lambda x: {**x, "difficulty": "hard"})

    combined_train = concatenate_datasets([easy["train"], challenge["train"]])
    combined_validation = concatenate_datasets([easy["validation"], challenge["validation"]])

    combined_train = combined_train.shuffle(seed=42)
    combined_validation = combined_validation.shuffle(seed=42)

    combined = DatasetDict({
        "train": combined_train,
        "validation": combined_validation
    })

    print(combined["train"][0])

test()
'''

from datasets import load_dataset, DatasetDict, concatenate_datasets
import os
import json
import random
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from sklearn.model_selection import train_test_split

class ARCDataManager:
    def __init__(self,
                 easy_path: str = 'ai2_arc/ARC-Easy',
                 challenge_path: str = 'ai2_arc/ARC-Challenge'):
        self.easy_data_path = easy_path
        self.challenge_data_path = challenge_path
        self.arc_easy = None
        self.arc_challenge = None
        self.arc_combined = None

    def load_data(self) -> None:
        print("Loading ARC-Easy dataset...")
        arc_easy_split = self.easy_data_path.split("/")
        self.arc_easy = load_dataset(arc_easy_split[0], arc_easy_split[1])

        print("Loading ARC-Challenge dataset...")
        arc_hard_split = self.challenge_data_path.split("/")
        self.arc_challenge = load_dataset(arc_hard_split[0], arc_hard_split[1])

        print(f"Loaded ARC-Easy dataset with: ")
        print(f"\t {len(self.arc_easy['train'])} training examples")
        print(f"\t {len(self.arc_easy['validation'])} validation examples")
        print(f"\t {len(self.arc_easy['test'])} test examples")

        print(f"Loaded ARC-Challenge dataset with: ")
        print(f"\t {len(self.arc_challenge['train'])} training examples")
        print(f"\t {len(self.arc_challenge['validation'])} validation examples")
        print(f"\t {len(self.arc_challenge['test'])} test examples")

    def _preprocess_question(self, question_data: Dict) -> str:
        question = question_data['question']
        choices = question_data['choices']

        formatted_choices = ""
        for i, choice in enumerate(choices['text']):
            label = choices['label'][i]
            formatted_choices += f"{label}. {choice}\n"

        formatted_question = f"Question: {question} \n\nChoices:\n{formatted_choices}"
        return formatted_question

    def create_router_training_data(self,
                                    val_split_ratio: float = 0.1,
                                    test_split_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:

        if self.arc_easy is None or self.arc_challenge is None:
            self.load_data()

        easy_data = []
        for split in ['train', 'validation']:
            for item in self.arc_easy[split]:
                formatted_text = self._preprocess_question(item)
                easy_data.append({
                    'text': formatted_text,
                    'label': 0,
                    'id': item['id'],
                    'original_data': item
                })

        challenge_data = []
        for split in ['train', 'validation']:
            for item in self.arc_challenge[split]:
                formatted_text = self._preprocess_question(item)
                challenge_data.append({
                    'text': formatted_text,
                    'label': 1,
                    'id': item['id'],
                    'original_data': item
                })

        combined_data = easy_data + challenge_data
        random.shuffle(combined_data)

        test_size = val_split_ratio + test_split_ratio
        train_data, temp_data = train_test_split(combined_data, test_size=test_size, random_state=42)

        relative_val_size = val_split_ratio / test_size
        val_data, test_data = train_test_split(temp_data, test_size=(1-relative_val_size), random_state=42)

        print(f"Created router training data: ")
        print(f"\t {len(train_data)} training examples")
        print(f"\t {len(val_data)} validation examples")
        print(f"\t {len(test_data)} test examples")

        return train_data, val_data, test_data

    def get_arc_evaluation_set(self,
                               use_router_test_split: bool = False,
                               router_test_data: Optional[List[Dict]] = None) -> List[Dict]:
        if use_router_test_split and router_test_data:
            eval_set = []
            for item in router_test_data:
                eval_set.append({
                    'question': item['original_data']['question'],
                    'choices': item['original_data']['choices'],
                    'correct_answer': item['original_data']['answerKey'],
                    'id': item['id'],
                    'difficulty': 'easy' if item['label'] == 0 else 'hard',
                    'formatted_text': item['text']
                })
            return eval_set
        else:
            if self.arc_easy is None or self.arc_challenge is None:
                self.load_data()

            eval_set = []

            for item in self.arc_easy['test']:
                eval_set.append({
                    'question': item['question'],
                    'choices': item['choices'],
                    'correct_answer': item['answerKey'],
                    'id': item['id'],
                    'difficulty': 'easy',
                    'formatted_text': self._preprocess_question(item)
                })

            for item in self.arc_challenge['test']:
                eval_set.append({
                    'question': item['question'],
                    'choices': item['choices'],
                    'correct_answer': item['answerKey'],
                    'id': item['id'],
                    'difficulty': 'hard',
                    'formatted_text': self._preprocess_question(item)
                })

            print(f"Created evaluation set with {len(eval_set)} examples")
            return eval_set

    def create_prompt(self, question_data: Dict) -> str:
        formatted_question = self._preprocess_question(question_data)
        prompt = f"{formatted_question}\n\nPlease select the correct answer (A, B, C, or D) and only answer with the choice letter."
        return prompt

def test_arc_data_manager():
    manager = ARCDataManager()
    manager.load_data()

    train_data, val_data, test_data = manager.create_router_training_data()

    eval_set = manager.get_arc_evaluation_set()

    router_eval_set = manager.get_arc_evaluation_set(use_router_test_split=True, router_test_data=test_data)

    sample = eval_set[0]
    prompt = manager.create_prompt(sample)

    print(prompt)

    return {
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'eval_size': len(eval_set),
        'router_eval_size': len(router_eval_set)
    }

if __name__ == "__main__":
    results = test_arc_data_manager()
    print(f"\nTest Results: ")
    for key, value in results.items():
        print(f"\t{key}: {value}")