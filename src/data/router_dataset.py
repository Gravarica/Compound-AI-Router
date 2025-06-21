# src/routing/datasets/router_dataset.py
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import re


class RouterDataset(Dataset):

    def __init__(self, data: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, max_length: int = 512, enhanced_features: bool = False):

        self.texts = [item['text'] for item in data]
        self.labels = [item['label'] for item in data]
        self.ids = [item['id'] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enhanced_features = enhanced_features
        
        # Pre-compute enhanced features if enabled
        if enhanced_features:
            self.text_features = [self._extract_text_features(text) for text in self.texts]

    def __len__(self):
        return len(self.texts)

    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract numerical features that might indicate question difficulty"""
        
        # Basic text statistics
        word_count = len(text.split())
        char_count = len(text)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        # Question complexity indicators
        question_marks = text.count('?')
        choices_count = len(re.findall(r'[A-D]\.|[1-4]\.', text))
        
        # Scientific complexity indicators
        scientific_terms = ['temperature', 'energy', 'molecule', 'atom', 'cell', 'DNA', 'gravity', 
                           'magnetic', 'electric', 'chemical', 'physical', 'reaction', 'equation',
                           'hypothesis', 'experiment', 'variable', 'control', 'data', 'graph']
        science_term_count = sum(1 for term in scientific_terms if term.lower() in text.lower())
        
        # Reasoning complexity indicators  
        reasoning_words = ['because', 'therefore', 'however', 'although', 'compare', 'contrast',
                          'analyze', 'evaluate', 'predict', 'explain', 'describe', 'determine']
        reasoning_word_count = sum(1 for word in reasoning_words if word.lower() in text.lower())
        
        # Quantitative indicators
        numbers = len(re.findall(r'\d+', text))
        
        return {
            'word_count': float(word_count),
            'char_count': float(char_count), 
            'avg_word_length': avg_word_length,
            'question_marks': float(question_marks),
            'choices_count': float(choices_count),
            'science_terms': float(science_term_count),
            'reasoning_words': float(reasoning_word_count),
            'numbers': float(numbers)
        }

    def _enhance_text(self, text: str, features: Dict[str, float]) -> str:
        """Add structural markers and feature hints to text"""
        
        # Add special tokens for structure
        enhanced = text.replace('Question:', '[QUESTION]')
        enhanced = enhanced.replace('Choices:', '[CHOICES]')
        
        # Add difficulty hint tokens based on features
        if features['science_terms'] >= 3:
            enhanced = '[HIGH_SCIENCE] ' + enhanced
        if features['reasoning_words'] >= 2:
            enhanced = '[HIGH_REASONING] ' + enhanced
        if features['word_count'] >= 50:
            enhanced = '[LONG_TEXT] ' + enhanced
            
        return enhanced

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Apply enhanced features if enabled
        if self.enhanced_features:
            features = self.text_features[idx]
            text = self._enhance_text(text, features)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(label, dtype=torch.long)

        return encoding

    @property
    def label_distribution(self) -> Dict[int, int]:

        import collections
        return dict(collections.Counter(self.labels))