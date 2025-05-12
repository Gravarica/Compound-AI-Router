import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

class ArcDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.texts = [item['text'] for item in data]
        self.labels = [item['label'] for item in data]
        self.ids = [item['id'] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

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

class QueryRouter:
    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int = 2,
            device: Optional[str] = None,
            max_length: int = 512
    ):
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.max_length = max_length
        self.label_map = {0: 'easy', 1: 'hard'}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading tokenizer from {self.model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

            logger.info(f"Loading model from {self.model_name_or_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
                trust_remote_code=True
            )

            self.model.to(self.device)
            logger.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {e}")
            raise

    def fine_tune(
            self,
            train_data: List[Dict[str, Any]],
            val_data: List[Dict[str, Any]],
            output_dir: str,
            epochs: int = 3,
            batch_size: int = 8,
            learning_rate: float = 5e-5,
            weight_decay: float = 0.01,
            warmup_ratio: float = 0.1,
            early_stopping_patience: int = 3,
            early_stopping_threshold: float = 0.01,
            eval_steps: int = 100,
            save_total_limit: int = 3
    ) -> None:

        train_dataset = ArcDataset(train_data, self.tokenizer, self.max_length)
        val_dataset = ArcDataset(val_data, self.tokenizer, self.max_length)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            eval_steps=eval_steps,
            eval_strategy='steps',
            save_strategy='steps',
            load_best_model_at_end=True,
            save_total_limit=save_total_limit,
            save_steps=eval_steps,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=eval_steps,
            report_to="none"
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average="weighted"),
                "precision": precision_score(labels, predictions, average="weighted"),
                "recall": recall_score(labels, predictions, average="weighted")
            }

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=early_stopping_threshold
                )
            ]
        )

        logger.info("Starting fine-tuning...")
        trainer.train()

        # Evaluate the model
        logger.info("Evaluating the fine-tuned model...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")

        # Save the model
        logger.info(f"Saving the fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Create an evaluation summary file
        with open(os.path.join(output_dir, "eval_results.txt"), "w") as f:
            for key, value in eval_results.items():
                f.write(f"{key}: {value}\n")

        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")

    def predict_difficulty(self, query_text: str) -> str:
        """Takes raw question text, tokenizes, predicts the class, and returns 'easy' or 'hard'."""
        try:
            inputs = self.tokenizer(
                query_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            self.model.eval()

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()

            difficulty = self.label_map.get(predicted_class, 'hard')

            return difficulty

        except Exception as e:
            logger.error(f"Error predicting difficulty: {e}")
            return 'hard'

    def load_fine_tuned_model(self, model_path: str) -> None:
        """Loads a previously fine-tuned router model."""
        try:
            logger.info(f"Loading fine-tuned tokenizer from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            logger.info(f"Loading fine-tuned model from {model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

            self.model.to(self.device)
            logger.info("Fine-tuned model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            raise

    def evaluate_router(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Evaluates the router on test data and returns metrics."""
        test_dataset = ArcDataset(test_data, self.tokenizer, self.max_length)

        self.model.eval()

        all_preds = []
        all_labels = []
        all_texts = []
        all_ids = []

        # Create a test dataloader
        from torch.utils.data import DataLoader
        test_dataloader = DataLoader(test_dataset, batch_size=8)

        processed_indices = []

        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to('cpu').numpy()

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

            start_idx = batch_idx * test_dataloader.batch_size
            for i, pred in enumerate(preds):
                idx = start_idx + i
                if idx < len(test_dataset):
                    processed_indices.append(idx)
                    all_texts.append(test_dataset.texts[idx])
                    all_ids.append(test_dataset.ids[idx])

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(self.label_map.values()),
                    yticklabels=list(self.label_map.values()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, F1: {f1:.4f}')

        cm_path = 'confusion_matrix.png'
        plt.savefig(cm_path)
        plt.close()

        misclassified = []
        for i in range(len(all_preds)):
            if all_preds[i] != all_labels[i]:
                misclassified.append({
                    'id': all_ids[i],
                    'text': all_texts[i],
                    'true_label': self.label_map[all_labels[i]],
                    'predicted_label': self.label_map[all_preds[i]]
                })

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_plot_path': cm_path,
            'misclassified_count': len(misclassified),
            'misclassified_examples': misclassified[:10]  # Include only first 10 for brevity
        }

        return results

def test_query_router():
    """Simple test function for QueryRouter."""
    from dataloader import ARCDataManager

    manager = ARCDataManager()
    train_data, val_data, test_data = manager.create_router_training_data()

    router = QueryRouter(
        model_name_or_path="distilbert-base-uncased",  # Use a lightweight model for testing
    )

    sample_question = train_data[0]['text']
    print(f"Sample question: {sample_question[:100]}...")
    difficulty = router.predict_difficulty(sample_question)
    print(f"Predicted difficulty: {difficulty}")

    # router.fine_tune(
    #     train_data=train_data[:100],  # Use a subset for testing
    #     val_data=val_data[:20],
    #     output_dir="./router_model",
    #     epochs=1
    # )

    print("QueryRouter test completed successfully!")

if __name__ == "__main__":
    test_query_router()