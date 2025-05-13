import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import gc

from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, DataLoader
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


class CustomTrainer(Trainer):
    def __init__(self, loss_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        device = next(model.parameters()).device

        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(**inputs)

        logits = outputs.logits.to(torch.float32)
        labels = labels.to(torch.long)

        loss = self.loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

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

        torch.mps.empty_cache()

        self.mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        self.device = torch.device("mps" if self.mps_available else "cpu")

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

            for param in self.model.parameters():
                if param.data.dtype == torch.float16:  # Or other non-float32 types
                    param.data = param.data.to(torch.float32)

            self.model = self.model.to(self.device)  # Crucial
            logger.info(f"Model and tokenizer loaded successfully on {self.device}")
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
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            save_total_limit=save_total_limit,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=eval_steps,
            report_to="none",
            # Remove MPS-specific settings that might be causing issues
            fp16=False,
            bf16=False,
            dataloader_pin_memory=False,
            # Add these to prevent potential issues
            gradient_accumulation_steps=2,
            optim="adamw_torch",  # Use basic PyTorch optimizer implementation
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average="weighted"),
                "precision": precision_score(labels, predictions, average="weighted"),
                "recall": recall_score(labels, predictions, average="weighted"),
                'hard_f1': f1_score(labels, predictions, pos_label=1),
                'easy_f1': f1_score(labels, predictions, pos_label=0),
                'macro_f1': f1_score(labels, predictions, average="macro"),
            }

        from torch import nn

        class_counts = np.bincount(train_dataset.labels)
        class_weights = torch.tensor([1.0, 2.0],
                                     dtype=torch.float32,
                                     device=self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            loss_fn=loss_fn,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=early_stopping_threshold
                )
            ]
        )

        logger.info("Starting fine-tuning...")
        trainer.train()

        logger.info("Evaluating the fine-tuned model...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")

        logger.info(f"Saving the fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "eval_results.txt"), "w") as f:
            for key, value in eval_results.items():
                f.write(f"{key}: {value}\n")

        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")

    def predict_difficulty(self, query_text: str, easy_confidence_threshold: float = 0.7) -> Tuple[str, float]:
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
                probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
                easy_prob = float(probs[0])
                hard_prob = float(probs[1])

                print(f'DEBUG - Easy prob: {easy_prob}, Hard prob: {hard_prob}, Threshold: {easy_confidence_threshold}')

                if easy_prob >= easy_confidence_threshold:
                    difficulty = 'easy'
                    confidence = easy_prob
                else:
                    difficulty = 'hard'
                    confidence = hard_prob if hard_prob > easy_prob else easy_prob

            return difficulty, confidence

        except Exception as e:
            logger.error(f"Error predicting difficulty: {e}")
            return 'hard'

    def load_fine_tuned_model(self, model_path: str) -> None:
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
        test_dataset = ArcDataset(test_data, self.tokenizer, self.max_length)

        self.model.eval()

        all_preds = []
        all_labels = []
        all_texts = []
        all_ids = []

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

    def analyze_confidence_distribution(self, test_data: List[Dict]) -> Dict[str, Any]:

        test_dataset = ArcDataset(test_data, self.tokenizer, self.max_length)

        self.model.eval()

        easy_confidences = []
        hard_confidences = []
        correct_confidences = []
        incorrect_confidences = []

        true_labels = []
        pred_probs = []

        test_dataloader = DataLoader(test_dataset, batch_size=8)

        for batch in tqdm(test_dataloader, desc='Analyzing confidence distribution'):
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to('cpu').numpy()

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

            for i, (label, pred, prob) in enumerate(zip(labels, preds, probs)):

                confidence = prob[pred]

                if label == 0:
                    easy_confidences.append((confidence, pred == label))
                else:
                    hard_confidences.append((confidence, pred == label))

                if pred == label:
                    correct_confidences.append(confidence)
                else:
                    incorrect_confidences.append(confidence)

                true_labels.append(label)
                pred_probs.append(prob[0])

        fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
        roc_auc = auc(fpr, tpr)

        precision, recall, pr_thresholds = precision_recall_curve(true_labels, pred_probs)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        thresholds_by_accuracy = {}
        for acc_req in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            accuracies = []
            for threshold in np.arange(0.1, 1.0, 0.01):
                preds = [1 if p < threshold else 0 for p in pred_probs]  # 1 = hard, 0 = easy
                accuracy = np.mean([p == t for p, t in zip(preds, true_labels)])
                accuracies.append((threshold, accuracy))

            closest = min(accuracies, key=lambda x: abs(x[1] - acc_req))
            thresholds_by_accuracy[acc_req] = closest

        analysis = {
            "confidence_stats": {
                "overall_mean": np.mean([c for c, _ in easy_confidences + hard_confidences]),
                "overall_std": np.std([c for c, _ in easy_confidences + hard_confidences]),
                "easy_mean": np.mean([c for c, _ in easy_confidences]) if easy_confidences else 0,
                "hard_mean": np.mean([c for c, _ in hard_confidences]) if hard_confidences else 0,
                "correct_mean": np.mean(correct_confidences) if correct_confidences else 0,
                "incorrect_mean": np.mean(incorrect_confidences) if incorrect_confidences else 0,
            },
            "threshold_analysis": {
                "roc_auc": roc_auc,
                "optimal_threshold": float(optimal_threshold),
                "thresholds_by_accuracy": {str(k): {"threshold": v[0], "accuracy": v[1]}
                                           for k, v in thresholds_by_accuracy.items()}
            },
            "calibration": {
                "confidence_bins": [],
            }
        }

        bin_edges = np.arange(0, 1.1, 0.1)
        for i in range(len(bin_edges) - 1):
            low, high = bin_edges[i], bin_edges[i + 1]
            bin_samples = [(c, p) for c, p in easy_confidences + hard_confidences
                           if low <= c < high]

            if bin_samples:
                bin_accuracy = np.mean([p for _, p in bin_samples])
                bin_confidence = np.mean([c for c, _ in bin_samples])
                bin_count = len(bin_samples)

                analysis["calibration"]["confidence_bins"].append({
                    "range": f"{low:.1f}-{high:.1f}",
                    "count": bin_count,
                    "mean_confidence": bin_confidence,
                    "accuracy": bin_accuracy,
                    "calibration_gap": bin_confidence - bin_accuracy
                })

        os.makedirs("router_analysis", exist_ok=True)

        # ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig("router_analysis/roc_curve.png")
        plt.close()

        # Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig("router_analysis/precision_recall_curve.png")
        plt.close()

        # Confidence distribution
        plt.figure(figsize=(12, 8))
        plt.hist([c for c, _ in easy_confidences], bins=20, alpha=0.5, label='Easy Questions')
        plt.hist([c for c, _ in hard_confidences], bins=20, alpha=0.5, label='Hard Questions')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution by Question Difficulty')
        plt.legend()
        plt.grid(True)
        plt.savefig("router_analysis/confidence_distribution.png")
        plt.close()

        # Calibration curve
        plt.figure(figsize=(10, 8))
        bin_ranges = [b["range"] for b in analysis["calibration"]["confidence_bins"]]
        mean_conf = [b["mean_confidence"] for b in analysis["calibration"]["confidence_bins"]]
        accuracy = [b["accuracy"] for b in analysis["calibration"]["confidence_bins"]]

        x = range(len(bin_ranges))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x, mean_conf, width, label='Mean Confidence')
        rects2 = ax.bar([i + width for i in x], accuracy, width, label='Accuracy')

        ax.set_xlabel('Confidence Range')
        ax.set_ylabel('Value')
        ax.set_title('Calibration: Confidence vs. Accuracy')
        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(bin_ranges)
        plt.xticks(rotation=45)
        ax.legend()

        fig.tight_layout()
        plt.savefig("router_analysis/calibration.png")
        plt.close()

        return analysis

def test_query_router():
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