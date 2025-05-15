# src/routing/evaluation/router_evaluator.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from torch.utils.data import DataLoader

from src.utils.logging import setup_logging
from src.data import RouterDataset

logger = setup_logging(name="router_evaluator")


class RouterEvaluator:
    """
    Handles evaluation for transformer-based routers.
    """

    def __init__(self, router, output_dir: str = "results/router_evaluation"):
        """
        Initialize the evaluator.

        Args:
            router: The router to evaluate
            output_dir: Directory to save evaluation results
        """
        self.router = router
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_router(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate the router on test data.

        Args:
            test_data: Test data

        Returns:
            Dictionary with evaluation results
        """
        test_dataset = RouterDataset(test_data, self.router.tokenizer, self.router.max_length)

        self.router.model.eval()

        all_preds = []
        all_labels = []
        all_texts = []
        all_ids = []
        all_probs = []

        test_dataloader = DataLoader(test_dataset, batch_size=8)

        processed_indices = []

        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            inputs = {k: v.to(self.router.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to('cpu').numpy()

            with torch.no_grad():
                outputs = self.router.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)

            start_idx = batch_idx * test_dataloader.batch_size
            for i, pred in enumerate(preds):
                idx = start_idx + i
                if idx < len(test_dataset):
                    processed_indices.append(idx)
                    all_texts.append(test_dataset.texts[idx])
                    all_ids.append(test_dataset.ids[idx])

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        # Create confusion matrix plot
        self._plot_confusion_matrix(cm, accuracy, f1)

        # Identify misclassified examples
        misclassified = self._get_misclassified_examples(all_ids, all_texts, all_labels, all_preds)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_plot_path': os.path.join(self.output_dir, 'confusion_matrix.png'),
            'misclassified_count': len(misclassified),
            'misclassified_examples': misclassified[:10]  # Include only first 10 for brevity
        }

        # Save detailed results
        self._save_detailed_results(all_ids, all_texts, all_labels, all_preds, all_probs)

        return results

    def _plot_confusion_matrix(self, cm, accuracy, f1):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            accuracy: Model accuracy
            f1: Model F1 score
        """
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(self.router.label_map.values()),
                    yticklabels=list(self.router.label_map.values()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, F1: {f1:.4f}')

        cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()

    def _get_misclassified_examples(self, all_ids, all_texts, all_labels, all_preds):
        """
        Get misclassified examples.

        Args:
            all_ids: List of example IDs
            all_texts: List of example texts
            all_labels: List of true labels
            all_preds: List of predicted labels

        Returns:
            List of misclassified examples
        """
        misclassified = []
        for i in range(len(all_preds)):
            if all_preds[i] != all_labels[i]:
                misclassified.append({
                    'id': all_ids[i],
                    'text': all_texts[i],
                    'true_label': self.router.label_map[all_labels[i]],
                    'predicted_label': self.router.label_map[all_preds[i]]
                })
        return misclassified

    def _save_detailed_results(self, all_ids, all_texts, all_labels, all_preds, all_probs):
        """
        Save detailed evaluation results.

        Args:
            all_ids: List of example IDs
            all_texts: List of example texts
            all_labels: List of true labels
            all_preds: List of predicted labels
            all_probs: List of prediction probabilities
        """
        import pandas as pd

        # Create dataframe with all results
        results_df = pd.DataFrame({
            'id': all_ids,
            'text': [t[:500] + ('...' if len(t) > 500 else '') for t in all_texts],
            'true_label': [self.router.label_map[label] for label in all_labels],
            'pred_label': [self.router.label_map[pred] for pred in all_preds],
            'prob_easy': [prob[0] for prob in all_probs],
            'prob_hard': [prob[1] for prob in all_probs],
            'correct': [pred == label for pred, label in zip(all_preds, all_labels)]
        })

        # Save to CSV
        results_df.to_csv(os.path.join(self.output_dir, 'detailed_predictions.csv'), index=False)

    def analyze_confidence_distribution(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze confidence distribution.

        Args:
            test_data: Test data

        Returns:
            Dictionary with analysis results
        """
        test_dataset = RouterDataset(test_data, self.router.tokenizer, self.router.max_length)

        self.router.model.eval()

        easy_confidences = []
        hard_confidences = []
        correct_confidences = []
        incorrect_confidences = []

        true_labels = []
        pred_probs = []

        test_dataloader = DataLoader(test_dataset, batch_size=8)

        for batch in tqdm(test_dataloader, desc='Analyzing confidence distribution'):
            inputs = {k: v.to(self.router.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to('cpu').numpy()

            with torch.no_grad():
                outputs = self.router.model(**inputs)
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
                pred_probs.append(prob[0])  # Probability of 'easy' class

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
        roc_auc = auc(fpr, tpr)

        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(true_labels, pred_probs)

        # Find optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Calculate thresholds for different target accuracies
        thresholds_by_accuracy = self._calculate_thresholds_by_accuracy(pred_probs, true_labels)

        # Prepare analysis results
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
                "confidence_bins": self._calculate_calibration_bins(easy_confidences + hard_confidences),
            }
        }

        # Create visualizations
        self._create_confidence_visualizations(easy_confidences, hard_confidences, fpr, tpr, roc_auc, precision, recall,
                                               analysis["calibration"]["confidence_bins"])

        return analysis

    def _calculate_thresholds_by_accuracy(self, pred_probs, true_labels):
        """
        Calculate thresholds for different target accuracies.

        Args:
            pred_probs: Prediction probabilities
            true_labels: True labels

        Returns:
            Dictionary mapping target accuracies to thresholds
        """
        thresholds_by_accuracy = {}
        for acc_req in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            accuracies = []
            for threshold in np.arange(0.1, 1.0, 0.01):
                preds = [1 if p < threshold else 0 for p in pred_probs]  # 1 = hard, 0 = easy
                accuracy = np.mean([p == t for p, t in zip(preds, true_labels)])
                accuracies.append((threshold, accuracy))

            closest = min(accuracies, key=lambda x: abs(x[1] - acc_req))
            thresholds_by_accuracy[acc_req] = closest

        return thresholds_by_accuracy

    def _calculate_calibration_bins(self, confidences, num_bins=10):
        """
        Calculate calibration bins.

        Args:
            confidences: List of (confidence, correct) tuples
            num_bins: Number of bins

        Returns:
            List of calibration bin statistics
        """
        bin_edges = np.linspace(0, 1.0, num_bins + 1)
        bins = []

        for i in range(len(bin_edges) - 1):
            low, high = bin_edges[i], bin_edges[i + 1]
            bin_samples = [(c, p) for c, p in confidences if low <= c < high]

            if bin_samples:
                bin_accuracy = np.mean([p for _, p in bin_samples])
                bin_confidence = np.mean([c for c, _ in bin_samples])
                bin_count = len(bin_samples)

                bins.append({
                    "range": f"{low:.1f}-{high:.1f}",
                    "count": bin_count,
                    "mean_confidence": bin_confidence,
                    "accuracy": bin_accuracy,
                    "calibration_gap": bin_confidence - bin_accuracy
                })

        return bins

    # src/routing/evaluation/router_evaluator.py (continued)
    def _create_confidence_visualizations(self, easy_confidences, hard_confidences, fpr, tpr, roc_auc, precision,
                                          recall, calibration_bins):
        """
        Create visualizations of confidence distributions.

        Args:
            easy_confidences: Confidences for easy examples
            hard_confidences: Confidences for hard examples
            fpr: False positive rates for ROC curve
            tpr: True positive rates for ROC curve
            roc_auc: Area under ROC curve
            precision: Precision values for PR curve
            recall: Recall values for PR curve
            calibration_bins: Calibration bin statistics
        """
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        vis_dir = os.path.join(self.output_dir, "visualizations")

        # 1. ROC curve
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
        plt.savefig(os.path.join(vis_dir, "roc_curve.png"))
        plt.close()

        # 2. Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig(os.path.join(vis_dir, "precision_recall_curve.png"))
        plt.close()

        # 3. Confidence distribution
        plt.figure(figsize=(12, 8))
        plt.hist([c for c, _ in easy_confidences], bins=20, alpha=0.5, label='Easy Questions')
        plt.hist([c for c, _ in hard_confidences], bins=20, alpha=0.5, label='Hard Questions')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution by Question Difficulty')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(vis_dir, "confidence_distribution.png"))
        plt.close()

        # 4. Calibration curve
        if calibration_bins:
            bin_ranges = [b["range"] for b in calibration_bins]
            mean_conf = [b["mean_confidence"] for b in calibration_bins]
            accuracy = [b["accuracy"] for b in calibration_bins]

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
            plt.savefig(os.path.join(vis_dir, "calibration.png"))
            plt.close()

        # 5. Correct vs. Incorrect confidence
        plt.figure(figsize=(10, 6))
        correct_confs = [c for c, is_correct in easy_confidences + hard_confidences if is_correct]
        incorrect_confs = [c for c, is_correct in easy_confidences + hard_confidences if not is_correct]

        plt.hist(correct_confs, bins=20, alpha=0.5, label='Correct Predictions')
        plt.hist(incorrect_confs, bins=20, alpha=0.5, label='Incorrect Predictions')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution by Prediction Correctness')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(vis_dir, "confidence_by_correctness.png"))
        plt.close()