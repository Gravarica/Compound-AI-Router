#!/usr/bin/env python
import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch

from query_router import QueryRouter
from src.data.dataloader import ARCDataManager


def analyze_router_thresholds(router_model_path, output_dir, num_samples=500):
    """
    Analyze the router performance across different thresholds.

    Args:
        router_model_path: Path to the router model
        output_dir: Directory to save analysis results
        num_samples: Number of samples to analyze
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading router from {router_model_path}")
    router = QueryRouter(model_name_or_path=router_model_path)

    print("Loading ARC data")
    data_manager = ARCDataManager()
    test_data = data_manager.get_arc_evaluation_set()

    import random
    random.seed(42)
    if num_samples and num_samples < len(test_data):
        test_data = random.sample(test_data, num_samples)

    print(f"Analyzing {len(test_data)} samples")

    predictions = []

    for item in tqdm(test_data, desc="Processing samples"):
        text = item['text']
        true_label = item['label']
        true_difficulty = "easy" if true_label == 0 else "hard"

        inputs = router.tokenizer(
            text,
            truncation=True,
            max_length=router.max_length,
            padding="max_length",
            return_tensors="pt"
        ).to(router.device)

        router.model.eval()
        with torch.no_grad():
            outputs = router.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()

            easy_prob = float(probs[0])
            hard_prob = float(probs[1])

            raw_prediction = "easy" if easy_prob > hard_prob else "hard"

        predictions.append({
            "id": item['id'],
            "true_label": true_label,
            "true_difficulty": true_difficulty,
            "easy_probability": easy_prob,
            "hard_probability": hard_prob,
            "raw_prediction": raw_prediction
        })

    # Analyze different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []

    for threshold in thresholds:
        # Apply threshold
        threshold_predictions = []
        for p in predictions:
            # If easy probability is above threshold, predict easy
            # Otherwise predict hard
            if p["easy_probability"] >= threshold:
                thresholded_prediction = "easy"
            else:
                thresholded_prediction = "hard"

            threshold_predictions.append({
                **p,
                "thresholded_prediction": thresholded_prediction,
                "correct": thresholded_prediction == p["true_difficulty"]
            })

        # Calculate metrics
        accuracy = sum(1 for p in threshold_predictions if p["correct"]) / len(threshold_predictions)

        true_positives = sum(1 for p in threshold_predictions
                             if p["true_difficulty"] == "hard" and p["thresholded_prediction"] == "hard")
        false_positives = sum(1 for p in threshold_predictions
                              if p["true_difficulty"] == "easy" and p["thresholded_prediction"] == "hard")
        true_negatives = sum(1 for p in threshold_predictions
                             if p["true_difficulty"] == "easy" and p["thresholded_prediction"] == "easy")
        false_negatives = sum(1 for p in threshold_predictions
                              if p["true_difficulty"] == "hard" and p["thresholded_prediction"] == "easy")

        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate ratio routed to small LLM
        small_llm_ratio = sum(1 for p in threshold_predictions if p["thresholded_prediction"] == "easy") / len(
            threshold_predictions)

        # Save result
        results.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "small_llm_ratio": small_llm_ratio
        })

    # Save raw predictions
    with open(os.path.join(output_dir, "raw_predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    # Save threshold analysis
    with open(os.path.join(output_dir, "threshold_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Create plots
    plot_threshold_metrics(results, output_dir)

    # Print summary of best thresholds
    print("\nBest thresholds by metric:")
    best_accuracy = max(results, key=lambda x: x["accuracy"])
    best_f1 = max(results, key=lambda x: x["f1"])
    best_precision = max(results, key=lambda x: x["precision"])
    best_recall = max(results, key=lambda x: x["recall"])

    print(f"Best accuracy: {best_accuracy['accuracy']:.4f} at threshold {best_accuracy['threshold']:.2f}")
    print(f"Best F1: {best_f1['f1']:.4f} at threshold {best_f1['threshold']:.2f}")
    print(f"Best precision: {best_precision['precision']:.4f} at threshold {best_precision['threshold']:.2f}")
    print(f"Best recall: {best_recall['recall']:.4f} at threshold {best_recall['threshold']:.2f}")

    return results


def plot_threshold_metrics(results, output_dir):
    """
    Create plots for threshold analysis.

    Args:
        results: List of dictionaries with metrics for each threshold
        output_dir: Directory to save plots
    """
    thresholds = [r["threshold"] for r in results]

    # Accuracy vs Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [r["accuracy"] for r in results], marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Router Accuracy vs. Confidence Threshold')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_threshold.png"))
    plt.close()

    # Precision, Recall, F1 vs Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [r["precision"] for r in results], marker='o', label='Precision')
    plt.plot(thresholds, [r["recall"] for r in results], marker='s', label='Recall')
    plt.plot(thresholds, [r["f1"] for r in results], marker='^', label='F1')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, F1 vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "prf_vs_threshold.png"))
    plt.close()

    # True/False Positives/Negatives vs Threshold
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, [r["true_positives"] for r in results], marker='o', label='True Positives')
    plt.plot(thresholds, [r["false_positives"] for r in results], marker='s', label='False Positives')
    plt.plot(thresholds, [r["true_negatives"] for r in results], marker='^', label='True Negatives')
    plt.plot(thresholds, [r["false_negatives"] for r in results], marker='d', label='False Negatives')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.title('Confusion Matrix Counts vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "confusion_vs_threshold.png"))
    plt.close()

    # Small LLM Ratio vs Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [r["small_llm_ratio"] for r in results], marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Ratio')
    plt.title('Proportion Routed to Small LLM vs. Threshold')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "small_llm_ratio_vs_threshold.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze router performance across thresholds")
    parser.add_argument("--router-model-path", type=str, default="./router_model",
                        help="Path to the router model")
    parser.add_argument("--output-dir", type=str, default="./router_threshold_analysis",
                        help="Directory to save analysis results")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of samples to analyze")

    args = parser.parse_args()

    analyze_router_thresholds(args.router_model_path, args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()