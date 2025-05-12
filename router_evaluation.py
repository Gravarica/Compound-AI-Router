import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from query_router import QueryRouter
from dataloader import ARCDataManager
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_model_performance(router, test_data, output_dir):
    """Run detailed analysis of model performance and generate visualizations."""

    # Create test dataset
    from query_router import ArcDataset
    test_dataset = ArcDataset(test_data, router.tokenizer, router.max_length)

    # Set model to evaluation mode
    router.model.eval()

    # Get predictions with probabilities
    all_probs = []
    all_preds = []
    all_labels = []
    all_texts = []
    all_ids = []

    # Create a test dataloader
    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    # Make predictions
    for batch in test_dataloader:
        inputs = {k: v.to(router.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].cpu().numpy()

        with torch.no_grad():
            outputs = router.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)

        # Get indices in the original dataset
        batch_idx = range(len(preds))
        indices = [i + batch_idx[0] for i in range(len(batch_idx))]

        # Keep track of texts and IDs for error analysis
        for idx in indices:
            if idx < len(test_dataset):
                all_texts.append(test_dataset.texts[idx])
                all_ids.append(test_dataset.ids[idx])

    # Create a dataframe for analysis
    results_df = pd.DataFrame({
        'id': all_ids,
        'text': all_texts,
        'true_label': [router.label_map[label] for label in all_labels],
        'true_label_idx': all_labels,
        'pred_label': [router.label_map[pred] for pred in all_preds],
        'pred_label_idx': all_preds,
        'prob_easy': [prob[0] for prob in all_probs],
        'prob_hard': [prob[1] for prob in all_probs],
        'correct': [pred == label for pred, label in zip(all_preds, all_labels)]
    })

    # Save results dataframe
    results_df.to_csv(os.path.join(output_dir, 'detailed_predictions.csv'), index=False)

    # Analysis 1: Distribution of probabilities
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df, x='prob_hard', hue='true_label', bins=30, alpha=0.6)
    plt.title('Distribution of Prediction Probabilities for Hard Class')
    plt.xlabel('Probability of "Hard" Class')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'))
    plt.close()

    # Analysis 2: ROC Curve
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(results_df['true_label_idx'], results_df['prob_hard'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # Analysis 3: Precision-Recall Curve
    plt.figure(figsize=(8, 8))
    precision, recall, _ = precision_recall_curve(results_df['true_label_idx'], results_df['prob_hard'])
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()

    # Analysis 4: Confidence analysis
    plt.figure(figsize=(10, 6))
    # Calculate confidence as maximum probability
    results_df['confidence'] = results_df[['prob_easy', 'prob_hard']].max(axis=1)

    # Group by whether prediction is correct and calculate mean confidence
    confidence_by_correctness = results_df.groupby('correct')['confidence'].mean().reset_index()
    sns.barplot(x='correct', y='confidence', data=confidence_by_correctness)
    plt.title('Average Confidence by Prediction Correctness')
    plt.xlabel('Correct Prediction')
    plt.ylabel('Average Confidence')
    plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'))
    plt.close()

    # Analysis 5: Error analysis
    # Extract misclassified examples
    misclassified = results_df[~results_df['correct']]

    # Count easy vs hard misclassifications
    error_types = Counter()
    for _, row in misclassified.iterrows():
        error_types[f"{row['true_label']} as {row['pred_label']}"] += 1

    # Create error analysis table
    with open(os.path.join(output_dir, 'error_analysis.txt'), 'w') as f:
        f.write("ERROR ANALYSIS\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total examples: {len(results_df)}\n")
        f.write(f"Correct predictions: {results_df['correct'].sum()} ({results_df['correct'].mean():.2%})\n")
        f.write(f"Incorrect predictions: {len(misclassified)} ({len(misclassified) / len(results_df):.2%})\n\n")

        f.write("Error types:\n")
        for error_type, count in error_types.items():
            f.write(f"  {error_type}: {count} ({count / len(misclassified):.2%} of errors)\n")

        f.write("\nTop misclassified examples:\n")
        # Sort by confidence to find most confidently wrong predictions
        top_errors = misclassified.sort_values('confidence', ascending=False).head(10)
        for i, (_, row) in enumerate(top_errors.iterrows()):
            f.write(f"\nError {i + 1} (ID: {row['id']})\n")
            f.write(
                f"  True: {row['true_label']}, Predicted: {row['pred_label']}, Confidence: {row['confidence']:.4f}\n")
            f.write(f"  Text (truncated): {row['text'][:200]}...\n")

    # Return summary dict
    return {
        'total_examples': len(results_df),
        'accuracy': results_df['correct'].mean(),
        'roc_auc': roc_auc,
        'error_types': dict(error_types),
        'avg_confidence_correct':
            confidence_by_correctness.loc[confidence_by_correctness['correct'], 'confidence'].values[0],
        'avg_confidence_incorrect':
            confidence_by_correctness.loc[~confidence_by_correctness['correct'], 'confidence'].values[0]
    }


def test_on_sample(router, text, output_dir, example_idx=0):
    """Test the router with a sample query and visualize attention."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get model prediction
    difficulty = router.predict_difficulty(text)

    # Tokenize the query
    inputs = router.tokenizer(
        text,
        truncation=True,
        max_length=router.max_length,
        padding="max_length",
        return_tensors="pt"
    ).to(router.device)

    # Set model to evaluation mode
    router.model.eval()

    # Run with attention output
    with torch.no_grad():
        outputs = router.model(**inputs, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions  # Tuple of attention tensors

        # Get prediction and confidence
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
        pred_class = torch.argmax(logits, dim=1).item()

    # Save prediction info
    result = {
        'text': text[:500] + ('...' if len(text) > 500 else ''),
        'predicted_difficulty': difficulty,
        'confidence': {
            'easy': float(probs[0]),
            'hard': float(probs[1])
        }
    }

    # Save result
    with open(os.path.join(output_dir, f'sample_{example_idx}_result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # Visualize attention for the [CLS] token (first token) from the last layer
    # This varies by model architecture - works for BERT-like models
    if attentions:
        # Get last layer attention
        last_layer_attention = attentions[-1][0].cpu().numpy()  # shape: [num_heads, seq_len, seq_len]

        # Average over heads
        avg_attention = last_layer_attention.mean(axis=0)  # shape: [seq_len, seq_len]

        # Get attention from CLS token to all other tokens
        cls_attention = avg_attention[0]  # shape: [seq_len]

        # Decode tokens for visualization
        tokens = router.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Truncate to actual tokens (remove padding)
        seq_len = len([t for t in tokens if t != router.tokenizer.pad_token])
        tokens = tokens[:seq_len]
        cls_attention = cls_attention[:seq_len]

        # Plot attention weights
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(tokens[:50])), cls_attention[:50], alpha=0.6)
        plt.xticks(range(len(tokens[:50])), tokens[:50], rotation=90)
        plt.title(f'Attention from [CLS] token - Predicted: {difficulty}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{example_idx}_attention.png'))
        plt.close()

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate the QueryRouter model and analyze performance")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model for evaluation")
    parser.add_argument("--output_dir", type=str, default="./router_evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--num_examples", type=int, default=10,
                        help="Number of examples to analyze in detail")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize ARCDataManager and load data
    logger.info("Loading ARC data...")
    manager = ARCDataManager()
    _, _, test_data = manager.create_router_training_data()

    # Initialize QueryRouter with the fine-tuned model
    logger.info(f"Loading router model from {args.model_path}")
    router = QueryRouter(
        model_name_or_path=args.model_path
    )

    logger.info("Running router evaluation...")
    eval_results = router.evaluate_router(test_data)

    eval_output_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(eval_output_path, "w") as f:
        cleaned_results = {}
        for k, v in eval_results.items():
            if k == 'misclassified_examples':
                cleaned_results[k] = v
            elif isinstance(v, (list, dict, str)):
                cleaned_results[k] = v
            else:
                cleaned_results[k] = str(v)

        json.dump(cleaned_results, f, indent=2)

    logger.info("Running detailed performance analysis...")
    analysis_results = analyze_model_performance(router, test_data, args.output_dir)

    # Save analysis results
    with open(os.path.join(args.output_dir, "performance_analysis.json"), "w") as f:
        cleaned_results_performance = {}
        for k,v in analysis_results.items():
            if k == 'error_types':
                cleaned_results_performance[k] = v
            else:
                cleaned_results_performance[k] = str(v)

        json.dump(cleaned_results_performance, f, indent=2)

    logger.info(f"Testing router with {args.num_examples} examples...")

    import random
    random.seed(42)
    sample_indices = random.sample(range(len(test_data)), min(args.num_examples, len(test_data)))

    for i, idx in enumerate(sample_indices):
        sample_text = test_data[idx]['text']
        logger.info(f"Testing sample {i + 1}/{len(sample_indices)}...")
        test_on_sample(router, sample_text, args.output_dir, i)

    # Print summary
    logger.info(f"Evaluation complete! Results saved to {args.output_dir}")
    logger.info(f"Accuracy: {eval_results['accuracy']:.4f}")
    logger.info(f"F1 Score: {eval_results['f1']:.4f}")
    logger.info(f"Precision: {eval_results['precision']:.4f}")
    logger.info(f"Recall: {eval_results['recall']:.4f}")
    logger.info(f"Misclassified examples: {eval_results['misclassified_count']}")
    logger.info(f"ROC AUC: {analysis_results['roc_auc']:.4f}")
    logger.info(f"Average confidence when correct: {analysis_results['avg_confidence_correct']:.4f}")
    logger.info(f"Average confidence when incorrect: {analysis_results['avg_confidence_incorrect']:.4f}")


if __name__ == "__main__":
    main()