import os
import argparse
import json
from query_router import QueryRouter
from src.data.dataloader import ARCDataManager
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the QueryRouter model")

    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Base model to use for the router (default: distilbert-base-uncased)")
    parser.add_argument("--output_dir", type=str, default="./router_model",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate the model, no training")
    parser.add_argument("--eval_model_path", type=str, default=None,
                        help="Path to the fine-tuned model for evaluation")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Loading ARC data...")
    manager = ARCDataManager()
    train_data, val_data, test_data = manager.create_router_training_data()

    if args.eval_only and args.eval_model_path:
        logger.info(f"Initializing router with pre-trained model from {args.eval_model_path}")
        router = QueryRouter(
            model_name_or_path=args.eval_model_path,
            max_length=args.max_length
        )
    else:
        logger.info(f"Initializing router with base model {args.model_name}")
        router = QueryRouter(
            model_name_or_path=args.model_name,
            max_length=args.max_length
        )

    if not args.eval_only:
        logger.info("Starting router fine-tuning...")
        router.fine_tune(
            train_data=train_data,
            val_data=val_data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        router.load_fine_tuned_model(args.output_dir)

    logger.info("Evaluating router performance...")
    eval_results = router.evaluate_router(test_data)

    eval_output_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(eval_output_path, "w") as f:
        cleaned_results = {}
        for k, v in eval_results.items():
            if k == 'misclassified_examples':
                cleaned_results[k] = v
            elif isinstance(v, (list, dict)):
                cleaned_results[k] = v
            else:
                cleaned_results[k] = float(v)

        json.dump(cleaned_results, f, indent=2)

    logger.info(f"Evaluation complete! Results saved to {eval_output_path}")
    logger.info(f"Accuracy: {eval_results['accuracy']:.4f}")
    logger.info(f"F1 Score: {eval_results['f1']:.4f}")
    logger.info(f"Precision: {eval_results['precision']:.4f}")
    logger.info(f"Recall: {eval_results['recall']:.4f}")
    logger.info(f"Misclassified examples: {eval_results['misclassified_count']}")
    logger.info(f"Confusion matrix plot saved to: {eval_results['confusion_matrix_plot_path']}")


if __name__ == "__main__":
    main()