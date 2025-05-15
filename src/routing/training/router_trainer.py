import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback
)

from src.utils import setup_logging
from src.routing.training.custom_trainer import CustomTrainer
from src.data import RouterDataset

logger = setup_logging(name="router_trainer")

class RouterTrainer:

    def __init__(self, router):

        self.router = router

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
    ) -> Dict[str, Any]:
        """
        Fine-tune the router on the provided data.

        Args:
            train_data: Training data
            val_data: Validation data
            output_dir: Directory to save the fine-tuned model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Ratio of steps for warmup
            early_stopping_patience: Number of evaluations with no improvement after which training will stop
            early_stopping_threshold: Minimum change to qualify as improvement
            eval_steps: Number of steps between evaluations
            save_total_limit: Maximum number of checkpoints to keep

        Returns:
            Dictionary with training results
        """
        train_dataset = RouterDataset(train_data, self.router.tokenizer, self.router.max_length)
        val_dataset = RouterDataset(val_data, self.router.tokenizer, self.router.max_length)

        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Training dataset label distribution: {train_dataset.label_distribution}")

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
            fp16=False,
            bf16=False,
            dataloader_pin_memory=False,
            gradient_accumulation_steps=2,
            optim="adamw_torch",
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average="weighted"),
                "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
                "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
                'hard_f1': f1_score(labels, predictions, pos_label=1, zero_division=0),
                'easy_f1': f1_score(labels, predictions, pos_label=0, zero_division=0),
                'macro_f1': f1_score(labels, predictions, average="macro", zero_division=0),
            }

        class_counts = np.bincount(train_dataset.labels)
        total_samples = sum(class_counts)
        class_weights = torch.tensor(
            [total_samples / (len(class_counts) * count) for count in class_counts],
            dtype=torch.float32,
            device=self.router.device
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        trainer = CustomTrainer(
            model=self.router.model,
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
        train_result = trainer.train()

        logger.info("Evaluating the fine-tuned model...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")

        logger.info(f"Saving the fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)
        self.router.tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "eval_results.txt"), "w") as f:
            for key, value in eval_results.items():
                f.write(f"{key}: {value}\n")

        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")

        self.router._load_model()

        return {
            "train_results": train_result.metrics,
            "eval_results": eval_results
        }