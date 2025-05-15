# src/routing/training/custom_trainer.py
from transformers import Trainer
import torch

class CustomTrainer(Trainer):

    def __init__(self, loss_fn=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):

        device = next(model.parameters()).device

        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits.to(torch.float32)
        labels = labels.to(torch.long)

        if self.loss_fn:
            loss = self.loss_fn(logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss