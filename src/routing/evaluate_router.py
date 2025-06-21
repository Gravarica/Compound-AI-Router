from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.routing import TransformerRouter
from src.data.dataloader import ARCDataManager
from typing import List, Dict
from tqdm import tqdm


def evaluate_router(router: TransformerRouter, test_data: List[Dict], confidence_threshold: float = 0.7):
    y_true = []
    y_pred = []

    for item in tqdm(test_data, desc="Routing evaluation"):
        text = item['text']
        true_label = item['label']
        predicted_label, confidence = router.predict_difficulty(query_text=text, confidence_threshold=confidence_threshold)
        predicted_label_id = router.inv_label_map[predicted_label]

        y_true.append(true_label)
        y_pred.append(predicted_label_id)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=['easy', 'hard']))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\nAccuracy: {accuracy:.4f}")

manager = ARCDataManager()

router = TransformerRouter(model_name_or_path="model-store/router_model")

data = manager.get_arc_evaluation_set()

for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
    print("Evaluating router with confidence threshold", t)
    evaluate_router(router, data, confidence_threshold=t)
