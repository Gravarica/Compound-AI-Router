from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from model import Model
from dataloader import load_hf_dataset
from util import format_question
from tqdm import tqdm

def calculate_metrics(results) -> dict:

    valid_results = {q_id: data for q_id, data in results.items()
                     if data['predicted_answer']}

    if not valid_results:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "invalid_rate": 1.0,
            "confusion_matrix": None
        }

    true_answers = [data['true_answer'] for data in valid_results.values()]
    predicted_answers = [data['predicted_answer'] for data in valid_results.values()]

    accuracy = accuracy_score(true_answers, predicted_answers)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_answers,
        predicted_answers,
        average='macro'
    )

    unique_labels = sorted(set(true_answers + predicted_answers))
    cm = confusion_matrix(true_answers, predicted_answers, labels=unique_labels)

    invalid_rate = 1.0 - (len(valid_results) / len(results))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "invalid_rate": invalid_rate,
        "confusion_matrix": (cm, unique_labels)
    }

def evaluate_question_by_type(results):
    question_types = defaultdict(list)

    for q_id, data in results.items():
        question = data['question'].lower()

        if any(word in question for word in ['most likely', 'probably', 'predict']):
            category = 'Prediction'
        elif any(word in question for word in ['cause', 'effect', 'result', 'lead to']):
            category = 'Causality'
        elif any(word in question for word in ['compare', 'contrast', 'difference', 'similar']):
            category = 'Comparison'
        elif any(word in question for word in ['define', 'meaning', 'what is']):
            category = 'Definition'
        elif any(word in question for word in ['example', 'instance']):
            category = 'Examples'
        else:
            category = 'General Knowledge'

        question_types[category].append(q_id)

    category_metrics = {}
    for category, q_ids in question_types.items():
        cat_results = {q_id: results[q_id] for q_id in q_ids}
        valid_results = {q_id: data for q_id, data in cat_results.items()
                         if data['predicted_answer']}

        if valid_results:
            true_answers = [data['true_answer'] for data in valid_results.values()]
            predicted_answers = [data['predicted_answer'] for data in valid_results.values()]
            accuracy = accuracy_score(true_answers, predicted_answers)

            category_metrics[category] = {
                'count': len(cat_results),
                'accuracy': accuracy,
                'valid_count': len(valid_results),
            }
        else:
            category_metrics[category] = {
                'count': len(cat_results),
                'accuracy': 0.0,
                'valid_count': 0,
            }

    return category_metrics

def plot_metrics(metrics, model_name):

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    performance_metrics = ['accuracy', 'precision', 'recall', 'f1']
    values = [metrics[m] for m in performance_metrics]

    axs[0, 0].bar(performance_metrics, values, color='skyblue')
    axs[0, 0].set_ylim(0, 1.0)
    axs[0, 0].set_title(f'{model_name} - Performance Metrics')
    axs[0, 0].set_ylabel('Score')

    cm, labels = metrics['confusion_matrix']
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=axs[0, 1])
    axs[0, 1].set_title('Confusion Matrix')
    axs[0, 1].set_ylabel('True')
    axs[0, 1].set_xlabel('Predicted')

    if 'categories' in metrics:
        categories = list(metrics['categories'].keys())
        category_acc = [metrics['categories'][c]['accuracy'] for c in categories]
        category_counts = [metrics['categories'][c]['count'] for c in categories]

        axs[1, 0].bar(categories, category_acc, color='lightgreen')
        axs[1, 0].set_ylim(0, 1.0)
        axs[1, 0].set_title('Accuracy by Question Category')
        axs[1, 0].set_ylabel('Accuracy')
        plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha='right')

        for i, count in enumerate(category_counts):
            axs[1, 0].text(i, 0.05, f'n={count}', ha='center')

    if 'answer_distribution' in metrics:
        labels = list(metrics['answer_distribution'].keys())
        sizes = list(metrics['answer_distribution'].values())

        axs[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        axs[1, 1].set_title('Answer Distribution')

    plt.tight_layout()
    plt.savefig(f'{model_name}_metrics.png')
    plt.close()

def main():

    model = Model("llama3")

    dataset = load_hf_dataset("ai2_arc", "ARC-Challenge", split="test")

    results = {}

    for i, question_data in enumerate(tqdm(dataset, desc = "Processing questions")):
        prompt, answer, id = format_question(question_data)

        response = model.query(prompt)

        results[id] = {
            "question": question_data['question'],
            "choices": question_data['choices'],
            "true_answer": answer,
            "predicted_answer": response,
            "correct": (response == answer) if response else False
        }

    metrics = calculate_metrics(results)
    category_metrics = evaluate_question_by_type(results)
    metrics['categories'] = category_metrics

    print(f"Metrics for {model.model_name}:")
    print(metrics)

if __name__ == "__main__":
    main()
