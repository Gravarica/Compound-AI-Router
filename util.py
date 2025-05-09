import re
import json
import time

def format_question(question_data):
    question = question_data['question']
    choices = question_data['choices']

    formatted_choices = ""
    for i, choice in enumerate(choices['text']):
        label = choices['label'][i]
        formatted_choices += f"{label}. {choice}\n"

    prompt = f"Question: {question}\n\nChoices:\n{formatted_choices}\n\nPlease select the correct answer (A, B, C or D). and only answer with the choice"
    return prompt, question_data['answerKey'], question_data['id']


def extract_answer(response):
    """Extract the answer (A, B, C, or D) from the model's response"""
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1)

    match = re.search(r'[Aa]nswer(?:\s+is)?(?:\s*:)?\s*([A-D])', response)
    if match:
        return match.group(1)

    match = re.search(r'([A-D])', response)
    if match:
        return match.group(1)

    return None


def save_results(results, filename="compound_system_results.json"):
    """
    Save results to a JSON file and print summary statistics.

    Args:
        results (List[Dict[str, Any]]): The results to save
        filename (str, optional): Output filename. Defaults to "compound_system_results.json".
    """
    # Calculate summary statistics
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / len(results) if results else 0

    # Distribution of LLM usage
    small_llm_count = sum(1 for r in results if 'small' in r['chosen_llm'].lower())
    large_llm_count = len(results) - small_llm_count

    # Accuracy by LLM
    small_llm_correct = sum(1 for r in results if 'small' in r['chosen_llm'].lower() and r['correct'])
    large_llm_correct = sum(1 for r in results if 'small' not in r['chosen_llm'].lower() and r['correct'])

    small_llm_accuracy = small_llm_correct / small_llm_count if small_llm_count else 0
    large_llm_accuracy = large_llm_correct / large_llm_count if large_llm_count else 0

    # Accuracy by true difficulty
    easy_queries = [r for r in results if r['true_difficulty'] == 'easy']
    hard_queries = [r for r in results if r['true_difficulty'] == 'hard']

    easy_correct = sum(1 for r in easy_queries if r['correct'])
    hard_correct = sum(1 for r in hard_queries if r['correct'])

    easy_accuracy = easy_correct / len(easy_queries) if easy_queries else 0
    hard_accuracy = hard_correct / len(hard_queries) if hard_queries else 0

    # Router accuracy
    router_correct = sum(1 for r in results if r['predicted_difficulty'] == r['true_difficulty'])
    router_accuracy = router_correct / len(results) if results else 0

    # False positives and negatives
    # False negative: easy question predicted as hard
    false_neg = sum(1 for r in results if r['true_difficulty'] == 'easy' and r['predicted_difficulty'] == 'hard')
    # False positive: hard question predicted as easy
    false_pos = sum(1 for r in results if r['true_difficulty'] == 'hard' and r['predicted_difficulty'] == 'easy')

    # Average times
    avg_routing_time = sum(r['routing_time_ms'] for r in results) / len(results) if results else 0
    avg_inference_time = sum(r['llm_latency_ms'] for r in results) / len(results) if results else 0
    avg_total_time = sum(r['total_time_ms'] for r in results) / len(results) if results else 0

    # Create summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(results),
        "overall_accuracy": accuracy,
        "routing_distribution": {
            "small_llm": small_llm_count,
            "large_llm": large_llm_count
        },
        "accuracy_by_llm": {
            "small_llm": small_llm_accuracy,
            "large_llm": large_llm_accuracy
        },
        "accuracy_by_difficulty": {
            "easy": easy_accuracy,
            "hard": hard_accuracy
        },
        "router_performance": {
            "accuracy": router_accuracy,
            "false_positives": false_pos,
            "false_negatives": false_neg
        },
        "average_times_ms": {
            "routing": avg_routing_time,
            "inference": avg_inference_time,
            "total": avg_total_time
        }
    }

    # Add summary to results
    output = {
        "summary": summary,
        "results": results
    }

    # Save to file
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary statistics
    print(f"\nResults saved to {filename}")
    print(f"\nSummary:")
    print(f"Overall accuracy: {accuracy:.2%}")
    print(f"\nRouter performance:")
    print(f"  Accuracy: {router_accuracy:.2%}")
    print(f"  False positives (hard as easy): {false_pos}")
    print(f"  False negatives (easy as hard): {false_neg}")
    print(f"\nLLM distribution:")
    print(f"  Small LLM: {small_llm_count}/{len(results)} ({small_llm_count / len(results):.2%})")
    print(f"  Large LLM: {large_llm_count}/{len(results)} ({large_llm_count / len(results):.2%})")
    print(f"\nAccuracy by LLM:")
    print(f"  Small LLM: {small_llm_accuracy:.2%}")
    print(f"  Large LLM: {large_llm_accuracy:.2%}")
    print(f"\nAccuracy by true difficulty:")
    print(f"  Easy questions: {easy_accuracy:.2%} ({easy_correct}/{len(easy_queries)})")
    print(f"  Hard questions: {hard_accuracy:.2%} ({hard_correct}/{len(hard_queries)})")
    print(f"\nAverage times:")
    print(f"  Routing: {avg_routing_time:.2f}ms")
    print(f"  Inference: {avg_inference_time:.2f}ms")
    print(f"  Total: {avg_total_time:.2f}ms")