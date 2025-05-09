import json
import random
import ollama
from util import format_question
from dataloader import load_hf_dataset
from model import Model

def main():

    print("Loading dataset...")
    dataset = load_hf_dataset("ai2_arc", "ARC-Challenge", split="test")

    print(f"Dataset contains {len(dataset)} questions")
    random_indices = random.sample(range(len(dataset)), 5)
    selected_questions = [dataset[i] for i in random_indices]

    model = Model("llama3")

    results_dict = {}

    for i, question_data in enumerate(selected_questions):
        print(f"\n---- Question {i} ------")

        prompt, answer, id = format_question(question_data)
        print(f"Prompt:\n {prompt}")
        print(f"Correct Answer: {answer}")

        response = model.query(prompt)
        print(f"\nModel response: \n{response}")
        print("\n" + "=" * 50)

        results_dict[id] = {
            "true": answer,
            "pred": response,
            "correct": (response == answer) if response else False
        }

    correct_count = 0
    for _, result in results_dict.items():
        correct_count += int(result["correct"])

    print(f"\nAccuracy: {correct_count / len(results_dict)}")

if __name__ == "__main__":
    main()