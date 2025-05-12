from typing import Dict, List, Optional, Any

def format_question(question_data):
    question = question_data['question']
    choices = question_data['choices']

    formatted_choices = ""
    for i, choice in enumerate(choices['text']):
        label = choices['label'][i]
        formatted_choices += f"{label}. {choice}\n"

    prompt = f"Question: {question}\n\nChoices:\n{formatted_choices}\n\nPlease select the correct answer (A, B, C or D). and only answer with the choice"
    return prompt, question_data['answerKey'], question_data['id']


def parse_answer(response: str, choices: List[str]) -> Optional[str]:
    import re

    available_labels = choices['label']

    for label in available_labels:
        if re.search(fr'\b{label}\b', response):
            return label

    for label in available_labels:
        answer_match = re.search(fr'[Aa]nswer(?:\s+is)?(?:\s*:)?\s*{label}', response)
        if answer_match:
            return label

    for label in available_labels:
        if label in response:
            return label

    return available_labels[0] if available_labels else "A"

def create_llm_prompt(query: str, choices: Dict) -> str:

    formatted_choices = ""
    for i, choice in enumerate(choices['text']):
        label = choices['label'][i]
        formatted_choices += f"{label}. {choice}\n"

    prompt = f"""Question: {query}

    Choices:
    {formatted_choices}

    Please select the correct answer. Respond with only the letter of the correct choice (A, B, C, or D).
    """

    return prompt

def create_llm_router_prompt(query: str, choices: Dict) -> str:

    formatted_choices = ""
    for i, choice in enumerate(choices['text']):
        label = choices['label'][i]
        formatted_choices += f"{label}. {choice}\n"

    prompt = f"""You are an expert at analyzing questions.
    Your task is to determine if the following question is EASY or HARD.

    Question: {query}

    Choices:
    {formatted_choices}

    First, analyze the question step by step:
    1. What knowledge domains does this question involve?
    2. Does it require specialized knowledge?
    3. Does it involve multi-step reasoning?
    4. How much context or background information is needed?

    Based on your analysis, classify this question as either "EASY" or "HARD".
    Only respond with the single word "EASY" or "HARD".
    """
    return prompt

def create_bert_router_prompt(query: str, choices: Dict) -> str:
    formatted_choices = ""
    for i, choice in enumerate(choices['text']):
        label = choices['label'][i]
        formatted_choices += f"{label}. {choice}\n"

    return f"Question: {query}\n\nChoices:\n{formatted_choices}"