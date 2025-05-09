import re

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