from datasets import load_dataset, DatasetDict, concatenate_datasets

def load_hf_dataset(path, name, split = None):
    arc_easy = load_dataset(path, name, split=split)
    return arc_easy


def test():
    easy = load_hf_dataset("ai2_arc", "ARC-Easy")
    challenge = load_hf_dataset("ai2_arc", "ARC-Challenge")

    easy = easy.map(lambda x: {**x, "difficulty": "easy"})
    challenge = challenge.map(lambda x: {**x, "difficulty": "hard"})

    combined_train = concatenate_datasets([easy["train"], challenge["train"]])
    combined_validation = concatenate_datasets([easy["validation"], challenge["validation"]])

    combined_train = combined_train.shuffle(seed=42)
    combined_validation = combined_validation.shuffle(seed=42)

    combined = DatasetDict({
        "train": combined_train,
        "validation": combined_validation
    })

    print(combined["train"][0])

test()
