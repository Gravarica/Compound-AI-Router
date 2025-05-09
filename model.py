import ollama

class Model:
    def __init__(self, model_name):
        self.model_name = model_name

    def query(self, prompt):
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response['message']['content']