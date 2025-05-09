import requests
import json
import sys


def chat_with_ollama(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    print("\nAssistant: ", end="", flush=True)
    with requests.post(url, headers=headers, json=data, stream=True) as response:
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                chunk = json_response.get("response", "")
                print(chunk, end="", flush=True)
    print("\n")


def main():
    model = "llama3"  # Default model

    # Allow model selection from command line
    if len(sys.argv) > 1:
        model = sys.argv[1]
        print(f"Using model: {model}")

    print(f"Chat with {model} (type 'exit' to quit)")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        chat_with_ollama(user_input, model)


if __name__ == "__main__":
    main()