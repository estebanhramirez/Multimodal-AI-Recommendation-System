import requests
import json

url = "http://127.0.0.1:11434/api/chat"

payload = {
    "model": "llama2:latest",
    "messages": [ {"role": "user", "content": "What are you?"} ]
}

response = requests.post(url, json=payload, stream=True)

if response.status_code == 200:
    print("Streamming the response from Ollama...")

    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                json_data = json.loads(line)
                if "message" in json_data and "content" in json_data["message"]:
                    content = json_data["message"]["content"]
                    print(content, end="", flush=True)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from line: {line}")
        print()
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)