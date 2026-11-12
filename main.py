import requests
import json



def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    EXAMPLE_RESTAURANT_PARAGRAPH = """
    Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range: $
    """

    EXAMPLE_OUTPUT = """
        {{
        "name": "Mar de Cortez",
        "location": "Santa Monica",
        "type": "casual taqueria",
        "food_style": "Baja-style seafood",
        "rating": 4.2,
        "price_range": 1,
        "signatures": [
            "beer-battered snapper tacos",
            "zesty octopus ceviche"
        ],
        "vibe": "salt-air energy",
        "environment": "a premier sun-drenched spot for open-air dining near the pier."
        "shortcomings": []
        }}
    """

    base_system_msg = f"""
    You are a 'Data Transformation Engine'. Your purpose is to convert unstructured
    textual data into structured JSON files with the specified fields and format given
    by the user.
    """

    base_user_prompt = f"""
    Task:
    Identify, within the given the 'Restaurant description', the name, location, restaurant type,
    food style, rating, price range, signature dishes, specialties and shortcomings, of the restaurant.
    Then, use this information to generate an output JSON file, as shown in the example below.

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}
    
    """
    return (base_system_msg, base_user_prompt)


def JSON_auto_repair_prompts(candidate_json_output, error_message):
    auto_repair_system_msg = """
    You are the final part of a 'Data Transformation Engine'.
    Your goal is to fix a given JSON file, according to an error message.
    """
    auto_repair_prompt = f"""
    Task:
    Repair the given 'candidate JSON', to fix the 'Error message'.
    Then output only the repaired JSON, without any message.

    Candidate JSON:
    {candidate_json_output}

    Error message:
    {error_message}
    """
    return (auto_repair_system_msg, auto_repair_prompt)



def main():
    url = "http://127.0.0.1:11434/api/chat"

    restaurant_paragraph = "In the heart of **Glendale**, **The Golden Hummus** is a **warm**, **family-style** **modern Lebanese** eatery. It earns a **4.6/5** for its communal atmosphere and standout signature hummus topped with spiced wagyu beef, offering a bustling environment that makes every guest feel like part of a large, hungry family. Price range: $$"
    base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph)

    payload = {
        "model": "llama2:latest",
        "messages": [ 
            {"role": "system", "content": base_system_msg},
            {"role": "user", "content": base_user_prompt}
            ]
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



if __name__ == "__main__":
    main()