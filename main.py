import requests
import json

from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional



class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)



def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    EXAMPLE_RESTAURANT_PARAGRAPH = """
    Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, **casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** rating, it captures the salt-air energy of the coast through its signature beer-battered snapper tacos and zesty octopus ceviche, making it a premier spot for open-air dining near the pier. Price range: $
    """

    EXAMPLE_OUTPUT = """
        {
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
        }
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
    Hide your reasoning and thought process, and ommit any messages or explanations.
    Only output the final JSON file, as shown in the example below.

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
    Hide your reasoning and thought process, and ommit any messages or explanations.
    Output only the repaired JSON.

    Candidate JSON:
    {candidate_json_output}

    Error message:
    {error_message}
    """

    return (auto_repair_system_msg, auto_repair_prompt)


def llm_model(system_msg, user_prompt):
    """
    Function to send a request to the LLM model and stream the response.

    Args:
        system_msg (str): The system message to provide context for the LLM.
        user_prompt (str): The user prompt containing the task and input data.

    Returns:
        str: The final response from the LLM after processing the input.
    """

    url = "http://127.0.0.1:11434/api/chat"

    payload = {
        "model": "llama2:latest",
        "messages": [ 
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
            ]
    }

    response = requests.post(url, json=payload, stream=True)

    if response.status_code == 200:
        final_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        content = json_data["message"]["content"]
                        final_response += content
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from line: {line}")
        return(final_response)
    else:
        return(f"Request failed with status code: {response.status_code}")



def main():
    restaurant_paragraph = "In the heart of **Glendale**, **The Golden Hummus** is a **warm**, **family-style** **modern Lebanese** eatery. It earns a **4.6/5** for its communal atmosphere and standout signature hummus topped with spiced wagyu beef, offering a bustling environment that makes every guest feel like part of a large, hungry family. Price range: $$"
    base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph)


    candidate_json_output = llm_model(system_msg=base_system_msg, user_prompt=base_user_prompt)

    ### 2.2: Validation and Auto Correction loop on the output 
    error_flag = True
    while error_flag:
        print(f"Validating JSON Output: {candidate_json_output}")
        try:
            restaurant_data = Restaurant.model_validate_json(candidate_json_output)
            print(f"Success! Validated: {restaurant_data.name}")
            error_flag = False
        except ValidationError as error_message:
            auto_repair_system_msg, auto_repair_prompt = JSON_auto_repair_prompts(candidate_json_output, error_message)
            candidate_json_output = llm_model(system_msg=auto_repair_system_msg, user_prompt=auto_repair_prompt)
            print(f"Validation failed: {error_message.json()}")

    print(f"Final Validated JSON Output: {candidate_json_output}")



if __name__ == "__main__":
    main()