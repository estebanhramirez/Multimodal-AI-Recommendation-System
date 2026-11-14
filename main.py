# ==========================================================================================================
# Imports
# ==========================================================================================================

import base64

import requests
import json

from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional



# ==========================================================================================================
# Data Schemas
# ==========================================================================================================

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



# ==========================================================================================================
# Prompt templates
# ==========================================================================================================

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
    You are a 'Data Transformation Engine'.
    Your purpose is to convert unstructured textual data into structured JSON files.
    """

    base_user_prompt = f"""
    Task:
    Identify, within the given 'Restaurant description', the name, location, restaurant type, food style, rating, price range, signature dishes, specialties and shortcomings, of the restaurant. Use this information to generate an output JSON file, as shown in the example below. For the 'price range' field, convert the dollar signs (e.g., $$) into an integer representing the number of dollar symbols.
    Hide your reasoning and thought process, and ommit any messages or explanations: only output the final JSON file, as shown in the example below.

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
    You are a 'Data Transformation Engine'.
    Your purpose is to convert unstructured textual data into structured JSON files.
    """

    auto_repair_prompt = f"""
    Task:
    Identify, within the given 'Candidate JSON', the name, location, restaurant type, food style, rating, price range, signature dishes, specialties and shortcomings, of the restaurant.
    Use this information to generate an proper JSON file.
    Hide your reasoning and thought process, and ommit any messages or explanations: only output the final JSON file.
    
    Candidate JSON:
    {candidate_json_output}

    The following error message can help you identify the issues in the Candidate JSON:
    {error_message}
    """

    return (auto_repair_system_msg, auto_repair_prompt)


def image_caption_prompt_template(food_name):
    # food_name: the food name of the recipe

    ### Step 3.1: Design the prompts
    image_caption_system_msg = "Your objective is to generate high-quality textual descriptions of images."
    image_caption_prompt_txt = """
    Task:
    list the ingredients in the image.
    """

    return (image_caption_system_msg, image_caption_prompt_txt)



# ==========================================================================================================
# LLM functions
# ==========================================================================================================

def unwrap_response(response):
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
        "model": "llama3.2:3b",
        "messages": [ 
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
            ]
    }

    response = requests.post(url, json=payload, stream=True)

    return unwrap_response(response)


def vision_llm(system_msg, prompt_txt, image_path):
    """
    Args:
        system_msg (str): The system message to provide context for the LLM.
        prompt_txt (str): The user prompt containing the task and input data.
        image_path (str): The file path of the input image.
    Returns:
        str: The final response from the LLM after processing the input.
    """

    url = "http://127.0.0.1:11434/api/chat"

    with open(image_path, "rb") as image_file:
        image = image_file.read()
        encoded_image = base64.b64encode(image).decode("utf-8")

    messages = [
        {
            "role": "system",
            "content": system_msg
        },
        {
            "role": "user",
            "content": prompt_txt,
            "images": [encoded_image]
        }
    ]

    payload = {
        "model": "qwen3-vl:4b",
        "messages": messages
    }

    response = requests.post(url, json=payload, stream=True)

    return unwrap_response(response)



# ==========================================================================================================
# Execution
# ==========================================================================================================

def legacy_main():
    # 1: Load the data
    file_path = "../Data/California-Culinary-Map.txt"
    with open(file_path, 'r') as file:
        data = file.read()

    # 2: Split the restaurant paragraphs into a Python list
    ## 2.1: Split the restaurant paragraphs into list (hint: use .split('\n\n')
    restaurant_list = data.split('\n\n')

    ## 2.2: Since the first item is the dataset name, we remove it
    restaurant_list = restaurant_list[1:]

    structured_restaurant_lists = []
    for i, restaurant_paragraph in enumerate(restaurant_list):
    
        base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph)

        candidate_json_output = llm_model(system_msg=base_system_msg, user_prompt=base_user_prompt)

        error_flag = True
        while error_flag:
            print(f"Validating JSON Output: {candidate_json_output}")
            try:
                restaurant_data = Restaurant.model_validate_json(candidate_json_output)
                print(f"Success! Structured the {i}-th paragraph: {restaurant_data.name}")
                error_flag = False
            except ValidationError as error_message:
                auto_repair_system_msg, auto_repair_prompt = JSON_auto_repair_prompts(candidate_json_output, error_message)
                candidate_json_output = llm_model(system_msg=auto_repair_system_msg, user_prompt=auto_repair_prompt)
                print(f"Validation failed: {error_message.json()}")

        structured_restaurant_lists.append(candidate_json_output)

    print(len(structured_restaurant_lists))
    print(type(structured_restaurant_lists))

    structured_restaurant_lists_json = [json.loads(response) for response in structured_restaurant_lists]

    #For each item in the restaurant list, assign it with an itemId to be consistent with the one in the user review data:
    for i, response in enumerate(structured_restaurant_lists_json):
        response['itemId'] = 1000001 + i
        structured_restaurant_lists_json[i] = response

    filename = 'structured_restaurant_data.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(structured_restaurant_lists_json, f, indent=4)


def main():
    base_image_caption_system_msg, base_image_caption_prompt_txt = image_caption_prompt_template("Classic Margherita Pizza")

    response =  vision_llm(base_image_caption_system_msg, base_image_caption_prompt_txt, "../Data/synthetic_recipe_images/recipe2.png")
    print(response)



if __name__ == "__main__":
    legacy_main()