# ==================================================================================
# Imports
# ==================================================================================

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil
import io
import requests
import unittest
from unittest.mock import patch
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional



# ==========================================================================================================
# Global variables
# ==========================================================================================================

FILEPATH = '../Data/structured_restaurant_data.json'
BACKUP_PATH = '../Data/structured_restaurant_data.json.bak'


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
# Helper functions
# ==========================================================================================================

def load_data(file_path):
    with open(file_path, 'r') as file:
        raw_data = file.read()
        data = json.loads(raw_data)
    return (data)


# ==========================================================================================================
# Prompt Templates
# ==========================================================================================================

JSON_FORMAT = """
        {
        "name": str,
        "location": str,
        "type": str,
        "food_style": str,
        "rating": Optional[float],
        "price_range": Optional[int],
        "signatures": List[str],
        "vibe": Optional[str],
        "environment": str,
        "shortcomings": List[str]
        }
    """

def restaurant_data_structure_prompt_generation(restaurant_paragraph):

    base_system_msg = "You are a 'Data Transformation Engine'. Your purpose is to convert unstructured textual data into structured JSON files."

    base_user_prompt = f"""
    Task:
    Identify, within the given 'Restaurant description' below, the name, location, restaurant type, food style, rating, price range, signature dishes, specialties and shortcomings, of the restaurant.
    Use this information to generate an output JSON file, with the structure shown below. For the 'price range' field, convert the dollar signs (e.g., $$) into an integer representing the number of dollar symbols.
    Hide your reasoning and thought process, and ommit any messages or explanations: only output the final JSON file.

    Restaurant description:
    {restaurant_paragraph}

    Structure of the output JSON file:
    {JSON_FORMAT}
    """

    return (base_system_msg, base_user_prompt)


def JSON_auto_repair_prompts(candidate_json_output, error_message):
    
    auto_repair_system_msg = "Your purpose is to convert defective JSON candidates into proper JSON files."

    auto_repair_prompt = f"""
    Task:
    Identify, within the given 'Candidate JSON', the name, location, restaurant type, food style, rating, price range, signature dishes, specialties and shortcomings, of the restaurant.
    Use this information to generate an proper JSON file, with the structure shown below.
    Hide your reasoning and thought process, and ommit any messages or explanations: only output the final JSON file.

    Candidate JSON:
    {candidate_json_output}

    The following error message can help you identify the issues in the Candidate JSON:
    {error_message}

    Structure of the output JSON file:
    {JSON_FORMAT}
    """

    return (auto_repair_system_msg, auto_repair_prompt)



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
        return (final_response)
    else:
        return (f"Request failed with status code: {response.status_code}")


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

    return (unwrap_response(response))



# ==========================================================================================================
# GUI menu
# ==========================================================================================================

def new_data_entry_process(paragraph, itemId):
    base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph=paragraph)
    candidate_json_output = llm_model(system_msg=base_system_msg, user_prompt=base_user_prompt)

    # Validation and Auto Correction loop 
    error_flag = True
    while error_flag:
        try:
            restaurant_data = Restaurant.model_validate_json(candidate_json_output)
            print(f"Success! Validated: {restaurant_data.name}")
            error_flag = False
            restaurant_data_json = json.loads(candidate_json_output)
        except ValidationError as error_message:
            auto_repair_system_msg, auto_repair_prompt = JSON_auto_repair_prompts(candidate_json_output, error_message)
            candidate_json_output = llm_model(system_msg=auto_repair_system_msg, user_prompt=auto_repair_prompt)
            print(f"Validation failed: {error_message.json()}")

    restaurant_data_json['id'] = itemId

    return (restaurant_data_json)


def manage_restaurants(file_path, backup_path):
    data = load_data(file_path)
    while True:
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")
        
        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---") 
			# Iterate through the records and show their names. If name doesn't exist, print 'N/A'.
            for i in range(len(data)):
                name = data[i]['name']

                if name:
                    print(name)
                else:
                    print('N/A')

        elif choice == '2':
			# Get the record index in demand from the user with input(). Check 
			# the validity of the input index. If the index is valid, use the  
			# helper function show_restaurant_card(res, index); Otherwise, 
			# print "invalid index."

            index = input("\nIndex: ")

            if 0 < index and index < len(data):
                show_restaurant_card(res, index)
            else:
                print("invalid index.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3': # ADD NEW DATA
                itemId = 1000000 + len(data) + 1 #the item id for the new data

				# First: ask the user to input a new restaurant description.
				# Second: use new_data_entry_process() to process the new paragraph.
				# Third: append the new restaurant data to the original data list.
				# Finally: save using save_data().
                new_restaurant_description = input("\nRestaurant Description: ")
                new_restaurant_json = new_data_entry_process(new_restaurant_description, itemId)

                data.append(new_restaurant_json)

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)

                print("✅ Restaurant added.")

            elif choice == '4': # EDIT DATA
				# First: ask for the input record index.
				# Second: iterate over the keys of the current record, and ask for
				#         new values. If the user doesn't want to update, a simple 
				#         Enter can skip. Update it only when the input index is 
				#         valid.
				# Third: save with save_data() and notify ("✅ Record updated.")

                input_index = input("\nIndex: ")
                index = int(input_index) if input_index.isdigit() else -1

                if 0 <= index and index < len(data):
                    json_scheme = data[index]

                    for key in json_scheme.keys():
                        new_value = input(f"\nNew {key}: ")

                        if new_value:
                            data[index][key] = new_value
                else:
                    print("invalid index.")

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)

                print("✅ Record updated.")

            elif choice == '5': # DELETE DATA
				# First: ask for the input record index.
				# Second: use pop() to delete if the index is valid.
				# Third: save_data() and notify.
                index = input("\nIndex: ")

                if 0 < index and index < len(data):
                    data.pop(index)
                else:
                    print("invalid index.")

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)

                print("✅ Record deleted.")

        elif choice == '6': # EXIT
            break
        else:
            print("Invalid input.")



if __name__ == "__main__":
	manage_restaurants(FILEPATH, BACKUP_PATH) # Actual UI Call
