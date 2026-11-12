


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

### Your Code Here:
structured_restaurant_lists = []
for i, restaurant_paragraph in enumerate(restaurant_list):
    ### 2.1: Produce your initial output

    #### 2.1.1. Get prompt (context + instructions + input + output format)
    base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph=restaurant_paragraph)

    candidate_json_output = llm_model(system_msg=base_system_msg, prompt_txt=base_user_prompt)

    ### 2.2: Validation and Auto Correction loop on the output 
    error_flag = True
    while error_flag:
        try:
            restaurant_data = Restaurant.model_validate_json(candidate_json_output)
            print(f"Success! Validated: {restaurant_data.name}")
            error_flag = False
        except ValidationError as error_message:
            auto_repair_system_msg, auto_repair_prompt = JSON_auto_repair_prompts(candidate_json_output, error_message)
            candidate_json_output = llm_model(system_msg=auto_repair_system_msg, prompt_txt=auto_repair_prompt)
            print(f"Validation failed: {error_message.json()}")

    ### 2.3: Append your finalized response to the structured_restaurant_lists
    structured_restaurant_lists.append(candidate_json_output)

    # A manual progress bar
    if (i+1)%20 == 0:
        print(f'{i+1} out of {len(restaurant_list)} is done')

# A final message to notify the completion
print('ALL DONE!!')