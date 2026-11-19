# ==========================================================================================================
# Imports
# ==========================================================================================================

import asyncio
import json
import sys
import requests
import ast
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
from MCPClient import MCPClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (Root, TextContent, CreateMessageResult, CreateMessageRequestParams)
from typing import Any


agent_configs = {
    "user_profile_generator": {
        "role": "User Profile Generator",
        "goal": "Analyze user restaurant visit history and social media posts to create a comprehensive profile.",
        "backstory": """You are an expert user behavior analyst with 10 years of experience in the food industry. 
        You excel at identifying patterns in dining behavior and building rich user profiles."""
    },
    "rag_retriever": {
        "role": "RAG Retriever",
        "goal": "Query multimodal vector databases to retrieve relevant restaurants and recipes.",
        "backstory": """You are a data retrieval specialist with expertise in vector databases and semantic search."""
    },
    "food_trend_analyst": {
        "role": "Food Trend Analyst",
        "goal": "Identify current food trends and emerging dining concepts.",
        "backstory": """You are a culinary journalist who has spent 15 years covering food culture across global markets."""
    },
    "food_style_expert": {
        "role": "Food Style Expert",
        "goal": "Analyze cuisine types and flavor profiles to match user preferences.",
        "backstory": """You are a trained chef and culinary anthropologist with expertise in global cuisines."""
    },
    "nutrition_expert": {
        "role": "Nutrition Expert",
        "goal": "Evaluate nutritional content and ensure dietary compliance.",
        "backstory": """You are a registered dietitian with 8 years of clinical experience."""
    },
    "recommendation_expert": {
        "role": "Recommendation Expert",
        "goal": "Synthesize insights from all agents into final recommendations.",
        "backstory": """You are a recommendation systems architect with experience in personalization engines."""
    }
}



# ==========================================================================================================
# Helper functions
# ==========================================================================================================

def extract_raw_text(source: Any) -> str:
  """Extract the inner text payload from a CallToolResult object or a stringified representation."""
  # 1. Native CallToolResult (or similar object with .content)
  if hasattr(source, "content") and source.content:
    first_item = source.content[0]
    if hasattr(first_item, "text") and isinstance(first_item.text, str):
      return first_item.text
    if isinstance(first_item, dict) and isinstance(first_item.get("text"), str):
      return first_item["text"]

  # 2. Fallback if already a string
  if isinstance(source, (str, bytes)):
    text_val = (
        source.decode("utf-8", errors="replace")
        if isinstance(source, bytes)
        else source
    )
    match = re.search(
        r"text=(?:'((?:\\'|[^'])*)'|\"((?:\\\"|[^\"])*)\")", text_val
    )
    if match:
      raw_literal = (
          match.group(1) if match.group(1) is not None else match.group(2)
      )
      quote_char = "'" if match.group(1) is not None else '"'
      try:
        return ast.literal_eval(f"{quote_char}{raw_literal}{quote_char}")
      except SyntaxError:
        return raw_literal
    return text_val

  raise TypeError(
      f"Expected CallToolResult, str, or bytes, got {type(source).__name__}"
  )


def parse_text_content_lists(
    source: Any,
) -> tuple[list[str], list[str], list[dict[str, Any]], list[float]]:
  """Extract [ids, descriptions, metadatas, scores] from a CallToolResult or stringified output."""
  inner_json_str = extract_raw_text(source)
  parsed = json.loads(inner_json_str)

  if not isinstance(parsed, list) or len(parsed) != 4:
    raise ValueError(
        f"Expected inner JSON to be a 4-element list, got {type(parsed).__name__}"
        " of length"
        f" {len(parsed) if isinstance(parsed, list) else 'N/A'}."
    )

  ids, descriptions, metadatas, scores = parsed
  return ids, descriptions, metadatas, scores



# ==========================================================================================================
# Client declaration
# ==========================================================================================================

# Configuration
SERVER_SCRIPT = str("server.py")

# StdioServerParameters launches "python server.py" via stdin/stdout
server_params = StdioServerParameters(
    command="python",
    args=[SERVER_SCRIPT],
)



# ==========================================================================================================
# Helper functions
# ==========================================================================================================

def _unwrap(res: dict):
    """Chroma returns lists-of-lists; unwrap the first query."""
    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists


def unwrap_response(response):
    """
    Function to unwrap the response from the Ollama server and extract the content from the streamed JSON lines.

    Args:
        response (requests.Response): The response object from the Ollama server.
    Returns:
        str: The final response content extracted from the streamed JSON lines.
    """

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


# Textual LLM
def call_agent(agent_key: str, user_prompt: str) -> str:
    """
        Function to send a model execution request to the Ollama server running locally through port 11434,
        and receive and stream the response from the model.
    
        Args:
            system_msg (str): The system message to provide context for the LLM.
            user_prompt (str): The user prompt containing the task and input data.
    
        Returns:
            str: The final response from the LLM after processing the input.
    """

    config = agent_configs[agent_key]
    
    system_prompt = f"""You are a {config['role']}. Your goal: {config['goal']} Your background: {config['backstory']} Respond with structured, actionable output."""

    url = "http://127.0.0.1:11434/api/chat"

    payload = {
        "model": "phi4-mini:3.8b",
        "messages": [ 
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ]
    }

    response = requests.post(url, json=payload, stream=True)

    return unwrap_response(response)



# ==========================================================================================================
# Multi-Agentic functions
# ==========================================================================================================

def node_generate_profile(state: dict) -> dict:
    """Generate user profile from input data."""
    print("\n[Phase 1] Generating user profile...")
    
    user_message = f"""Analyze this user data and create a comprehensive profile:
    {state['user_input']}
    Provide output in JSON format with these keys:
        - favorite_cuisines (list)
        - dietary_restrictions (list)
        - dining_occasions (list)
        - price_range (string)
        - adventurousness_score (1-10)
        - flavor_preferences (list)
        - summary (string)
    Hide your reasoning and thought process. Ommit any messages or explanations.
    Output only the final JSON file, without quotation marks, as a plain string.
    """

    try:
        response = call_agent("user_profile_generator", user_message)
        user_profile = json.loads(response)
        print(f"User profile generated: {user_profile.get('summary', 'No summary')}")
    except Exception as e:
        print(f"Error generating profile: {e}")
        user_profile = {"error": str(e)}

    state["user_profile"] = user_profile
    state["workflow_step"] = "profile_generated"
    return (state)


async def node_retrieve_candidates(client, state: dict) -> dict:
    """Retrieve restaurant and recipe candidates from vector database."""
    print("\n[Phase 2] Retrieving candidates from vector database...")

    profile = state["user_profile"]
    
    user_message = f"""Based on this user profile:
    {json.dumps(profile, indent=2)}
    Create a query to retrieve relevant restaurants and recipes from a multimodal vector database.
    """

    try:
        prompt = call_agent("rag_retriever", user_message)
        print(f"PROMPT: {prompt}")
        tuple_res = await client.call_tool("retrieve_articles", {"chromadb_dir": "chroma_multimodal", "query": prompt})
        ids, docs, metas, dists = parse_text_content_lists(tuple_res)
        print(f"IDS RETRIEVED: {ids}\n")
        print(f"DOCS RETRIEVED: {docs}\n")
        print(f"META RETRIEVED: {metas}\n")
        restaurants = docs
        recipes = docs
    except Exception as e:
        print(f"Error retrieving candidates: {e}")
        restaurants, recipes = [], []

    state["retrieved_restaurants"] = restaurants
    state["retrieved_recipes"] = recipes
    state["workflow_step"] = "candidates_retrieved"
    return (state)


def node_analyze_trends(state: dict) -> dict:
    """Analyze food trends in the retrieved candidates."""
    print("\n[Phase 3a] Analyzing food trends...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    
    user_message = f"""Analyze current food trends in these options:
    
    Restaurants: {json.dumps(restaurants[:5], indent=2)}
    Recipes: {json.dumps(recipes[:5], indent=2)}
    
    Identify 3-5 relevant trends and explain how they align with modern dining culture.
    
    Return JSON: {{"trends": [{{"name": str, "description": str, "relevance": str}}]}}
    """
    
    try:
        response = call_agent("food_trend_analyst", user_message)
        trend_analysis = json.loads(response)
        print(f"Identified {len(trend_analysis.get('trends', []))} trends")
    except Exception as e:
        print(f"Error analyzing trends: {e}")
        trend_analysis = {"error": str(e)}
    
    state["trend_analysis"] = trend_analysis
    return (state)


def node_analyze_styles(state: dict) -> dict:
    """Analyze food styles and flavor profiles."""
    print("\n[Phase 3b] Analyzing food styles...")

    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]

    user_message = f"""
    Understand how the 'cuisine type' of each of the following restaurants match the user's profile given below:
    {restaurants}.
    Likewise, understand how the cuisine type, cooking methods, and flavor profiles of each of the following recipes match the user's profile:
    {recipes}.

    User's profile:
    {profile}.
    
    Return a list of food styles and dining experiences that match the user preferences.
    """
    
    try:
        response = call_agent("food_style_expert", user_message)
        style_analysis = json.loads(response)
        print(f"Style analysis completed")
    except Exception as e:
        print(f"Error analyzing styles: {e}")
        style_analysis = {"error": str(e)}
    
    state["style_analysis"] = style_analysis
    return (state)


def node_evaluate_nutrition(state: dict) -> dict:
    """Evaluate nutritional aspects and dietary compliance."""
    print("\n[Phase 3c] Evaluating nutrition...")

    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]
    
    user_message = f"""Evaluate the nutritional fit of these options:
    
    User Profile: {json.dumps(profile, indent=2)}
    Restaurants: {json.dumps(restaurants[:5], indent=2)}
    Recipes: {json.dumps(recipes[:5], indent=2)}
    
    Check dietary restrictions, allergens, and nutritional balance.
    
    Return JSON: {{"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}}
    """
    
    try:
        response = call_agent("nutrition_expert", user_message)
        nutrition_analysis = json.loads(response)
        print(f"Nutrition evaluation completed")
    except Exception as e:
        print(f"Error evaluating nutrition: {e}")
        nutrition_analysis = {"error": str(e)}
    
    state["nutrition_analysis"] = nutrition_analysis
    return (state)


def node_generate_recommendations(state: dict) -> dict:
    """Synthesize all analyses into final recommendations."""
    print("\n[Phase 4] Generating final recommendations...")
    
    user_message = f"""Synthesize these insights into top 5 restaurant and top 5 recipe recommendations:
    User Profile: {json.dumps(state['user_profile'], indent=2)}
    Restaurants: {json.dumps(state['retrieved_restaurants'][:10], indent=2)}
    Recipes: {json.dumps(state['retrieved_recipes'][:10], indent=2)}
    Trends: {json.dumps(state['trend_analysis'], indent=2)}
    Styles: {json.dumps(state['style_analysis'], indent=2)}
    Nutrition: {json.dumps(state['nutrition_analysis'], indent=2)}
    
    Return JSON:
    {{
        "restaurants": [{{"name": str, "reasoning": str}}],
        "recipes": [{{"name": str, "reasoning": str}}]
    }}
    Each reasoning should be 2-3 sentences explaining why it's a great match.
    """
    
    try:
        response = call_agent("recommendation_expert", user_message)
        recommendations = json.loads(response)
        print(f"Generated {len(recommendations.get('restaurants', []))} restaurant recommendations")
        print(f"Generated {len(recommendations.get('recipes', []))} recipe recommendations")
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        recommendations = {"error": str(e)}
    
    state["final_recommendations"] = recommendations
    state["workflow_step"] = "complete"
    return (state)



# ==========================================================================================================
# Beginning of execution
# ==========================================================================================================

async def run_workflow(client, user_input: str) -> dict:
    """Run the full multi-agent workflow.
    
    Phases:
      1. User Analysis      (sequential)
      2. Data Retrieval      (sequential)
      3. Analysis            (parallel – trends, styles, nutrition)
      4. Synthesis           (sequential)
    """

    # Initialize shared state
    state = {
        # Input
        "user_input": "",
        # Phase 1: User Analysis
        "user_profile": {},
        # Phase 2: Data Retrieval
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        # Phase 3: Analysis (Parallel)
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        # Phase 4: Synthesis
        "final_recommendations": {},
        # Metadata
        "workflow_step": "start"
    }
    
    # Phase 1 – Sequential
    state = node_generate_profile(state)
    
    # Phase 2 – Sequential
    state = await node_retrieve_candidates(client, state)

    # Phase 3 – Parallel using ThreadPoolExecutor
    print("\n[Phase 3] Running analysis agents in parallel...")
    
    # Each function needs its own copy of state to read from,
    # and we merge their outputs back afterwards.
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_trends   = executor.submit(node_analyze_trends, dict(state))
        future_styles   = executor.submit(node_analyze_styles, dict(state))
        future_nutrition = executor.submit(node_evaluate_nutrition, dict(state))
        
        result_trends   = future_trends.result()
        result_styles   = future_styles.result()
        result_nutrition = future_nutrition.result()
        
    # Merge parallel results back into state
    state["trend_analysis"]    = result_trends["trend_analysis"]
    state["style_analysis"]    = result_styles["style_analysis"]
    state["nutrition_analysis"] = result_nutrition["nutrition_analysis"]
    
    # Phase 4 – Sequential
    state = node_generate_recommendations(state)
    
    return (state)


async def main():
    test_user_1 = """
    Restaurant Visit History:
    - Visited "Green Bowl" (Vegan, $$) 8 times
    - Visited "Mediterranean Grill" (Mediterranean, $$) 5 times
    - Visited "Juice Lab" (Smoothies, $) 3 times

    Social Media Posts:
    - "Loving my plant-based journey! 🌱"
    - "This gluten-free Mediterranean bowl is amazing!"
    - "Fresh juice is the best way to start the day."

    Dietary Restrictions: Vegan, Gluten-Free
    """

    if len(sys.argv) < 2:
        print("Usage: python recommendation_client.py <server_script>")
        sys.exit(1)

    client = MCPClient()

    try:
        await client.connect(sys.argv[1])
        await run_workflow(client, test_user_1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        await client.cleanup()



# ==========================================================================================================
# Entry point
# ==========================================================================================================

if __name__ == "__main__":
    asyncio.run(main())