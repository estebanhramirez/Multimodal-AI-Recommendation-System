import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import requests
import uvicorn


# Set a custom secret API key so unauthorized people cannot use your laptop's GPU
load_dotenv()
API_KEY = os.getenv('API_KEY')


class PromptRequest(BaseModel):
    prompt: str



app = FastAPI()
@app.post("/generate")
def generate_response(data: PromptRequest, authorization: str = Header(None)):
    # Verify the API Key passed in the header
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Example: Send the prompt to a local Ollama instance running on port 11434
    try:
        ollama_res = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": "qwen3-vl:4b", "prompt": data.prompt, "stream": False},
            timeout=60
        )
        response_json = ollama_res.json()
        return {"response": response_json.get("response", "No response generated")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    # Run FastAPI on local port 8080
    uvicorn.run(app, host="127.0.0.1", port=8080)
