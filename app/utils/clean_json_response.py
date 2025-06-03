import re
import json

def clean_json_response(text: str) -> dict:
    # Remove Markdown-style code fences
    text = text.strip()
    if text.startswith("```json"):
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    elif text.startswith("```"):
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    
    return json.loads(text)