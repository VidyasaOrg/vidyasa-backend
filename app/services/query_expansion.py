import json
import os
import google.generativeai as genai
from app.utils.clean_json_response import clean_json_response

from dotenv import load_dotenv
load_dotenv()

# Initialize Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=api_key) # type: ignore
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

def expand_query_kb(original_query: str, relevant_documents: list[str], expansion_terms_count) -> dict:
    """
    Use Gemini to perform query expansion based on relevant documents.
    
    Args:
        original_query (str): The original search query.
        relevant_documents (List[str]): A list of relevant document texts.
        
    Returns:
        dict: Contains the expanded query as {"expanded-query": "..."}
    """
    if expansion_terms_count == "all":
        max_terms = ""
    else:
        max_terms = f"- Make sure to only add {expansion_terms_count} terms to the original query, no more no less\n"

    prompt = f"""
Use your knowledge to perform query expansion on the following sentence so that document retrieval becomes more effective and relevant.

⚠️ IMPORTANT:
- Focus on expanding the original query by adding relevant context or details from the documents, not by completely replacing the original query.
- If any part of the query is less relevant or not found in the documents, only then may it be replaced with something more accurate.
- Provide only 1 best expanded query.
- DO NOT use boolean operators like AND, OR, or double quotation marks (" ").
- Do not provide a list of synonyms or alternatives in one string.
- Avoid formats like: "<option 1>" OR "<option 2>".
- Generate the query as a natural, explicit, and specific sentence, as if the user knows exactly what they are looking for.
{max_terms}
🔍 Goal:
Make the query more focused, contextual, and aligned with the contents of relevant documents, without losing the original intent.

Original query:
{original_query}

Provide the result in JSON format:
{{
  "expanded-query": "<expanded query>"
}}

Example:
Original query: "BPJS regulations"
Document: mentions "BPJS Health claim regulations 2022"
Additional terms: 3
Expanded result:
{{
  "expanded-query": "BPJS Health claim regulations 2022"
}}

Original query: "BPJS regulations"
Document: mentions "BPJS Health claim regulations 2022"
Additional terms: 2
Expanded result:
{{
  "expanded-query": "BPJS Health claim regulations"
}}

Original query: "BPJS regulations"
Document: mentions "BPJS Health claim regulations 2022"
Additional terms: **not mentioned**
Expanded result:
{{
  "expanded-query": "BPJS Health claim regulations in 2022"
}}
"""

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        parsed = clean_json_response(content)
        return parsed
    except Exception as e:
        raise RuntimeError(f"Failed to expand query: {e}")
