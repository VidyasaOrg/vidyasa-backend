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
- Expand the query by **adding** relevant context or terms **before or after** the original query.
- ❌ DO NOT modify, rephrase, or merge the original query in any way.
  - For example, if the original query is "music culture", you **must not** change it to "musical culture", "music and culture", or "music culturally".
  - ✅ You may expand like "traditional music culture in Indonesia" or "influences of pop music culture".
- The original query should appear **exactly as is** and remain **intact** in the expanded version.
- Provide only 1 best expanded query.
- DO NOT use boolean operators like AND, OR, or double quotation marks (" ").
- Do not provide a list of synonyms or alternatives in one string.
- Avoid formats like: "<option 1>" OR "<option 2>".
- Generate the query as a natural, explicit, and specific sentence, as if the user knows exactly what they are looking for.

🔍 Goal:
Make the query more focused, contextual, and aligned with the contents of relevant documents, without losing or altering the original phrasing.

Original query:
{original_query}

Provide the result in JSON format:
{{
  "expanded-query": "<expanded query>"
}}

Example:
Original query: "BPJS regulations"
Expanded result:
{{
  "expanded-query": "BPJS Health claim regulations 2022"
}}

Original query: "music culture"
Expanded result:
{{
  "expanded-query": "Music culture and heritage"
}}
"""

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        parsed = clean_json_response(content)
        return parsed
    except Exception as e:
        raise RuntimeError(f"Failed to expand query: {e}")

def expand_query_from_exp(original_query: str) -> str:
    """
    Use Gemini to perform query expansion based on the explanation of the original query.
    
    Args:
        original_query (str): The original search query.
        relevant_documents (List[str]): A list of relevant document texts.
        
    Returns:
        dict: Contains the expanded query as {"expanded-query": "..."}
    """
    
    prompt = f"""
Your task is to expand the user's search query into a more informative and contextual version.

Rules:
- Use the original query as a base to elaborate on the intended meaning of the search.
- Do not use boolean formats such as "OR".
- Return only **one** expanded query.
- The answer **MUST** be in JSON format, as shown in the example below:

Example:
Original query: "What is the capital of Malaysia?"
Output:
{{
  "expanded-query": "Kuala Lumpur is the capital of Malaysia. Its modern skyline is dominated by the 451m-tall Petronas Twin Towers, a pair of glass-and-steel-clad skyscrapers with Islamic motifs. The towers also offer a public skybridge and observation deck."
}}

Example:
Original query: "Sunspot"
Output:
{{
  "expanded-query": "Sunspots are temporary spots on the Sun's surface that are darker than the surrounding area. They are one of the most recognizable Solar phenomena and despite the fact that they are mostly visible in the solar photosphere they usually affect the entire solar atmosphere."
}}

Now, use that format for the following query:

Original query: "{original_query}"
"""

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        parsed = clean_json_response(content)
        return parsed
    except Exception as e:
        raise RuntimeError(f"Failed to expand query: {e}")