import os
import openai
from dotenv import load_dotenv

class GPTQueryExpander:
    def __init__(self, model_name="gpt-4.1"):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model_name