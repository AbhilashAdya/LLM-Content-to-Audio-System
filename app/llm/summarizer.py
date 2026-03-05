import os
from dotenv import load_dotenv
from google import genai


class GeminiClient:
    def __init__(self):
        load_dotenv()  # loads .env file

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")

        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
        )

        return response.text.strip()
