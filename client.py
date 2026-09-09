import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class AIClient:
    """Client for Groq generation."""

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set."
            )

        self.groq = Groq(api_key=api_key)

        # Current Groq production model.
        self.gen_model = "openai/gpt-oss-20b"

    def generate(self, prompt):
        response = self.groq.chat.completions.create(
            model=self.gen_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content

    def is_available(self):
        return bool(os.environ.get("GROQ_API_KEY"))
