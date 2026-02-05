import os
from openai import OpenAI
from typing import Optional

class LLMClient:
    """
    OpenAI API client for actuarial Q&A.
    Uses GPT-4o-mini for fast, cost-effective responses.
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.last_error = None
        self.client = None

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def is_available(self) -> bool:
        """Check if OpenAI API is configured."""
        if not self.api_key:
            self.last_error = "OPENAI_API_KEY not found in environment"
            return False
        if not self.client:
            self.last_error = "OpenAI client not initialized"
            return False
        return True

    def get_completion(self, system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
        """
        Get a chat completion from OpenAI.
        Default model: gpt-4o-mini (fast and cheap)
        """
        if not self.is_available():
            return f"Error: {self.last_error}"

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0  # Deterministic responses
            )
            return response.choices[0].message.content
        except Exception as e:
            self.last_error = str(e)
            return f"Error calling OpenAI: {str(e)}"
