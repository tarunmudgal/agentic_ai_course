import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


SYSTEM = "You are StudyBuddy. Use conversation memory to answer follow-up questions correctly."


class StudyBuddyWithMemory:
    def __init__(self, client: genai.Client):
        self.client = client
        self.history: list[dict] = []

    def chat(self, message: str) -> str:
        self.history.append({"role": "user", "parts": [{"text": message}]})
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.4),
            contents=self.history,
        )
        text = response.text
        self.history.append({"role": "model", "parts": [{"text": text}]})
        return text


def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    buddy = StudyBuddyWithMemory(genai.Client(api_key=key))
    prompts = [
        "Hi, I am Alex. My exam is on 2026-06-20.",
        "I struggle with graph algorithms.",
        "What did I say I struggle with?",
    ]
    for p in prompts:
        print(f"\nUser: {p}")
        print(f"Bot: {buddy.chat(p)}")


if __name__ == "__main__":
    main()

