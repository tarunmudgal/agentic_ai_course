import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


SYSTEM = (
    "You are StudyBuddy. Be concise, accurate, and student-friendly. "
    "Always end with one short self-check question."
)


def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.3),
        contents="Teach me stack vs queue with one real-life analogy.",
    )
    print(response.text)


if __name__ == "__main__":
    main()

