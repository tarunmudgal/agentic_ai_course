import os

from dotenv import load_dotenv
from google import genai


PROMPTS = {
    "weak": "Explain dynamic programming.",
    "strong": (
        "Role: You are a CS tutor. Task: Explain dynamic programming for a second-year CSE student. "
        "Format: definition + analogy + 3-step approach + one interview tip. Limit to 120 words."
    ),
}


def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    client = genai.Client(api_key=key)
    for label, prompt in PROMPTS.items():
        print(f"\n--- {label.upper()} PROMPT ---")
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        print(response.text)


if __name__ == "__main__":
    main()

