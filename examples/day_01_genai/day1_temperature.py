import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


QUESTION = "Write a short explanation of recursion for beginners."


def generate(client: genai.Client, temperature: float) -> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(temperature=temperature),
        contents=QUESTION,
    )
    return response.text


def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    client = genai.Client(api_key=key)
    for temp in (0.0, 0.7, 1.0):
        print(f"\n=== temperature={temp} ===")
        print(generate(client, temp))


if __name__ == "__main__":
    main()

