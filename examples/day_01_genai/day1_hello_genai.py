import os

from dotenv import load_dotenv
from google import genai


def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Explain what a binary search tree is in 5 simple lines."
    )
    print(response.text)


if __name__ == "__main__":
    main()

