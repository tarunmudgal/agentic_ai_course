import os
import pathlib
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import types


OUTPUT_DIR = pathlib.Path("resources/study_notes")


def generate_notes(topic: str) -> str:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    client = genai.Client(api_key=key)
    prompt = (
        f"Create structured study notes for '{topic}'. Include: definition, analogy, complexity, "
        "common mistakes, and 3 practice questions."
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(temperature=0.3),
        contents=prompt,
    )
    return response.text


def save_notes(topic: str, notes: str) -> pathlib.Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = topic.lower().replace(" ", "_")
    path = OUTPUT_DIR / f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path.write_text(notes, encoding="utf-8")
    return path


def main() -> None:
    topic = "Binary Search Tree"
    notes = generate_notes(topic)
    path = save_notes(topic, notes)
    print(f"Saved notes: {path}")


if __name__ == "__main__":
    main()

