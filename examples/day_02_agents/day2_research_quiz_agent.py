import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai


@dataclass
class StudentProfile:
    name: str
    topics: list[str]
    weak_areas: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


def build_prompt(topic: str) -> str:
    return (
        f"Create a short study guide and 3 MCQs for '{topic}'. "
        "Return JSON with keys: guide, quiz (array with question/options/correct)."
    )


def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    client = genai.Client(api_key=key)
    profile = StudentProfile(name="Alex", topics=["Binary Search Tree"])

    topic = profile.topics[0]
    response = client.models.generate_content(model="gemini-2.0-flash", contents=build_prompt(topic))
    print(response.text)


if __name__ == "__main__":
    main()

