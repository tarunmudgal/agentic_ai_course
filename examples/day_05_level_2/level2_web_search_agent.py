import os

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from google import genai


def web_search(query: str, max_results: int = 3) -> str:
    with DDGS() as ddgs:
        items = list(ddgs.text(query, max_results=max_results))
    if not items:
        return "No results found"
    return "\n\n".join(
        f"{i+1}. {x.get('title')}\n{x.get('href')}\n{x.get('body')}" for i, x in enumerate(items)
    )


def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    context = web_search("best binary search tree tutorial for beginners")
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Using these search results:\n{context}\n\nGive a 1-week study plan.",
    )
    print(response.text)


if __name__ == "__main__":
    main()

