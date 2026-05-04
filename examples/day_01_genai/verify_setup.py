import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if key:
        print(f"OK: GEMINI_API_KEY found ({key[:6]}...{key[-4:]})")
    else:
        print("WARN: GEMINI_API_KEY not found in environment/.env")

    try:
        from google import genai  # noqa: F401
        print("OK: google-genai import works")
    except Exception as exc:
        print(f"WARN: google-genai import failed: {exc}")


if __name__ == "__main__":
    main()

