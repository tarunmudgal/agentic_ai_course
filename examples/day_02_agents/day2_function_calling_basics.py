import datetime
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


def get_current_date() -> str:
    return datetime.datetime.now().strftime("%A, %B %d, %Y")


def days_until_exam(exam_date: str) -> str:
    exam = datetime.datetime.strptime(exam_date, "%Y-%m-%d").date()
    days = (exam - datetime.date.today()).days
    return f"{days} day(s) until exam." if days >= 0 else f"Exam was {-days} day(s) ago."


TOOL_MAP = {"get_current_date": get_current_date, "days_until_exam": days_until_exam}

TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_current_date",
        description="Return today's date",
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    ),
    types.FunctionDeclaration(
        name="days_until_exam",
        description="Calculate remaining days to exam",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"exam_date": types.Schema(type=types.Type.STRING)},
            required=["exam_date"],
        ),
    ),
])


def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    client = genai.Client(api_key=key)
    history = [{"role": "user", "parts": [{"text": "My exam is 2026-06-20. How long do I have?"}]}]

    for _ in range(4):
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(tools=[TOOLS], temperature=0.2),
            contents=history,
        )
        candidate = response.candidates[0]
        tool_calls = [p for p in candidate.content.parts if getattr(p, "function_call", None)]
        if not tool_calls:
            print(response.text)
            return

        history.append({"role": "model", "parts": candidate.content.parts})
        tool_parts = []
        for call in tool_calls:
            name = call.function_call.name
            args = dict(call.function_call.args) if call.function_call.args else {}
            result = TOOL_MAP[name](**args)
            tool_parts.append(types.Part(function_response=types.FunctionResponse(name=name, response={"result": result})))
        history.append({"role": "user", "parts": tool_parts})


if __name__ == "__main__":
    main()

