import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None


class StudyPlanRequest(BaseModel):
    topic: str
    days: int = 5


app = FastAPI(title="Agentic AI Study Buddy API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/study-plan")
def create_study_plan(payload: StudyPlanRequest) -> dict:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY or google-genai package")

    client = genai.Client(api_key=api_key)
    prompt = (
        f"Create a concise {payload.days}-day plan for {payload.topic}. "
        "Include daily goal, 1 practice task, and 1 QA validation check."
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return {"topic": payload.topic, "days": payload.days, "plan": response.text}


if __name__ == "__main__":
    # Run with: uvicorn examples.day_05_level_3.level3_fastapi_service:app --reload
    print("Use uvicorn to start this API.")

