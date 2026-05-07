"""
DAY 2 — StudyBuddy v2.0: Research & Quiz Agent
-----------------------------------------------
What's new compared to Day 1?
  + The agent has TOOLS: it can search a knowledge base and generate quizzes.
  + It has SESSION MEMORY: it remembers what you told it earlier.
  + It ADAPTS: it tracks your weak areas and focuses the quiz there.

Still limited: every step is triggered by YOU. The agent waits for your command.
That's what we fix on Day 3 with Agentic AI.

Run this file:
  python examples/day_02_agents/day2_research_quiz_agent.py
"""

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ── Shared Student Profile ────────────────────────────────────────────────────
# Think of this as a shared whiteboard all agents can read and write to.
@dataclass
class StudentProfile:
    name: str
    topics: list[str]
    weak_areas: list[str] = field(default_factory=list)
    quiz_history: list[dict] = field(default_factory=list)

    def record_result(self, topic: str, correct: bool) -> None:
        self.quiz_history.append({"topic": topic, "correct": correct})
        if not correct and topic not in self.weak_areas:
            self.weak_areas.append(topic)


# ── Helper: single Gemini call ────────────────────────────────────────────────
def _call_gemini(client: genai.Client, system: str, user: str, as_json: bool = False) -> str:
    kwargs: dict = {"system_instruction": system, "temperature": 0.3}
    if as_json:
        kwargs["response_mime_type"] = "application/json"
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(**kwargs),
        contents=user,
    )
    return response.text


# ── AGENT 1: Researcher ────────────────────────────────────────────────────────
# Responsibility: explain a topic clearly for a CS student.
def research_topic(client: genai.Client, topic: str) -> str:
    print(f"\n🔬 [ResearchAgent] Looking up '{topic}'...")
    system = (
        "You are a CS educator. Explain topics clearly to college freshmen. "
        "Include: 1-line definition, real-life analogy, key operations, time complexity."
    )
    notes = _call_gemini(client, system, f"Explain: {topic} (max 150 words)")
    print("  ✅ Research complete.")
    return notes


# ── AGENT 2: Quiz Generator ────────────────────────────────────────────────────
# Responsibility: generate MCQ questions focused on weak areas.
def generate_quiz(client: genai.Client, profile: StudentProfile, num: int = 3) -> list[dict]:
    focus = profile.weak_areas or profile.topics
    print(f"\n❓ [QuizAgent] Generating {num} questions (focus: {focus[0]})...")
    system = "You are a CS quiz maker. Return a JSON array of MCQ objects."
    prompt = (
        f"Generate {num} MCQ questions on: {', '.join(focus[:2])}.\n"
        f"JSON format: "
        f'[{{"id":1,"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},'
        f'"correct":"A","explanation":"..."}}]'
    )
    raw = _call_gemini(client, system, prompt, as_json=True)
    try:
        questions = json.loads(raw)
        if isinstance(questions, dict):          # handle {"questions": [...]} wrapper
            questions = questions.get("questions", [])
    except json.JSONDecodeError:
        questions = []
    print(f"  ✅ {len(questions)} question(s) ready.")
    return questions


# ── AGENT 3: Evaluator ────────────────────────────────────────────────────────
# Responsibility: grade answers and give personalised feedback.
def evaluate_answers(client: genai.Client, profile: StudentProfile,
                     questions: list[dict], answers: dict) -> str:
    print("\n📊 [EvaluatorAgent] Grading your answers...")
    correct_count = 0
    lines = []
    for q in questions:
        qid = str(q.get("id", ""))
        student = answers.get(qid, "").upper()
        right = q.get("correct", "").upper()
        is_ok = student == right
        if is_ok:
            correct_count += 1
        profile.record_result(q.get("question", "unknown"), is_ok)
        mark = "✅" if is_ok else "❌"
        lines.append(
            f"  Q{qid}: {mark} (your: {student or '—'}, correct: {right})\n"
            f"       {q.get('explanation', '')}"
        )
    score = f"{correct_count}/{len(questions)}"
    feedback_lines = "\n".join(lines)
    summary = (
        f"\n🎯 Score: {score}\n"
        f"{feedback_lines}\n"
        + (f"\n⚠️  Weak areas to review: {', '.join(profile.weak_areas)}" if profile.weak_areas else "\n🎉 Great job!")
    )
    print(summary)
    return summary


# ── Interactive Study Session ─────────────────────────────────────────────────
def run_study_session(client: genai.Client, profile: StudentProfile) -> None:
    print(f"\n{'='*60}")
    print(f"  📚 StudyBuddy v2.0 — Research & Quiz Agent")
    print(f"  Student: {profile.name} | Topics: {', '.join(profile.topics)}")
    print(f"{'='*60}")
    print("\n  NOTE: YOU drive each step. The agent responds to your commands.")
    print("  (This limitation disappears in Day 3's Agentic AI!)\n")

    # Step 1: Research the first topic
    topic = profile.topics[0]
    notes = research_topic(client, topic)
    print(f"\n📖 Study Notes:\n{notes}")

    # Step 2: Quiz the student
    questions = generate_quiz(client, profile, num=3)
    if not questions:
        print("Quiz generation failed — try again.")
        return

    print("\n" + "─" * 50)
    print(f"  Time for your quiz on '{topic}'! Answer A / B / C / D.")
    print("─" * 50)
    answers: dict = {}
    for q in questions:
        qid = str(q.get("id", ""))
        print(f"\nQ{qid}: {q.get('question', '')}")
        for opt, text in q.get("options", {}).items():
            print(f"  {opt}) {text}")

        raw_answer = input("Your answer: ").strip().upper()
        # Accept only valid choices
        answers[qid] = raw_answer if raw_answer in ("A", "B", "C", "D") else "—"

    # Step 3: Evaluate
    evaluate_answers(client, profile, questions, answers)

    # Show Day 2 limitation
    print("\n" + "─" * 60)
    print("🔍 Day 2 Limitation: I waited for YOU to ask each step.")
    print("   Real tutors proactively plan, re-search, and follow up.")
    print("   Day 3 (Agentic AI) will fix this — the system runs autonomously!")


# ── Entry Point ───────────────────────────────────────────────────────────────
def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in your .env file before running.")

    client = genai.Client(api_key=key)
    profile = StudentProfile(name="Alex", topics=["Binary Search Tree", "Dynamic Programming"])
    run_study_session(client, profile)


if __name__ == "__main__":
    main()
