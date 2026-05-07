"""
DAY 3 — Orchestration Basics: Multi-Agent Coordination
------------------------------------------------------
The key concept today: an ORCHESTRATOR decides what to do and WHEN.
It calls specialised agents without being asked by the user.

Analogy:
  Think of a restaurant kitchen.
  - Head Chef (Orchestrator) reads the order and assigns tasks.
  - Sous Chef (ResearchAgent) preps the ingredients.
  - Pastry Chef (QuizAgent) makes the dessert.
  - Manager (EvaluatorAgent) checks quality before serving.
  Nobody calls the head chef to say "hey, assign the tasks now". It just happens.

Contrast with Day 2:
  Day 2: YOU typed "quiz me" → agent quizzed you.
  Day 3: Orchestrator automatically decides: plan → research → quiz → evaluate.

Run this file:
  python examples/day_03_agentic_ai/day3_orchestration_basics.py
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ── Shared State ──────────────────────────────────────────────────────────────
# All agents read from and write to this object.
# It's the "whiteboard" everyone can see.
@dataclass
class SessionState:
    student_name: str
    topic: str
    study_plan: str = ""
    research_notes: str = ""
    quiz_questions: list = field(default_factory=list)
    quiz_score: int = 0
    feedback: str = ""


# ── Gemini helper ─────────────────────────────────────────────────────────────
def _ask(client: genai.Client, instruction: str, prompt: str) -> str:
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction=instruction, temperature=0.3),
        contents=prompt,
    )
    return resp.text


# ── AGENT 1: Planner (creates study plan) ─────────────────────────────────────
def planner_agent(client: genai.Client, state: SessionState) -> None:
    """Step 1 — Create a 3-bullet study plan for the topic."""
    print(f"\n📅 [PlannerAgent] Creating study plan for '{state.topic}'...")
    state.study_plan = _ask(
        client,
        "You are a study planner. Create short, actionable plans.",
        f"Create a 3-bullet study plan to master '{state.topic}' in one week."
    )
    print("  ✅ Plan created.")


# ── AGENT 2: Researcher (generates content to study) ──────────────────────────
def researcher_agent(client: genai.Client, state: SessionState) -> None:
    """Step 2 — Generate concise study notes for the topic."""
    print(f"\n🔬 [ResearchAgent] Researching '{state.topic}'...")
    state.research_notes = _ask(
        client,
        "You are a CS educator. Be clear and use analogies.",
        f"Explain '{state.topic}' (definition, analogy, operations, complexity). Max 150 words."
    )
    print("  ✅ Notes ready.")


# ── AGENT 3: Quiz Master (generates and delivers quiz) ────────────────────────
def quiz_agent(client: genai.Client, state: SessionState, simulate: bool = True) -> None:
    """Step 3 — Generate 2 quick questions. In demo mode, simulate answers."""
    print(f"\n❓ [QuizAgent] Generating quiz for '{state.topic}'...")
    raw = _ask(
        client,
        "Generate exactly 2 True/False quiz questions.",
        f"Topic: {state.topic}. Return them as a numbered list with the answer on a new line."
    )
    print(f"  ✅ Quiz:\n{raw}")

    if simulate:
        # Demo: simulate a student who answers correctly
        state.quiz_questions = [{"question": raw, "student_answer": "simulated"}]
        state.quiz_score = 1  # 1 out of 2 simulated correct
        print("  [Simulated answers — set simulate=False for interactive mode]")
    else:
        # Real interactive mode
        answer = input("  Type your answers: ").strip()
        state.quiz_questions = [{"question": raw, "student_answer": answer}]


# ── AGENT 4: Evaluator (grades and gives feedback) ────────────────────────────
def evaluator_agent(client: genai.Client, state: SessionState) -> None:
    """Step 4 — Evaluate the quiz result and give encouraging feedback."""
    print(f"\n📊 [EvaluatorAgent] Generating feedback...")
    state.feedback = _ask(
        client,
        "You are an encouraging CS tutor. Give constructive feedback.",
        f"Student studied '{state.topic}'. Quiz score: {state.quiz_score}/2. "
        f"Give 2-sentence personalised feedback and one tip for improvement."
    )
    print(f"  ✅ Feedback:\n{state.feedback}")


# ── THE ORCHESTRATOR ──────────────────────────────────────────────────────────
# This is the key Day 3 concept:
# The orchestrator runs all agents IN SEQUENCE, WITHOUT the user asking.
def orchestrator(client: genai.Client, student_name: str, topic: str) -> SessionState:
    """
    Autonomous orchestration loop.

    The student provides the goal ONCE.
    The orchestrator decides what to do and when — no step-by-step human prompting.

    Flow: plan → research → quiz → evaluate
    """
    print(f"\n{'='*60}")
    print(f"  🚀 ORCHESTRATOR: Starting autonomous session")
    print(f"  Student: {student_name} | Topic: {topic}")
    print(f"{'='*60}")
    print("  [Watch how each agent fires WITHOUT you asking]")

    state = SessionState(student_name=student_name, topic=topic)

    # The orchestrator calls agents in order — autonomously
    planner_agent(client, state)       # Step 1: Plan
    researcher_agent(client, state)    # Step 2: Research
    quiz_agent(client, state)          # Step 3: Quiz (simulate=True for demo)
    evaluator_agent(client, state)     # Step 4: Evaluate

    # Summary
    print(f"\n{'='*60}")
    print(f"  ✅ Session Complete for {student_name}")
    print(f"{'='*60}")
    print(f"\n📋 Study Plan:\n{state.study_plan}")
    print(f"\n📖 Research Notes:\n{state.research_notes}")
    print(f"\n💬 Feedback:\n{state.feedback}")
    print("\n👉 Day 3 vs Day 2: You didn't type a single command after the goal!")
    print("   The orchestrator managed everything. That's AGENTIC AI.\n")

    return state


# ── Entry Point ───────────────────────────────────────────────────────────────
def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in your .env file before running.")

    client = genai.Client(api_key=key)
    orchestrator(client, student_name="Alex", topic="Binary Search Tree")


if __name__ == "__main__":
    main()
