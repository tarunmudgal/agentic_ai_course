"""
DAY 3 — StudyBuddy v3.0: Autonomous Exam Coach
----------------------------------------------
This is the full Agentic AI system. Compare to Day 1 and Day 2:

  Day 1: Student asks → AI answers. Done. (One turn)
  Day 2: Student asks each step → AI responds. (Human drives)
  Day 3: Student says "prep me for my exam" ONCE → system runs for multiple days. (Autonomous!)

How it works:
  1. Student provides goal (exam date, topics, hours/day). Just once.
  2. PlannerAgent creates a multi-day study schedule.
  3. ResearchAgent fetches content for today's focus topic.
  4. QuizAgent generates targeted questions (focusing on weak areas!).
  5. EvaluatorAgent grades answers and updates the student profile.
  6. ReportAgent saves a daily progress report to disk.
  7. Orchestrator decides IF re-planning is needed (adapts autonomously!).

Analogy:
  Like having a personal tutor who reads your performance overnight,
  adjusts tomorrow's plan without you asking, and sends you a progress
  report every morning. YOU only set the goal once.

Run this file:
  python examples/day_03_agentic_ai/day3_autonomous_exam_coach.py
"""

import datetime
import json
import os
import pathlib
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import types


REPORTS_PATH = pathlib.Path("resources/study_reports")


# ── Shared Student Profile ─────────────────────────────────────────────────────
@dataclass
class StudentProfile:
    name: str
    exam_date: str
    topics: list[str]
    study_hours_per_day: int = 2
    # Filled in during sessions:
    study_plan: list = field(default_factory=list)
    topic_research: dict = field(default_factory=dict)
    quiz_history: list[dict] = field(default_factory=list)
    weak_areas: list[str] = field(default_factory=list)
    strong_areas: list[str] = field(default_factory=list)

    def days_until_exam(self) -> int:
        exam = datetime.datetime.strptime(self.exam_date, "%Y-%m-%d").date()
        return (exam - datetime.date.today()).days

    def overall_accuracy(self) -> float:
        if not self.quiz_history:
            return 0.0
        return sum(1 for r in self.quiz_history if r["correct"]) / len(self.quiz_history)

    def record_quiz_result(self, topic: str, correct: bool) -> None:
        self.quiz_history.append({"topic": topic, "correct": correct, "date": str(datetime.date.today())})
        # Update weak/strong lists
        topic_results = [r for r in self.quiz_history if r["topic"] == topic]
        acc = sum(1 for r in topic_results if r["correct"]) / len(topic_results)
        if acc < 0.6 and topic not in self.weak_areas:
            self.weak_areas.append(topic)
        elif acc >= 0.8 and topic not in self.strong_areas:
            self.strong_areas.append(topic)


# ── Gemini helper ─────────────────────────────────────────────────────────────
def _ask(client: genai.Client, system: str, user: str, as_json: bool = False) -> str:
    kwargs: dict = {"system_instruction": system, "temperature": 0.3}
    if as_json:
        kwargs["response_mime_type"] = "application/json"
    return client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(**kwargs),
        contents=user,
    ).text


# ── AGENT 1: Planner ──────────────────────────────────────────────────────────
class PlannerAgent:
    SYSTEM = (
        "You are a CS study planner. Create realistic, day-by-day schedules. "
        "Give weak areas extra time. The day before the exam = review only."
    )

    def create_or_update_plan(self, client: genai.Client, profile: StudentProfile) -> None:
        print("  📅 [PlannerAgent] Creating study schedule...")
        prompt = (
            f"Student: {profile.name} | Exam: {profile.exam_date} "
            f"({profile.days_until_exam()} days away)\n"
            f"Topics: {', '.join(profile.topics)}\n"
            f"Hours/day: {profile.study_hours_per_day}\n"
            f"Weak areas (needs extra time): {profile.weak_areas or 'None yet'}\n\n"
            "Return a JSON array: [{day: 1, topic: '...', goal: '...', hours: 2}]"
        )
        raw = _ask(client, self.SYSTEM, prompt, as_json=True)
        try:
            plan = json.loads(raw)
            profile.study_plan = plan if isinstance(plan, list) else []
        except json.JSONDecodeError:
            profile.study_plan = []
        print(f"  ✅ Plan: {len(profile.study_plan)} days scheduled.")


# ── AGENT 2: Researcher ───────────────────────────────────────────────────────
class ResearchAgent:
    SYSTEM = (
        "You are a CS educator. Create clear, beginner-friendly study guides. "
        "Include: definition, analogy, key steps, time complexity."
    )

    def research_topic(self, client: genai.Client, topic: str, profile: StudentProfile) -> str:
        if topic in profile.topic_research:
            print(f"  📚 [ResearchAgent] '{topic}' already cached. Skipping.")
            return profile.topic_research[topic]
        print(f"  🔬 [ResearchAgent] Researching '{topic}'...")
        notes = _ask(client, self.SYSTEM, f"Study guide for '{topic}' (max 200 words).")
        profile.topic_research[topic] = notes
        print("  ✅ Research cached.")
        return notes


# ── AGENT 3: Quiz Agent ────────────────────────────────────────────────────────
class QuizAgent:
    SYSTEM = "You are a CS quiz maker for college students. Create MCQ questions."

    def generate_quiz(self, client: genai.Client, profile: StudentProfile,
                      topics: list[str], num: int = 3) -> list[dict]:
        focus = (profile.weak_areas + topics)[:2]
        print(f"  ❓ [QuizAgent] Generating {num} questions (focus: {focus})...")
        prompt = (
            f"Generate {num} MCQ questions covering: {', '.join(focus)}.\n"
            f"JSON format: "
            f'[{{"id":1,"topic":"...","question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},'
            f'"correct":"A","explanation":"..."}}]'
        )
        raw = _ask(client, self.SYSTEM, prompt, as_json=True)
        try:
            qs = json.loads(raw)
            return qs if isinstance(qs, list) else qs.get("questions", [])
        except json.JSONDecodeError:
            return []


# ── AGENT 4: Evaluator ────────────────────────────────────────────────────────
class EvaluatorAgent:
    def evaluate(self, profile: StudentProfile, questions: list[dict], answers: dict) -> dict:
        print("  📊 [EvaluatorAgent] Grading...")
        correct = 0
        for q in questions:
            qid = str(q.get("id", ""))
            is_ok = answers.get(qid, "").upper() == q.get("correct", "").upper()
            correct += int(is_ok)
            profile.record_quiz_result(q.get("topic", "unknown"), is_ok)
        pct = int(correct / max(len(questions), 1) * 100)
        print(f"  ✅ Score: {correct}/{len(questions)} ({pct}%)")
        return {"score": f"{correct}/{len(questions)}", "percentage": pct}


# ── AGENT 5: Reporter ─────────────────────────────────────────────────────────
class ReportAgent:
    def save_report(self, profile: StudentProfile, eval_result: dict, day: int) -> str:
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        content = (
            f"# StudyBuddy Report — Day {day}\n"
            f"**Student:** {profile.name} | **Exam:** {profile.exam_date} "
            f"({profile.days_until_exam()} days left)\n\n"
            f"## Quiz Result\n- Score: {eval_result['score']} ({eval_result['percentage']}%)\n"
            f"- Overall accuracy: {profile.overall_accuracy():.0%}\n\n"
            f"## Strengths\n{chr(10).join(f'- {s}' for s in profile.strong_areas) or '- Still building!'}\n\n"
            f"## Needs Work\n{chr(10).join(f'- {w}' for w in profile.weak_areas) or '- All good!'}\n\n"
            f"*Generated by StudyBuddy v3.0*\n"
        )
        path = REPORTS_PATH / f"day_{day:02d}_{profile.name.lower()}.md"
        path.write_text(content, encoding="utf-8")
        print(f"  📝 [ReportAgent] Report saved: {path}")
        return str(path)


# ── THE ORCHESTRATOR ──────────────────────────────────────────────────────────
class AutonomousExamCoach:
    """
    StudyBuddy v3.0 — runs multiple study sessions AUTONOMOUSLY.
    The student gives a goal once; the system handles the rest.
    """

    def __init__(self) -> None:
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent()
        self.quiz = QuizAgent()
        self.evaluator = EvaluatorAgent()
        self.reporter = ReportAgent()
        print("🚀 AutonomousExamCoach (v3.0) initialised — 5 agents ready.")

    def run_session(self, client: genai.Client, profile: StudentProfile,
                    day: int, simulate: bool = True) -> dict:
        print(f"\n{'='*60}")
        print(f"  🤖 ORCHESTRATOR: Day {day} session")
        print(f"{'='*60}")

        # Decision 1: Create plan if missing
        if not profile.study_plan:
            print("\n📋 [Orchestrator] No plan yet → delegating to PlannerAgent...")
            self.planner.create_or_update_plan(client, profile)

        # Decision 2: Pick today's topic
        if day <= len(profile.study_plan):
            today = profile.study_plan[day - 1]
            topic = today.get("topic", profile.topics[0])
            print(f"\n🎯 [Orchestrator] Today's focus: {topic}")
        else:
            topic = profile.weak_areas[0] if profile.weak_areas else profile.topics[0]
            print(f"\n🎯 [Orchestrator] Adaptive focus (weak area): {topic}")

        # Decision 3: Research if not cached
        self.researcher.research_topic(client, topic, profile)

        # Decision 4: Generate quiz
        questions = self.quiz.generate_quiz(client, profile, [topic])

        # Decision 5: Collect answers
        answers: dict = {}
        if simulate:
            import random
            for q in questions:
                qid = str(q.get("id", ""))
                # Simulate ~70% accuracy
                answers[qid] = q["correct"] if random.random() < 0.7 \
                    else random.choice([k for k in "ABCD" if k != q["correct"]])
            print(f"\n  [Demo] Simulated {len(questions)} answers")
        else:
            for q in questions:
                qid = str(q.get("id", ""))
                print(f"\nQ{qid}: {q['question']}")
                for k, v in q.get("options", {}).items():
                    print(f"  {k}) {v}")
                answers[qid] = input("  Your answer: ").strip().upper()

        # Decision 6: Evaluate
        eval_result = self.evaluator.evaluate(profile, questions, answers)

        # Decision 7: Re-plan if weak areas found (autonomous adaptation!)
        if profile.weak_areas and day % 2 == 0:
            print("\n🔄 [Orchestrator] Weak areas detected → triggering re-plan...")
            self.planner.create_or_update_plan(client, profile)

        # Decision 8: Generate report
        report_path = self.reporter.save_report(profile, eval_result, day)

        print(f"\n✅ [Orchestrator] Day {day} complete!")
        return {"day": day, "topic": topic, "score": eval_result["score"],
                "accuracy": f"{profile.overall_accuracy():.0%}", "report": report_path}

    def run_program(self, client: genai.Client, profile: StudentProfile,
                    sessions: int = 2, simulate: bool = True) -> None:
        print(f"\n{'='*60}")
        print(f"  🚀 AUTONOMOUS EXAM PREP — {sessions} sessions")
        print(f"  Student: {profile.name} | Exam: {profile.exam_date}")
        print(f"{'='*60}")

        for day in range(1, sessions + 1):
            result = self.run_session(client, profile, day, simulate)
            print(f"\n  Day {result['day']} summary: {result['topic']} | {result['score']} | Accuracy: {result['accuracy']}")

        print(f"\n{'='*60}")
        print(f"  🎓 Program complete for {profile.name}!")
        print(f"  Final accuracy: {profile.overall_accuracy():.0%}")
        print(f"  Strong: {profile.strong_areas or 'Still building...'}")
        print(f"  Needs review: {profile.weak_areas or 'All looking good!'}")
        print(f"  Reports saved in: {REPORTS_PATH}")
        print(f"{'='*60}")


# ── Entry Point ───────────────────────────────────────────────────────────────
def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in your .env file before running.")

    client = genai.Client(api_key=key)
    coach = AutonomousExamCoach()

    # Student provides goal ONCE — system handles the rest
    profile = StudentProfile(
        name="Alex",
        exam_date="2026-06-20",
        topics=["Binary Search Tree", "Dynamic Programming"],
        study_hours_per_day=2,
    )

    # Run 2 autonomous sessions (set simulate=False for real interactive mode)
    coach.run_program(client, profile, sessions=2, simulate=True)


if __name__ == "__main__":
    main()
