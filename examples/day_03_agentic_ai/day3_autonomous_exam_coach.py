import os
import pathlib
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai


DB_PATH = pathlib.Path("resources/vector_store/day3_coach_db")
REPORTS_PATH = pathlib.Path("resources/study_reports")


@dataclass
class StudentProfile:
    name: str
    exam_date: str
    topics: list[str]
    weak_areas: list[str] = field(default_factory=list)


class ResearchAgent:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(DB_PATH))
        ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection("coach_knowledge", embedding_function=ef)
        if self.collection.count() == 0:
            self.collection.add(
                documents=[
                    "BST average operations are O(log n), worst case O(n) when skewed.",
                    "DP solves overlapping subproblems with memoization or tabulation.",
                ],
                ids=["k1", "k2"],
                metadatas=[{"topic": "BST"}, {"topic": "DP"}],
            )

    def retrieve(self, topic: str) -> str:
        result = self.collection.query(query_texts=[topic], n_results=2)
        return "\n".join(result.get("documents", [[]])[0])


class AutonomousExamCoach:
    def __init__(self, gemini_client: genai.Client):
        self.gemini = gemini_client
        self.research = ResearchAgent()

    def run_session(self, profile: StudentProfile, day: int) -> str:
        topic = profile.weak_areas[0] if profile.weak_areas else profile.topics[0]
        context = self.research.retrieve(topic)
        prompt = (
            f"Student: {profile.name}, exam: {profile.exam_date}, day: {day}.\n"
            f"Topic focus: {topic}.\n"
            f"Context:\n{context}\n\n"
            "Generate: plan + 3 quiz questions + one risk warning."
        )
        response = self.gemini.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        report_file = REPORTS_PATH / f"day_{day:02d}_{profile.name.lower()}_report.md"
        report_file.write_text(response.text, encoding="utf-8")
        return str(report_file)


def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    coach = AutonomousExamCoach(genai.Client(api_key=key))
    profile = StudentProfile(name="Alex", exam_date="2026-06-20", topics=["Binary Search Tree", "Dynamic Programming"])
    path = coach.run_session(profile, day=1)
    print(f"Report generated: {path}")


if __name__ == "__main__":
    main()

