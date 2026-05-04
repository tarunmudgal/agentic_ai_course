import os
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from langgraph.graph import END, StateGraph


class State(TypedDict):
    topic: str
    notes: str
    quiz: str


def plan_node(state: State) -> dict:
    return {"notes": f"Plan: learn {state['topic']} basics -> practice -> revise."}


def quiz_node(state: State) -> dict:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return {"quiz": "Set GEMINI_API_KEY to generate quiz."}
    client = genai.Client(api_key=key)
    q = client.models.generate_content(model="gemini-2.0-flash", contents=f"Generate 3 MCQs on {state['topic']}")
    return {"quiz": q.text}


def build_graph():
    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("quiz", quiz_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "quiz")
    g.add_edge("quiz", END)
    return g.compile()


def main() -> None:
    graph = build_graph()
    out = graph.invoke({"topic": "Binary Search Tree", "notes": "", "quiz": ""})
    print(out["notes"])
    print("\nQuiz:\n", out["quiz"])


if __name__ == "__main__":
    main()

