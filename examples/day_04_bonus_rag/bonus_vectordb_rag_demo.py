import os
import pathlib

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai


DB_PATH = pathlib.Path("resources/vector_store/day4_rag_db")


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection("course_chunks", embedding_function=ef)


def seed(collection) -> None:
    if collection.count() > 0:
        return
    docs = [
        "BST average operations: O(log n), worst O(n).",
        "DP uses memoization/tabulation for overlapping subproblems.",
        "Graphs: BFS gives shortest path in unweighted graphs.",
    ]
    collection.add(documents=docs, ids=["d1", "d2", "d3"], metadatas=[{"source": "seed"}] * 3)


def rag_answer(question: str) -> str:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    collection = get_collection()
    seed(collection)
    results = collection.query(query_texts=[question], n_results=2)
    context = "\n".join(results.get("documents", [[]])[0])

    client = genai.Client(api_key=key)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer in 5 lines."
    return client.models.generate_content(model="gemini-2.0-flash", contents=prompt).text


def main() -> None:
    q = "How efficient is BST search and when can it degrade?"
    print(rag_answer(q))


if __name__ == "__main__":
    main()

