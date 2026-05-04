import os
import pathlib

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai


DB_PATH = pathlib.Path("resources/vector_store/day3_orchestration_db")


def get_collection() -> chromadb.api.models.Collection.Collection:
    client = chromadb.PersistentClient(path=str(DB_PATH))
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection("study_chunks", embedding_function=ef)


def seed_if_needed(collection) -> None:
    if collection.count() > 0:
        return
    docs = [
        "Binary Search Tree supports average O(log n) search when balanced.",
        "Dynamic Programming works for overlapping subproblems and optimal substructure.",
        "Graph traversal starts with DFS or BFS depending on requirement.",
    ]
    collection.add(documents=docs, ids=["bst", "dp", "graph"], metadatas=[{"topic": "BST"}, {"topic": "DP"}, {"topic": "Graph"}])


def retrieve_context(collection, query: str) -> str:
    result = collection.query(query_texts=[query], n_results=2)
    docs = result.get("documents", [[]])[0]
    return "\n".join(docs)


def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    collection = get_collection()
    seed_if_needed(collection)
    context = retrieve_context(collection, "How fast is BST search?")

    client = genai.Client(api_key=key)
    prompt = (
        "Answer using the provided context first.\n\n"
        f"Context:\n{context}\n\n"
        "Question: Explain BST search complexity in 4 lines."
    )
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    print(response.text)


if __name__ == "__main__":
    main()

