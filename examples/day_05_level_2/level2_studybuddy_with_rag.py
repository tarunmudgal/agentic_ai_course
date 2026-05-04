import os
import pathlib

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai


DB_PATH = pathlib.Path("resources/vector_store/day5_studybuddy_rag")


def get_collection():
    c = chromadb.PersistentClient(path=str(DB_PATH))
    ef = embedding_functions.DefaultEmbeddingFunction()
    return c.get_or_create_collection("studybuddy", embedding_function=ef)


def research_topic_with_rag(topic: str) -> str:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")

    collection = get_collection()
    if collection.count() == 0:
        collection.add(
            documents=["BST left < root < right. Average lookup O(log n)."],
            ids=["seed_bst"],
            metadatas=[{"topic": "BST"}],
        )

    found = collection.query(query_texts=[topic], n_results=2)
    context = "\n".join(found.get("documents", [[]])[0])

    client = genai.Client(api_key=key)
    prompt = f"Context:\n{context}\n\nCreate concise study notes for {topic}."
    answer = client.models.generate_content(model="gemini-2.0-flash", contents=prompt).text

    if not context.strip():
        new_id = topic.lower().replace(" ", "_")
        collection.add(documents=[answer], ids=[new_id], metadatas=[{"topic": topic}])
    return answer


def main() -> None:
    print(research_topic_with_rag("Heap Data Structure"))


if __name__ == "__main__":
    main()

