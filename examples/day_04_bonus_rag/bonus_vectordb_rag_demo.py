"""
DAY 4 — VectorDB & RAG: Teaching AI to Know YOUR Knowledge
----------------------------------------------------------
The problem this day solves:
  StudyBuddy v1-v3 only knows what Gemini was trained on (general internet).
  What if you want it to answer questions about YOUR notes, YOUR professor's slides?

The solution: RAG = Retrieval Augmented Generation
  1. Store YOUR documents in ChromaDB (a local vector database).
  2. When a question arrives, search ChromaDB for the MOST RELEVANT chunks.
  3. Feed ONLY those chunks to Gemini → accurate, grounded answers!

Analogy:
  Imagine an open-book exam vs closed-book exam.
  Without RAG: Gemini uses only its "memory" (training data) — like closed-book.
  With RAG:    Gemini looks up your notes first — like open-book.
  Which gives better specific answers? Open-book every time!

Why NOT just paste the whole textbook into the prompt?
  - Context limits: Gemini can only "see" ~128k tokens at once.
  - A textbook is millions of tokens.
  - RAG picks the RELEVANT 3-5 paragraphs — efficient and focused.

Run this file:
  python examples/day_04_bonus_rag/bonus_vectordb_rag_demo.py
"""

import os
import pathlib

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai


DB_PATH = pathlib.Path("resources/vector_store/day4_rag_db")

# ── Your "knowledge base" — pretend these are excerpts from your textbook ─────
# In a real project, you would load these from PDF / markdown files.
CS_NOTES = [
    {
        "id": "bst_definition",
        "text": (
            "A Binary Search Tree (BST) is a tree where every node has at most two children. "
            "The left child is always SMALLER than the parent, and the right child is always LARGER. "
            "This ordering property makes lookups very efficient."
        ),
        "topic": "BST",
    },
    {
        "id": "bst_complexity",
        "text": (
            "BST operations (search, insert, delete) are O(log n) on average when the tree is balanced. "
            "In the worst case (when all insertions are in sorted order), the tree becomes a linked list "
            "and operations degrade to O(n)."
        ),
        "topic": "BST",
    },
    {
        "id": "dp_definition",
        "text": (
            "Dynamic Programming (DP) solves complex problems by breaking them into smaller overlapping "
            "subproblems and storing each result so it is never recomputed. "
            "It requires two properties: optimal substructure and overlapping subproblems."
        ),
        "topic": "DP",
    },
    {
        "id": "dp_vs_recursion",
        "text": (
            "Pure recursion recomputes the same subproblems exponentially (e.g., fib(5) calls fib(4) "
            "and fib(3), each of which call fib(3) again). DP eliminates duplication using memoization "
            "(top-down: cache results in a dict) or tabulation (bottom-up: fill an array)."
        ),
        "topic": "DP",
    },
    {
        "id": "sort_merge",
        "text": (
            "Merge sort divides the list in half, recursively sorts each half, then merges them. "
            "Time: O(n log n) always. Space: O(n) auxiliary. It is a stable sort — equal elements "
            "keep their original order. Preferred when stability matters."
        ),
        "topic": "Sorting",
    },
]


def get_collection() -> chromadb.api.models.Collection.Collection:
    """Create (or load) a ChromaDB collection on disk."""
    # PersistentClient saves the DB to disk — survives script restarts.
    # Use chromadb.Client() instead for an in-memory (temporary) DB.
    chroma = chromadb.PersistentClient(path=str(DB_PATH))
    ef = embedding_functions.DefaultEmbeddingFunction()
    return chroma.get_or_create_collection("course_chunks", embedding_function=ef)


def seed_if_needed(collection) -> None:
    """Add study notes to the DB on first run. Skip if already loaded."""
    if collection.count() > 0:
        print(f"  📚 Knowledge base already has {collection.count()} chunks — skipping seed.")
        return
    print("  🌱 Seeding knowledge base with study notes...")
    collection.add(
        documents=[n["text"] for n in CS_NOTES],
        ids=[n["id"] for n in CS_NOTES],
        metadatas=[{"topic": n["topic"]} for n in CS_NOTES],
    )
    print(f"  ✅ {len(CS_NOTES)} chunks stored in ChromaDB.")


def search(collection, query: str, n: int = 2) -> list[str]:
    """
    SEMANTIC search — finds chunks similar in MEANING, not just keywords.

    Example:
      Query:  "how fast is tree lookup?"
      Finds:  bst_complexity (about O(log n)) — even though the query has no matching words!
    """
    results = collection.query(query_texts=[query], n_results=n)
    return results.get("documents", [[]])[0]


def answer_with_rag(question: str, collection, gemini_client: genai.Client) -> str:
    """
    Full RAG pipeline:
      ① RETRIEVE: find the most relevant chunks from ChromaDB.
      ② AUGMENT:  build a prompt that includes those chunks as context.
      ③ GENERATE: Gemini reads the context and answers — grounded in YOUR notes!
    """
    print(f"\n{'─'*50}")
    print(f"  ❓ Question: {question}")
    print("  🔍 Step ① — Searching VectorDB for relevant context...")
    chunks = search(collection, question, n=2)
    for i, c in enumerate(chunks):
        print(f"     Chunk {i+1}: {c[:80]}...")

    context = "\n\n".join(chunks)
    prompt = (
        "You are StudyBuddy. Answer the question using ONLY the study notes below.\n"
        "If the answer isn't in the notes, say 'I don't have that in my notes'.\n\n"
        f"=== STUDY NOTES ===\n{context}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        "=== ANSWER (cite which note you used) ==="
    )
    print("  🤖 Step ② + ③ — Augmenting prompt & generating answer...")
    answer = gemini_client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    ).text
    print(f"\n  ✅ Answer:\n{answer}")
    return answer


def answer_without_rag(question: str, gemini_client: genai.Client) -> str:
    """Plain Gemini — no context, just training knowledge."""
    return gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"You are a CS tutor. Answer in 3 sentences: {question}",
    ).text


def compare(question: str, collection, gemini_client: genai.Client) -> None:
    """Side-by-side: plain Gemini vs RAG Gemini."""
    print(f"\n{'='*60}")
    print(f"  📊 RAG vs Plain Gemini Comparison")
    print(f"  Q: {question}")
    print(f"{'='*60}")
    print("\n❌ WITHOUT RAG (Gemini training knowledge only):")
    print("-" * 40)
    print(answer_without_rag(question, gemini_client))
    print("\n✅ WITH RAG (using YOUR study notes from ChromaDB):")
    print("-" * 40)
    answer_with_rag(question, collection, gemini_client)
    print("\n💡 Key insight: With RAG, answers are GROUNDED in your specific material!")


def main() -> None:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in your .env file before running.")

    gemini_client = genai.Client(api_key=key)
    collection = get_collection()
    seed_if_needed(collection)

    # Demo 1: Semantic search — shows that meaning-based search works
    print("\n" + "=" * 60)
    print("  🔍 Demo 1: Semantic Search (not keyword search!)")
    print("=" * 60)
    for query in ["how fast does tree search run?", "what avoids recomputing subproblems?"]:
        print(f"\n🔎 Query: '{query}'")
        for chunk in search(collection, query, n=1):
            print(f"  Found: {chunk[:120]}...")

    # Demo 2: Compare RAG vs plain Gemini
    compare(
        "What is BST time complexity and when does it degrade?",
        collection, gemini_client,
    )

    # Demo 3: Interactive
    print("\n" + "=" * 60)
    print("  💬 Ask your own question (or press Enter to skip):")
    print("=" * 60)
    q = input("Your question: ").strip()
    if q:
        answer_with_rag(q, collection, gemini_client)


if __name__ == "__main__":
    main()

