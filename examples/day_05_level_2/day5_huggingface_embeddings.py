from sentence_transformers import SentenceTransformer


def cosine_similarity(vec_a, vec_b) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> None:
    # Small model for fast local embeddings in course demos.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [
        "Binary Search Tree average search is O(log n).",
        "Depth First Search uses a stack and explores deep first.",
        "QA engineers validate expected outcomes and edge cases.",
    ]
    vectors = model.encode(texts, normalize_embeddings=False)

    query = "How efficient is BST lookup?"
    query_vec = model.encode([query], normalize_embeddings=False)[0]
    scores = [(text, cosine_similarity(query_vec, vec)) for text, vec in zip(texts, vectors)]
    scores.sort(key=lambda item: item[1], reverse=True)

    print(f"Query: {query}\n")
    for text, score in scores:
        print(f"score={score:.4f} | {text}")


if __name__ == "__main__":
    main()
