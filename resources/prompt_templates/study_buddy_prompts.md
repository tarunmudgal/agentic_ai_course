# Study Buddy Prompt Templates

## 1) Concept explainer

"Explain `{topic}` for a 2nd-year CSE student in 5 bullets, then add one real-world use case and one common mistake."

## 2) QA-oriented breakdown

"Explain `{topic}` and then provide test scenarios: happy path, edge case, invalid input, and performance risk."

## 3) Quiz generator

"Generate 5 MCQs for `{topic}` with answers and one-line explanation per answer."

## 4) Retrieval-grounded answer

"Answer using only this context. If context is insufficient, say what is missing.\nContext:\n{context}\nQuestion:\n{question}"

## 5) Hallucination check

"Review the answer below. Mark each claim as Supported, Weakly Supported, or Unsupported based on context.\nContext:\n{context}\nAnswer:\n{answer}"

