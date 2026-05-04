import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
def main() -> None:
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY in .env before running this example.")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = ChatPromptTemplate.from_template("Explain {topic} for a QA engineer moving to AI in 5 bullets.")
    chain = prompt | llm
    result = chain.invoke({"topic": "RAG evaluation"})
    print(result.content)
if __name__ == "__main__":
    main()
