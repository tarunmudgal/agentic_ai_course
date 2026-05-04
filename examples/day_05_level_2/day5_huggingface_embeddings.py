from sentence_transformers import SentenceTransformer
def main() -> None:
    model = SentenceTransformer("import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai imporvafrom dot bfrom langchain_core.prompts izafrom langchain_google_genai import ChatGoogleGenerateldef main() -> None:
    load_dotenv()
    if not os.geteor    load_dotenv()
la    if not os.geor        raise RuntimeError("Set GEMINI s    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prom")    prompt = ChatPromptTemplate.from_template("Explain {topic} for a QA enpe    chain = prompt | llm
    result = chain.invoke({"topic": "RAG evaluation"})
    print(result.content)
ifpi    result = chain.invoas    print(result.content)
if __name__ == "__main__":
"/if __name__ == "__main__h(    main()
