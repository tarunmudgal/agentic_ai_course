import os

import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None


st.set_page_config(page_title="AI Study Buddy", page_icon="🤖")
st.title("AI Study Buddy - Day 5 Demo")
st.caption("Streamlit UI for quick GenAI + QA-minded study flows")


topic = st.text_input("Topic", value="Binary Search Tree")
focus = st.selectbox("Focus", ["CSE concepts", "QA test strategy", "Interview prep"])

if st.button("Generate notes"):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        st.error("Missing GEMINI_API_KEY or google-genai package.")
    else:
        client = genai.Client(api_key=api_key)
        prompt = (
            f"Topic: {topic}\n"
            f"Audience focus: {focus}\n"
            "Return: (1) core explanation, (2) 3 quiz questions, (3) 2 failure cases to test."
        )
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        st.markdown(response.text)

