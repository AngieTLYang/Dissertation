import google.generativeai as genai

with open("genai_api_key.txt", "r") as f:
    api_key = f.read().strip()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemma-3n-e4b-it')

def ask_model(question: str) -> str:
    response = model.generate_content(question)
    return response.text if response and hasattr(response, "text") else ""
