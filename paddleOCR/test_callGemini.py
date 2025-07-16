# Install the SDK first:
# pip install --upgrade google-genai

import google.generativeai as genai
import os

# Read the API key from file
with open("genai_api_key.txt", "r") as f:
    api_key = f.read().strip()

# Configure the SDK
genai.configure(api_key=api_key)

models = genai.list_models()
for m in models:
    print(m.name, m.supported_generation_methods)

model = genai.GenerativeModel('gemma-3n-e4b-it')
response = model.generate_content("explain computer architecture to a 5-year-old")
print(response.text)