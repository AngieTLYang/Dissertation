import google.generativeai as genai
import PIL.Image
import os
import time

# --- Configuration ---
# Read the API key from file
try:
    with open(rf"C:\Users\88690\Desktop\Dissertation\image processing\genai_api_key.txt", "r") as f:
        api_key = f.read().strip()
    genai.configure(api_key=api_key)
except FileNotFoundError:
    raise FileNotFoundError(
        "Please create a file named 'genai_api_key.txt' with your API key inside."
    )

# --- Load Image and Model ---
# Load the image using the Pillow library
try:
    image_path = 'C:/Users/88690/Desktop/Dissertation/imagesFromPhone/photo_2025-08-22_19-36-13.jpg'
    img = PIL.Image.open(image_path)
except FileNotFoundError:
    raise FileNotFoundError(f"The image file was not found at {image_path}")

# Use the specific model name you want to use
#model = genai.GenerativeModel('gemini-2.5-flash-lite')
model = genai.GenerativeModel('gemma-3n-e4b-it')

# --- Generate Content with Text and Image ---
start_time = time.time()
response = model.generate_content([
    "Give me the answer to the question pointed to by the red pen. State the question first, then provide the answer in a clear and concise manner.",
    img
])
end_time = time.time()

print(f"Model reasoning time: {end_time - start_time:.2f} seconds")

# Print the model's response
print(response.text)
