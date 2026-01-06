import requests
import json
import os

# --- SETUP ---
PDF_PATH = "C:/Users/aswat/Downloads/new_cyber_law.pdf"  # <--- UPDATE THIS
TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImFkbWluIiwiZXhwIjoxNzY1OTQ3MDU1fQ.pR4Y-G73gGGSH1lwA1eMWhQNlWdUNjpaki_LihzhC7A"  # <--- UPDATE THIS
URL = "http://127.0.0.1:5000/api/admin/pdf/preview/"

if not os.path.exists(PDF_PATH):
    print("Error: File not found.")
    exit()

print(f"Uploading {PDF_PATH} for preview...")
try:
    with open(PDF_PATH, 'rb') as f:
        files = {'file': f}
        headers = {'Authorization': TOKEN}
        response = requests.post(URL, headers=headers, files=files)

    if response.status_code == 200:
        data = response.json()
        text_content = data['preview_text']
        filename = data['filename']

        # Save to a text file for editing
        with open("law_draft.txt", "w", encoding="utf-8") as text_file:
            text_file.write(text_content)

        print("SUCCESS! Text extracted.")
        print("Check the file 'law_draft.txt' in your project folder.")
        print("Edit that text file, remove junk, and save it.")
    else:
        print("Error:", response.text)
except Exception as e:
    print("Failed:", e)