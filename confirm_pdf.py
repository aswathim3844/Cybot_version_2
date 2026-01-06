import requests
import json

# --- SETUP ---
TXT_FILE = "law_draft.txt"  # The file you edited
TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImFkbWluIiwiZXhwIjoxNzY1OTQ3MDU1fQ.pR4Y-G73gGGSH1lwA1eMWhQNlWdUNjpaki_LihzhC7A"  # <--- UPDATE THIS
URL = "http://127.0.0.1:5000/api/admin/pdf/confirm/"
FILENAME = "My_New_Law.pdf"  # Name to save in DB

print(f"Reading {TXT_FILE}...")
try:
    with open(TXT_FILE, "r", encoding="utf-8") as f:
        final_text = f.read()

    print("Sending cleaned text to database...")
    headers = {'Authorization': TOKEN, 'Content-Type': 'application/json'}
    payload = {
        "filename": FILENAME,
        "final_text": final_text
    }

    response = requests.post(URL, headers=headers, json=payload)

    if response.status_code == 201:
        print("SUCCESS! Law ingested into database.")
        print(response.json())
    else:
        print("Error:", response.text)

except Exception as e:
    print("Failed:", e)