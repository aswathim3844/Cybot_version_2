from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from functools import wraps
import bcrypt
import jwt
import datetime
import psycopg2
import psycopg2.extras
import pgvector.psycopg2
from sentence_transformers import SentenceTransformer, CrossEncoder
import pdfplumber
import io
import nltk
import os
import json
import requests
import uuid
from dotenv import load_dotenv
import fitz
import numpy as np
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# LOAD SECRETS
load_dotenv()

app = Flask(__name__)

# CONFIG FROM .ENV
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "cyber_law_db")
DB_USER = os.getenv("DB_USER", "postgres")
VAULT_KEY = os.getenv("VAULT_KEY", "CyberVigilance2025")

# --- V1 SYSTEM PROMPT ---
SYSTEM_PROMPT = """
YOUR ROLE
---------
You are Cy-Bot, a highly specialized and professional AI assistant for Kerala Cyber Laws.
Your primary goal is to provide clear, accurate, and actionable information based strictly on verified legal documents.

CORE IDENTITY:
- Official Government Cyber Law Assistant
- Professional, formal, and precise. Always cites relevant legal sections.

SCOPE:
- Kerala and India-specific cyber laws (IT Act, IPC).
- Digital Personal Data Protection (DPDP) Act.
- Cybersecurity reporting processes.

RULES:
- Max Length: Under 100 words per response.
- Format: Valid HTML (<p>, <ul>, <li>, <strong>).
- Attribution: Start with "Based on Kerala’s cyber laws..." or "According to the document provided...".
- Disclaimer: End with: "I am an AI assistant and not a qualified legal professional. The information provided is for general informational purposes only and should not be considered as legal advice."
"""

# --- ADMIN USERS ---
ADMIN_USERS = {
    "admin": "$2b$12$.F9IaN6lMkcNi1N0hH3KlOHH9PZNUfe9OnB/qox4umBBphVEMNs2G"
}

# --- GLOBAL STORES ---
session_pdf_store: Dict[str, Any] = {}
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# --- EMBEDDING MODELS ---
try:
    print("Loading Embedding Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Loading Re-Ranker Model...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    reranker = None


# --- HELPERS ---
def get_db_connection():
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        if model:
            pgvector.psycopg2.register_vector(conn)
        return conn
    except Exception as e:
        print(f"Database error: {e}")
        return None


def create_embedding_vector(chapter, section_name, description):
    text_chunk = f"Chapter: {chapter}. Section Name: {section_name}. Details: {description}"
    return model.encode([text_chunk], convert_to_tensor=False)[0]


def chunk_text(text, chunk_size=500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_chunk_length + sentence_words <= chunk_size:
            current_chunk.append(sentence)
            current_chunk_length += sentence_words
        else:
            chunks.append(" ".join(current_chunk))
            overlap_sentences = current_chunk[-3:]
            current_chunk = overlap_sentences + [sentence]
            current_chunk_length = sum(len(s.split()) for s in current_chunk)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['username']
        except Exception:
            return jsonify({'message': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)

    return decorated


def log_event(conn, table, rec_id, action, old_val, new_val, user):
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audit_logs (table_name, record_id, action_type, old_data, new_data, changed_by) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (table, rec_id, action, json.dumps(old_val, default=str), json.dumps(new_val, default=str), user))
    except Exception as e:
        print(f"Audit Logging Error: {e}")


# --- ROUTES ---

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    stored_hash = ADMIN_USERS.get(username)
    if stored_hash and bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
        conn = get_db_connection()
        log_event(conn, 'auth', 0, 'LOGIN', None, {"status": "success"}, username)
        conn.commit()
        conn.close()
        session['admin_logged_in'] = True
        session['admin_user'] = username
        expiration = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=24)
        token = jwt.encode({'username': username, 'exp': expiration}, app.config['SECRET_KEY'], algorithm="HS256")
        return jsonify({"message": "Login successful", "access_token": token}), 200
    return jsonify({"message": "Invalid credentials"}), 401


@app.route('/api/query', methods=['POST'])
def chatbot_query():
    data = request.get_json()
    user_query = data.get('query', '').strip()
    if not user_query:
        return jsonify({"message": "Query required"}), 400

    # GREETING PRE-PROCESSING
    greetings = ['hi', 'hello', 'hey', 'good morning', 'who are you']
    if any(word in user_query.lower() for word in greetings):
        full_prompt = f"{SYSTEM_PROMPT}\n\nUSER QUERY: {user_query}\n\nINSTRUCTION: Respond as Cy-Bot."
        ai_res = requests.post('http://localhost:11434/api/generate',
                               json={"model": "llama3.2", "prompt": full_prompt, "stream": False}, timeout=60)
        return jsonify({"response": ai_res.json()['response'], "relevant_sections": []}), 200

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Step A: Database Vector Search
        query_embedding = create_embedding_vector("", user_query, "")

        # RELAXED THRESHOLD: Increased to 0.90 to ensure results are found for re-ranking
        cur.execute("""SELECT c.chapter, c.section, le.section_text, le.embedding <-> %s AS score
                       FROM law_embeddings le
                       JOIN cyber_laws c ON le.law_section_id = c.law_section_id
                       WHERE (le.embedding <-> %s) < 0.90
                       ORDER BY score ASC LIMIT 10;""", (query_embedding, query_embedding))

        rows = cur.fetchall()
        initial_results = []
        for r in rows:
            initial_results.append({
                "text": r[2],
                "meta": f"{r[0]} - {r[1]}",
                "score": r[3]
            })

        print(f"DEBUG: Found {len(initial_results)} potential matches in DB.", flush=True)

        # Step B: Re-Rank with Cross-Encoder
        top_results = []
        if reranker and initial_results:
            print("DEBUG: Re-ranking matches...", flush=True)
            pairs = [[user_query, res['text']] for res in initial_results]
            scores = reranker.predict(pairs)
            print(f"DEBUG: Re-ranker scores: {scores}", flush=True)

            for i, res in enumerate(initial_results):
                res['rerank_score'] = scores[i]

            # Sort by highest similarity score
            top_results = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)[:3]
        else:
            top_results = initial_results[:3]

        # Step C: Build Context
        combined_context = ""
        sources = []
        for res in top_results:
            combined_context += f"SOURCE: {res['meta']}\nCONTENT: {res['text']}\n\n"
            sources.append({"source": res['meta'], "relevance": "High", "context": res['text']})

        # Step D: Fallback
        if not combined_context:
            return jsonify({
                "response": "<p>I could not find verified information for this query in Kerala’s cyber laws.</p>",
                "relevant_sections": []
            })

        # Step E: Generate AI Response
        full_prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{combined_context}\n\nUSER QUERY: {user_query}"
        print(f"DEBUG: Sending context to AI (Length: {len(combined_context)})", flush=True)

        ai_res = requests.post('http://localhost:11434/api/generate',
                               json={
                                   "model": "llama3.2",
                                   "prompt": full_prompt,
                                   "stream": False,
                                   "options": {
                                       "num_predict": 500,  # Sufficient length for full legal answers
                                       "temperature": 0.3,  # Low temp for factual accuracy
                                       "num_ctx": 4096
                                   }
                               }, timeout=60)

        return jsonify({"response": ai_res.json()['response'], "relevant_sections": sources}), 200

    except Exception as e:
        print(f"Detailed Search Error: {e}", flush=True)
        return jsonify({"message": "Search error", "error": str(e)}), 500
    finally:
        conn.close()


# --- REMAINING BOILERPLATE ---
@app.route('/')
def index(): return render_template('chat.html')


@app.route('/admin')
def view_admin():
    if not session.get('admin_logged_in'): return redirect('/login')
    return render_template('admin.html')


@app.route('/login')
def view_login(): return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True)