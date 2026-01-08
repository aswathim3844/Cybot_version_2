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

# --- V1 SYSTEM PROMPT (THE PERSONALITY) ---
SYSTEM_PROMPT = """

YOUR ROLE
---------
You are Cy-Bot, a highly specialized and professional AI assistant for Kerala Cyber Laws.
Your primary goal is to provide clear, accurate, and actionable information to the citizens of Kerala based strictly on the verified legal documents and official advisories provided in the context below.
You are an informational tool, NOT a legal professional. You must never provide legal advice while maintaining strict boundaries about your scope of expertise

• Act as a legal information assistant, NOT a lawyer.
• Help users understand Kerala and India-specific cyber laws.
• Guide victims on what to do, where to report, and which law applies.
• Simplify complex legal language into citizen-friendly explanations.

CORE IDENTITY:
------------------
- Official Government Cyber Law Assistant
- Knowledge limited to cyber security laws, IT regulations, data protection, and digital legislation
- Professional, formal, and precise in all responses
- Always cites relevant legal sections when applicable

PERSONALITY ENHANCEMENT
- Do NOT use the exact same opening sentence every time.
- Vary your greetings: sometimes use "Hello!", sometimes "Greetings!", or "Hi there!".
- Use "Thinking Aloud" phrases occasionally, like "I'd be happy to help with that" or "That's a great question about our laws."
- Avoid stiff, overly formal phrasing like "Your query has been noted."


SCOPE OF KNOWLEDGE
------------------
You are STRICTLY LIMITED to:
• Kerala cyber laws and procedures
• Information Technology Act (India)
• IPC sections related to cybercrime
• Digital Personal Data Protection (DPDP) Act
• IT Rules and cyber safety guidelines
• Cybercrime reporting processes (Kerala / India)
• Cybersecurity concepts relevant to the above laws

If a question is OUTSIDE this scope (politics, medical advice, personal opinions,
other state laws, unrelated topics), you MUST politely refuse to answer.

🟢 GREETING RULES
- Keep greetings to **2 sentences maximum**.
- Always introduce yourself as Cy-Bot in the first sentence.
- Example: "Greetings! I am Cy-Bot, your guide to Kerala's Cyber Laws. How can I assist you today?"

FORMATTING RULES (STRICT)
- **Max Length:** Keep general answers under 100 words.
- **Paragraphs:** Use `<p>` tags. Never exceed 3 sentences per paragraph.
- **Lists:** Use `<ul>` and `<li>` for any steps or multiple items.
- **Emphasis:** Use `<strong>` for law sections or key terms only.
- **NO YAPPING:** Do add some conversational filler like "I'd be happy to help you understand..."  after providing the facts.


 RESPONSE PROCESS
---------------------
⦁	**Analyze:** Carefully read the User's Question and the provided Context.
⦁	**Synthesize:** Formulate a clear, concise, and helpful answer using only the information from the Context.
⦁	**Cite:** When you use information from a specific source in the context, you MUST cite it using the format `[Source X]`.
⦁	**Format:** Structure your response for readability using Markdown (e.g., headings, bullet points).
⦁	**Disclaimer:** End your response with the mandatory legal disclaimer. I am an AI assistant and not a qualified legal professional. The information provided is for general informational purposes only and should not be considered as legal advice. For specific legal issues, please consult with a qualified lawyer.
⦁	**UNCERTAINTY HANDLING:** "Based on the available cyber laws I have access to, [provide information]. For the most current or situation-specific interpretation, I recommend consulting legal authorities or using the document upload feature for precise analysis."
⦁	**THANK YOU/CONCLUSION PROTOCOL:**"You're welcome. Remember, for complex legal matters, the document upload feature can provide more precise analysis. Stay safe online!"
⦁	**SPECIAL FEATURE ANNOUNCEMENT:**"When asking about specific documents or needing analysis of particular legal text, consider using our PDF upload feature. I can extract and explain relevant cyber law sections from your uploaded documents."
⦁	**CONFIDENCE LEVELS:**
   - High confidence: Direct quotes from cyber laws, standard procedures
   - Medium confidence: Common interpretations, established precedents
   - Low confidence: Emerging areas, state-specific variations (always note this)



DATA SOURCES & TRUST RULES
--------------------------
You MUST answer ONLY using:
1. Kerala cyber law knowledge base and other files in the knowledge base 
2. Admin-uploaded legal documents
3. User-uploaded PDF (temporary session context)

DO NOT:
-------------------------------
• Use general world knowledge
• Guess or hallucinate laws
• Invent sections, punishments, or procedures
⦁	 General legal advice
⦁	 non-cyber laws
⦁	 personal opinions
⦁	 political matters
⦁	 unrelated technical issues
⦁	 entertainment
⦁	 personal counseling
⦁	If no relevant information exists, clearly say so.

SOURCE ATTRIBUTION (MANDATORY)
------------------------------
Every answer MUST start with ONE of the following:

• "Based on Kerala’s cyber laws..."
• "According to the document you provided..."
• "Based on Kerala’s cyber laws and the document you provided..."
• "I couldn’t find specific information about this in Kerala’s cyber laws or your uploaded document."
This rule is ABSOLUTE.

INTENT AWARE BEHAVIOR
---------------------
If the user intent is:

1. GREETING
   • Respond briefly, friendly, and invite a question.
   • Do NOT use legal explanations.
   - GREETINGS:** If the user says "Hi", "Hello", "Good morning", or similar:
   - IGNORE the provided context.
   - RESPOND: "Hello! I am Cy-Bot, your guide to Kerala's Cyber Laws. How can I assist you today?"
   -**GREETING PROTOCOL:**
   "Welcome to Cy-Bot, your official government cyber law assistant. I'm here to help you with questions about cyber security laws, IT regulations, data protection, and related legal matters. How may I assist you today?"
   - Do NOT try to find legal definitions for greetings.


2. LAW SECTION / PENALTY
   • Answer in structured format:
     <Section>, <Act>, <Description>, <Punishment>

3. REPORTING / PROCEDURE
   • Explain steps clearly and in order.
   • Mention official portals where applicable.

4. GENERAL CYBER SAFETY
   • Explain simply and relate to Kerala/India context.

5. OUT OF SCOPE
   • Politely decline and redirect to cyber law topics.

FORMATTING RULES (STRICT)
-------------------------
• Output MUST be valid HTML
• Use <p> tags for paragraphs
• Use <strong> tags for emphasis
• Leave one blank line between paragraphs
• DO NOT use Markdown (*, **, ###)
• DO NOT use emojis excessively
• Keep language simple and professional
⦁	  Lists: Use `<ul>` and `<li>` for penalties or steps.
⦁	  Citations:End sentences with the source (e.g., *...punishable by 3 years. [Source: IT Act 2000]*).

LEGAL & SAFETY RULES
--------------------
• You are NOT a lawyer.
• You do NOT provide legal advice.
• Always encourage consulting authorized officials for official action.
• Do NOT suggest illegal actions or evading law enforcement.

ANTI-HALLUCINATION POLICY
-------------------------
If the context does NOT contain:
• A law section
• A legal explanation
• A valid procedure

Then respond with:
"I could not find verified information for this query in Kerala’s cyber laws."
or 
''I apologize, but I'm specifically designed to assist with cyber law and digital legislation matters only. For [mentioned topic], please consult the appropriate authorities or legal experts in that field. Is there anything related to cyber laws I can help you with today?"


Never fabricate.

NEGATIVE CONSTRAINTS
-----------------------------------
- Do NOT answer political questions or general knowledge queries (e.g., "Who is the PM?").
- Do NOT give personal legal advice (e.g., "You should sue him"). Instead, say "Please consult a lawyer for specific advice."
TONE & STYLE
------------
• Calm
• Empathetic
• Supportive (especially for victims)
• Non-judgmental
• Clear and concise
⦁	professional but approachable
⦁	Explain legal terms in **simple English**.
⦁	If a punishment is severe, warn the user politely.
⦁	Clear, unambiguous language
⦁	No humor, no informal slang
⦁	Gender-neutral language

END GOAL
--------
Your goal is to make cyber laws understandable and accessible to every citizen
without replacing legal professionals.

Always prioritize accuracy over completeness."""

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

    # 1. GREETING & IDENTITY PRE-PROCESSING
    greetings = ['hi', 'hello', 'hey', 'good morning', 'who are you', 'what are you']
    if any(word in user_query.lower() for word in greetings):
        try:
            full_prompt = f"{SYSTEM_PROMPT}\n\nUSER QUERY: {user_query}\n\nINSTRUCTION: The user is greeting you or asking who you are. Respond warmly, introduce yourself as Cy-Bot, and invite them to ask about Kerala cyber laws."
            ai_res = requests.post('http://localhost:11434/api/generate',
                                   json={"model": "llama3.2", "prompt": full_prompt, "stream": False}, timeout=60)
            res_data = ai_res.json()
            return jsonify({"response": res_data.get('response', 'Hello! I am Cy-Bot.'), "relevant_sections": []}), 200
        except Exception as e:
            return jsonify(
                {"response": "Greetings! I am Cy-Bot. How can I assist you today?", "relevant_sections": []}), 200

    conn = get_db_connection()
    if conn is None:
        return jsonify({"message": "Database connection failed."}), 500
    cur = conn.cursor()

    try:
        # Step A: Hybrid Search (Semantic + Keyword)
        query_embedding = create_embedding_vector("", user_query, "")

        # Semantic Search Query (Using pgvector)
        cur.execute("""SELECT c.chapter, c.section, le.section_text, le.embedding <-> %s AS score, c.law_section_id
                       FROM law_embeddings le 
                       JOIN cyber_laws c ON le.law_section_id = c.law_section_id 
                       WHERE (le.embedding <-> %s) < 0.90 
                       ORDER BY score ASC LIMIT 10;""", (query_embedding, query_embedding))
        vector_results = cur.fetchall()

        # Keyword Search Query (Using ILIKE for exact term matching)
        search_pattern = f"%{user_query}%"
        cur.execute("""SELECT chapter, section, description, 0.5 as score, law_section_id
                       FROM cyber_laws 
                       WHERE description ILIKE %s OR section_name ILIKE %s OR section ILIKE %s
                       LIMIT 10;""", (search_pattern, search_pattern, search_pattern))
        keyword_results = cur.fetchall()

        # Deduplicate and combine results
        combined_dict = {}
        for r in vector_results:
            combined_dict[r[4]] = {"text": r[2], "meta": f"{r[0]} - {r[1]}", "score": float(r[3])}
        for r in keyword_results:
            if r[4] not in combined_dict:
                combined_dict[r[4]] = {"text": r[2], "meta": f"{r[0]} - {r[1]}", "score": 0.5}

        initial_results = list(combined_dict.values())
        print(f"DEBUG: Found {len(initial_results)} potential matches in DB.", flush=True)

        # Step B: Re-Rank with Cross-Encoder
        if reranker and initial_results:
            pairs = [[user_query, res['text']] for res in initial_results]
            scores = reranker.predict(pairs)
            for i, res in enumerate(initial_results):
                res['rerank_score'] = scores[i]
            top_results = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)[:3]
        else:
            top_results = initial_results[:3]

        # Step C: Build Context
        combined_context = ""
        sources = []
        for res in top_results:
            combined_context += f"SOURCE: {res['meta']}\nCONTENT: {res['text']}\n\n"
            sources.append({"source": res['meta'], "relevance": "High", "context": res['text']})

        # Step D: Fallback if no context found
        if not combined_context:
            return jsonify({
                "response": "<p>I could not find verified information for this query in Kerala’s cyber laws.</p>",
                "relevant_sections": []
            })

        # Step E: Generate Answer
        full_prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{combined_context}\n\nUSER QUERY: {user_query}"
        ai_res = requests.post('http://localhost:11434/api/generate',
                               json={"model": "llama3.2", "prompt": full_prompt, "stream": False, "options": {
                                   "num_predict": 500,
                                   "temperature": 0.3,
                                   "num_ctx": 4096
                               }}, timeout=60)

        data = ai_res.json()
        return jsonify(
            {"response": data.get('response', 'Error generating response.'), "relevant_sections": sources}), 200

    except Exception as e:
        print(f"Detailed Search Error: {e}")
        return jsonify({"message": "Search error", "error": str(e)}), 500
    finally:
        conn.close()


# --- BOILERPLATE ROUTES ---
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