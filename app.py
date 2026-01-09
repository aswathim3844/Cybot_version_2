from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from functools import wraps
import bcrypt
import jwt
import datetime
import psycopg2
import psycopg2.extras
import pgvector.psycopg2
# 1. NEW IMPORT: CrossEncoder for Re-ranking
from sentence_transformers import SentenceTransformer, CrossEncoder
import pdfplumber
import io
import nltk
import os
import json
import requests
import uuid
from decimal import Decimal
# NEW IMPORTS FOR ENV & RAG
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
⦁   **Analyze:** Carefully read the User's Question and the provided Context.
⦁   **Synthesize:** Formulate a clear, concise, and helpful answer using only the information from the Context.
⦁   **Cite:** When you use information from a specific source in the context, you MUST cite it using the format `[Source X]`.
⦁   **Format:** Structure your response for readability using Markdown (e.g., headings, bullet points).
⦁   **Disclaimer:** End your response with the mandatory legal disclaimer. I am an AI assistant and not a qualified legal professional. The information provided is for general informational purposes only and should not be considered as legal advice. For specific legal issues, please consult with a qualified lawyer.
⦁   **UNCERTAINTY HANDLING:** "Based on the available cyber laws I have access to, [provide information]. For the most current or situation-specific interpretation, I recommend consulting legal authorities or using the document upload feature for precise analysis."
⦁   **THANK YOU/CONCLUSION PROTOCOL:**"You're welcome. Remember, for complex legal matters, the document upload feature can provide more precise analysis. Stay safe online!"
⦁   **SPECIAL FEATURE ANNOUNCEMENT:**"When asking about specific documents or needing analysis of particular legal text, consider using our PDF upload feature. I can extract and explain relevant cyber law sections from your uploaded documents."
⦁   **CONFIDENCE LEVELS:**
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
⦁    General legal advice
⦁    non-cyber laws
⦁    personal opinions
⦁    political matters
⦁    unrelated technical issues
⦁    entertainment
⦁    personal counseling
⦁   If no relevant information exists, clearly say so.

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
⦁     Lists: Use `<ul>` and `<li>` for penalties or steps.
⦁     Citations:End sentences with the source (e.g., *...punishable by 3 years. [Source: IT Act 2000]*).

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
⦁   professional but approachable
⦁   Explain legal terms in **simple English**.
⦁   If a punishment is severe, warn the user politely.
⦁   Clear, unambiguous language
⦁   No humor, no informal slang
⦁   Gender-neutral language

END GOAL
--------
Your goal is to make cyber laws understandable and accessible to every citizen
without replacing legal professionals.

Always prioritize accuracy over completeness."""

# --- NEW: MALAYALAM SYSTEM PROMPT ---
MALAYALAM_SYSTEM_PROMPT = """
[YOUR ORIGINAL PROMPT HERE - COPY PASTE THE ENTIRE SYSTEM_PROMPT ABOVE]

**CRUCIAL MALAYALAM OUTPUT INSTRUCTION:**
- You must respond **only in Malayalam**.
- The main body of your explanation, greetings, and all conversational text must be in Malayalam.
- **CRITICAL EXCEPTION:** Do not translate specific legal acts, section numbers, or technical terms. Keep them in their original English form.
    - Examples: "Information Technology Act, 2000", "Section 66A", "IPC", "Cyber Appellate Tribunal", "Phishing".
- Format your response in valid HTML, just as described in the original prompt.
- Your persona and all other rules from the original prompt remain the same.
"""

# --- NEW: SENIOR CITIZEN MODE INSTRUCTION ---
SENIOR_CITIZEN_INSTRUCTION = """
**IMPORTANT: SENIOR CITIZEN MODE ACTIVATED.**
You MUST adhere to the following rules in addition to all previous instructions:
- Use very simple, short, and clear sentences.
- Explain complex legal terms in a very easy-to-understand way.
- Be extra patient, empathetic, and encouraging in your tone.
- Break down any procedures into simple, numbered steps.
- Avoid using jargon. If a legal term is necessary, explain it immediately in parentheses.
- Your goal is to make the user feel safe, understood, and confident.
"""

# --- ADMIN USERS ---
ADMIN_USERS = {
    "admin": "$2b$12$.F9IaN6lMkcNi1N0hH3KlOHH9PZNUfe9OnB/qox4umBBphVEMNs2G"
}

# --- GLOBAL STORES ---
session_pdf_store: Dict[str, Any] = {}

# FIX: Download BOTH punkt and punkt_tab to prevent server crash
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

# --- EMBEDDING MODELS ---
try:
    print("Loading Embedding Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Loading Re-Ranker Model...")
    # 2. INITIALIZE RE-RANKER (Cross-Encoder)
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


# --- AUDIT LOGGING HELPER ---
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

        # Server-Side Session
        session['admin_logged_in'] = True
        session['admin_user'] = username

        # Generate Token
        expiration = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=24)
        token = jwt.encode({'username': username, 'exp': expiration}, app.config['SECRET_KEY'], algorithm="HS256")
        return jsonify({"message": "Login successful", "access_token": token}), 200
    return jsonify({"message": "Invalid credentials"}), 401


@app.route('/api/logout_log', methods=['POST'])
@token_required
def logout_log(current_user):
    conn = get_db_connection()
    log_event(conn, 'auth', 0, 'LOGOUT', None, {"status": "success"}, current_user)
    conn.commit()
    conn.close()
    session.clear()
    return jsonify({"success": True}), 200


@app.route('/api/admin/pdf/preview/', methods=['POST'])
@token_required
def preview_pdf(current_user):
    if 'file' not in request.files: return jsonify({"message": "No file uploaded"}), 400
    file = request.files['file']
    HEADER_CUTOFF, FOOTER_CUTOFF = 50, 50
    full_text = []
    try:
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                width, height = page.width, page.height
                if height > (HEADER_CUTOFF + FOOTER_CUTOFF):
                    bbox = (0, HEADER_CUTOFF, width, height - FOOTER_CUTOFF)
                    text = page.crop(bbox).extract_text() or ""
                else:
                    text = page.extract_text() or ""
                full_text.append(f"\n--- Page {i + 1} ---\n{text}")
        return jsonify({"filename": file.filename, "preview_text": "\n".join(full_text)}), 200
    except Exception as e:
        return jsonify({"message": "Error reading PDF.", "error": str(e)}), 500


@app.route('/api/admin/pdf/confirm/', methods=['POST'])
@token_required
def confirm_pdf_ingestion(current_user):
    data = request.get_json()
    filename = data.get('filename', 'Unknown.pdf')
    final_text = data.get('final_text')

    if not final_text:
        return jsonify({"message": "No text provided."}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    stats = {"inserted": 0, "updated": 0, "skipped": 0}

    try:
        # Save the master document record
        cur.execute("INSERT INTO uploaded_documents (file_name, raw_pdf_data) VALUES (%s, %s) RETURNING document_id;",
                    (filename, final_text.encode('utf-8')))
        doc_id = cur.fetchone()[0]
        log_event(conn, 'uploaded_documents', doc_id, 'UPLOAD_PDF', None, {"file": filename}, current_user)

        chunks = chunk_text(final_text)

        for i, chunk in enumerate(chunks):
            # Versioning & Conflict Detection Logic
            chapter_name = filename
            section_label = f"Part {i + 1}"

            cur.execute("""
                SELECT law_section_id, description, version_number 
                FROM cyber_laws 
                WHERE chapter = %s AND section = %s AND is_active = TRUE;
            """, (chapter_name, section_label))

            existing_record = cur.fetchone()

            if existing_record:
                old_id, old_desc, old_version = existing_record

                # Check for exact duplicate
                if old_desc.strip() == chunk.strip():
                    stats["skipped"] += 1
                    continue  # Skip this chunk as it already exists exactly as is

                # If content is different, perform Versioning (Phase 1, Item 15)
                cur.execute("UPDATE cyber_laws SET is_active = FALSE WHERE law_section_id = %s;", (old_id,))
                new_version = old_version + 1
                stats["updated"] += 1
            else:
                new_version = 1
                stats["inserted"] += 1

            # Insert new version or new record
            sec_name = f"{filename} - Chunk {i + 1}"
            cur.execute("""
                INSERT INTO cyber_laws (chapter, section, section_name, description, punishment, document_id, version_number, is_active) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE) RETURNING law_section_id;
            """, (chapter_name, section_label, sec_name, chunk, "N/A", doc_id, new_version))

            law_id = cur.fetchone()[0]
            log_event(conn, 'cyber_laws', law_id, 'BULK_INGEST', None, {"source": filename, "chunk": i + 1},
                      current_user)

            # Generate AI Embedding (Fixed numpy to list conversion)
            emb = create_embedding_vector(filename, sec_name, chunk)
            cur.execute("INSERT INTO law_embeddings (law_section_id, section_text, embedding) VALUES (%s, %s, %s);",
                        (law_id, chunk, emb.tolist()))

        conn.commit()
        return jsonify({
            "message": "Ingestion Complete.",
            "stats": stats
        }), 201

    except Exception as e:
        conn.rollback()
        print(f"Error trace: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


# --- NEW ROUTE FOR CHATBOT PDF UPLOADS ---
@app.route('/api/chat/upload_session', methods=['POST'])
def upload_chat_pdf():
    # Check if a file was sent in the request
    if 'file' not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files['file']
    # Create a unique ID for this user's specific session
    session_id = str(uuid.uuid4())

    try:
        # Extract text from the PDF using pdfplumber
        with pdfplumber.open(file) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)

        if not text.strip():
            return jsonify({"message": "PDF appears to be empty or unscannable."}), 400

        # Store in the global session_pdf_store (Temporary memory)
        session_pdf_store[session_id] = {
            "filename": file.filename,
            "content": text
        }

        # Return the session_id so the frontend can use it in queries
        return jsonify({
            "message": "File uploaded successfully!",
            "session_id": session_id,
            "filename": file.filename
        }), 200
    except Exception as e:
        return jsonify({"message": "Error processing PDF", "error": str(e)}), 500


@app.route('/api/query', methods=['POST'])
def chatbot_query():
    data = request.get_json()
    user_query = data.get('query', '').strip()
    # NEW: Get language and mode from request (Saketh's Feature)
    language = data.get('language', 'en')
    mode = data.get('mode', 'normal')

    if not user_query:
        return jsonify({"message": "Query required"}), 400

    # --- NEW: GREETING & IDENTITY PRE-PROCESSING ---
    greetings = ['hi', 'hello', 'hey', 'good morning', 'who are you', 'what are you', 'നമസ്കാരം']
    is_greeting = any(word in user_query.lower() for word in greetings)

    if is_greeting:
        try:
            # UPDATED: Construct the final prompt by combining base prompt and senior mode instruction
            base_prompt = MALAYALAM_SYSTEM_PROMPT if language == 'ml' else SYSTEM_PROMPT
            final_prompt = (SENIOR_CITIZEN_INSTRUCTION + base_prompt) if mode == 'senior' else base_prompt

            full_prompt = f"{final_prompt}\n\nUSER QUERY: {user_query}\n\nINSTRUCTION: The user is greeting you or asking who you are. Respond warmly, introduce yourself as Cy-Bot, and invite them to ask about Kerala cyber laws. Use slightly different wording each time."

            ai_res = requests.post('http://localhost:11434/api/generate',
                                   json={"model": "llama3.2", "prompt": full_prompt, "stream": False}, timeout=60)

            res_data = ai_res.json()
            if 'response' in res_data:
                return jsonify({"response": res_data['response'], "relevant_sections": []}), 200
            else:
                fallback_greeting = "നമസ്കാരം! ഞാൻ സൈ-ബോട്ട്, കേരളത്തിലെ സൈബർ നിയമങ്ങളുടെ നിങ്ങളുടെ ഗൈഡ്. എങ്ങനെ സഹായിക്കാം?" if language == 'ml' else "Hello! I am Cy-Bot, your guide to Kerala's Cyber Laws. How can I assist you today?"
                return jsonify({"response": fallback_greeting, "relevant_sections": []}), 200
        except Exception as e:
            print(f"Ollama Greeting Error: {e}")
            return jsonify({"response": "Greetings! I am Cy-Bot. How can I assist you with Kerala's cyber laws today?",
                            "relevant_sections": []}), 200

    conn = get_db_connection()
    if conn is None:
        return jsonify({"message": "Database connection failed. Please check your .env configuration."}), 500

    cur = conn.cursor()

    try:
        query_embedding = create_embedding_vector("", user_query, "")

        # Step A: Fetch 10 Candidates
        cur.execute("""SELECT c.chapter, c.section, le.section_text, le.embedding <-> %s AS score 
                       FROM law_embeddings le 
                       JOIN cyber_laws c ON le.law_section_id = c.law_section_id 
                       WHERE (le.embedding <-> %s) < 0.80
                       ORDER BY score ASC LIMIT 10;""", (query_embedding, query_embedding))

        initial_results = []
        for r in cur.fetchall():
            initial_results.append({
                "text": r[2],
                "meta": f"{r[0]} - {r[1]}",
                "score": r[3]
            })

        print(f"DEBUG: Found {len(initial_results)} initial results from DB with threshold 0.80")

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

        # Step D: Generate Answer
        if not combined_context:
            fallback_response = "<p>കേരളത്തിലെ സൈബർ നിയമങ്ങളിൽ ഈ ചോദ്യത്തിന് സ്ഥിരീകരിച്ച വിവരങ്ങൾ ഞാൻ കണ്ടെത്താനായില്ല. ദയവായി വീണ്ടും രൂപപ്പെടുത്തുക അല്ലെങ്കിൽ ഒരു നിയമ വിദഗ്ധനെ കാണുക.</p>" if language == 'ml' else "<p>I could not find verified information for this query in Kerala’s cyber laws. Please try rephrasing or consulting a legal professional.</p>"
            return jsonify({"response": fallback_response, "relevant_sections": []})

        # UPDATED: Construct the final prompt by combining base prompt and senior mode instruction
        base_prompt = MALAYALAM_SYSTEM_PROMPT if language == 'ml' else SYSTEM_PROMPT
        final_prompt = (SENIOR_CITIZEN_INSTRUCTION + base_prompt) if mode == 'senior' else base_prompt

        # Inject the V1 Personality and Context
        full_prompt = f"{final_prompt}\n\nCONTEXT:\n{combined_context}\n\nUSER QUERY: {user_query}"

        ai_res = requests.post('http://localhost:11434/api/generate',
                               json={"model": "llama3.2", "prompt": full_prompt, "stream": False, "options": {
                                   "num_predict": 100,  # Stops the bot after ~100-150 words (Saves time)
                                   "temperature": 0.5,  # Balanced: precise but still human-like
                                   "top_p": 0.9,
                                   "num_ctx": 2048  # Smaller context window for faster processing

                               }}, timeout=60)

        data = ai_res.json()
        if 'response' in data:
            return jsonify({"response": data['response'], "relevant_sections": sources}), 200
        else:
            return jsonify({"message": "Ollama did not return a valid response."}), 500

    except Exception as e:
        print(f"Detailed Search Error: {e}")
        return jsonify({"message": "Search error", "error": str(e)}), 500
    finally:
        conn.close()


# --- FRONTEND ROUTES ---
@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/chat')
def view_chat():
    return render_template('chat.html')


@app.route('/admin')
def view_admin():
    # Server-Side Security Check
    if not session.get('admin_logged_in'):
        return redirect('/login')
    return render_template('admin.html')


@app.route('/login')
def view_login():
    return render_template('login.html')


# --- UNIVERSAL CRUD ---
TABLE_CONFIG = {
    'cyber_laws': 'law_section_id',
    'scam_advisories': 'scam_id',
    'cybercells': 'station_id',
    'legal_guidance': 'crime_id',  # FIXED: Changed from guidance_id to crime_id
    'reporting_procedures': 'report_id',
    'user_queries': 'query_id',
    'audit_logs': 'log_id'
}


@app.route('/api/admin/universal/<table_name>', methods=['GET', 'POST'])
@token_required
def universal_crud(current_user, table_name):
    if table_name not in TABLE_CONFIG: return jsonify({"message": "Invalid table"}), 400

    if table_name == 'audit_logs' and request.method != 'GET':
        return jsonify({"message": "Logs are Immutable"}), 405
    if table_name == 'audit_logs' and request.headers.get('X-Vault-Key') != VAULT_KEY:
        return jsonify({"message": "Unauthorized Vault Key"}), 403

    pk = TABLE_CONFIG[table_name]
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # --- GET (READ) ---
    if request.method == 'GET':
        try:
            cur.execute(f"SELECT * FROM {table_name} WHERE is_active = TRUE ORDER BY {pk} DESC LIMIT 100")
        except psycopg2.Error:
            conn.rollback()
            cur.execute(f"SELECT * FROM {table_name} ORDER BY {pk} DESC LIMIT 100")

        rows = []
        for r in cur.fetchall():
            row_dict = dict(r)
            for k, v in row_dict.items():
                if isinstance(v, Decimal):
                    row_dict[k] = float(v)
                elif isinstance(v, (datetime.date, datetime.datetime)):
                    row_dict[k] = v.isoformat()
                # Display Lists as "Text" in the frontend
                elif isinstance(v, list):
                    row_dict[k] = ", ".join(str(x) for x in v)
            rows.append(row_dict)
        return jsonify(rows)

    # --- POST (CREATE) ---
    if request.method == 'POST':
        data = request.get_json()

        # ======================================================
        # 🛠️ FIX: CONVERT TEXT BACK TO ARRAY FOR DATABASE
        # This fixes the "malformed array literal" error
        # ======================================================
        if table_name == 'legal_guidance' and 'applicable_laws' in data:
            raw_val = data['applicable_laws']
            # If the user sent a string "Sec A, Sec B", turn it into ["Sec A", "Sec B"]
            if isinstance(raw_val, str):
                data['applicable_laws'] = [x.strip() for x in raw_val.split(',') if x.strip()]

        # Also check Cyber Cells if you use arrays there (e.g. services)
        if table_name == 'cybercells' and 'services' in data:
            if isinstance(data['services'], str):
                data['services'] = [x.strip() for x in data['services'].split(',') if x.strip()]
        # ======================================================

        # Conflict Detection Logic
        lookup_map = {
            'scam_advisories': 'scam_name',
            'cybercells': 'station_name',
            'reporting_procedures': 'crime_type',
            'cyber_laws': 'section',
            'legal_guidance': 'crime_type'
        }
        lookup_col = lookup_map.get(table_name)

        if lookup_col and lookup_col in data:
            try:
                cur.execute(
                    f"SELECT {pk}, version_number FROM {table_name} WHERE {lookup_col} = %s AND is_active = TRUE",
                    (data[lookup_col],))
                existing = cur.fetchone()

                if existing:
                    old_id, old_version = existing
                    cur.execute(f"UPDATE {table_name} SET is_active = FALSE WHERE {pk} = %s", (old_id,))
                    data['version_number'] = (old_version or 1) + 1
                else:
                    data['version_number'] = 1
                data['is_active'] = True
            except psycopg2.Error:
                conn.rollback()

        cols = [k for k in data.keys() if k != pk]
        cur.execute(
            f"INSERT INTO {table_name} ({','.join(cols)}) VALUES ({','.join(['%s'] * len(cols))}) RETURNING {pk}",
            [data[k] for k in cols])
        new_id = cur.fetchone()[0]
        log_event(conn, table_name, new_id, 'CREATE', None, data, current_user)
        conn.commit()
        return jsonify({"message": "Created", "id": new_id}), 201


@app.route('/api/admin/universal/<table_name>/<int:record_id>', methods=['PUT', 'DELETE'])
@token_required
def universal_item_ops(current_user, table_name, record_id):
    if table_name == 'audit_logs': return jsonify({"message": "Immutable"}), 405

    pk = TABLE_CONFIG.get(table_name)
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(f"SELECT * FROM {table_name} WHERE {pk}=%s", (record_id,))
    old = cur.fetchone()
    if not old: return jsonify({"msg": "Not found"}), 404

    if request.method == 'DELETE':
        try:
            if table_name == 'cyber_laws':
                cur.execute("DELETE FROM law_embeddings WHERE law_section_id=%s", (record_id,))

            # Try Soft Delete (Archive)
            try:
                cur.execute(f"UPDATE {table_name} SET is_active = FALSE WHERE {pk}=%s", (record_id,))
            except psycopg2.Error:
                conn.rollback()
                cur.execute(f"DELETE FROM {table_name} WHERE {pk}=%s", (record_id,))

            log_event(conn, table_name, record_id, 'DELETE', dict(old), None, current_user)
            conn.commit()
            return jsonify({"msg": "Deleted/Archived successfully"})
        except Exception as e:
            conn.rollback()
            return jsonify({"msg": "Delete failed", "error": str(e)}), 500

    if request.method == 'PUT':
        data = request.get_json()

        # ======================================================
        # 🛠️ FIX FOR EDITING: CONVERT TEXT BACK TO ARRAY
        # This prevents crashes when updating Cyber Cells or Legal Guidance
        # ======================================================
        # 1. Fix Legal Guidance (applicable_laws)
        if table_name == 'legal_guidance' and 'applicable_laws' in data:
            if isinstance(data['applicable_laws'], str):
                data['applicable_laws'] = [x.strip() for x in data['applicable_laws'].split(',') if x.strip()]

        # 2. Fix Cyber Cells (services)
        if table_name == 'cybercells' and 'services' in data:
            if isinstance(data['services'], str):
                data['services'] = [x.strip() for x in data['services'].split(',') if x.strip()]
        # ======================================================

        set_clause = ", ".join([f"{k}=%s" for k in data.keys() if k != pk])
        try:
            cur.execute(f"UPDATE {table_name} SET {set_clause} WHERE {pk}=%s", list(data.values()) + [record_id])
            log_event(conn, table_name, record_id, 'UPDATE', dict(old), data, current_user)
            conn.commit()
            return jsonify({"msg": "Updated"})
        except Exception as e:
            conn.rollback()
            print(f"Update Error: {e}")
            return jsonify({"msg": "Update Failed", "error": str(e)}), 500


@app.route('/api/admin/history/<table_name>/<int:record_id>', methods=['GET'])
@token_required
def get_history(current_user, table_name, record_id):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM audit_logs WHERE table_name=%s AND record_id=%s ORDER BY log_id DESC",
                (table_name, record_id))
    return jsonify([dict(r) for r in cur.fetchall()])


# --- NEW: VERSION HISTORY VIEWER ---
@app.route('/api/admin/versions/<table_name>', methods=['POST'])
@token_required
def view_versions(current_user, table_name):
    """
    Retrieve all versions (Active and Archived) of a specific record
    based on its unique name (e.g., 'Section 66F', 'Phishing Scam').
    """
    if table_name not in TABLE_CONFIG:
        return jsonify({"message": "Invalid table"}), 400

    data = request.get_json()

    # Define which column identifies the "item" for each table
    lookup_map = {
        'scam_advisories': 'scam_name',
        'cybercells': 'station_name',
        'reporting_procedures': 'crime_type',
        'cyber_laws': 'section',
        'legal_guidance': 'crime_type'
    }

    lookup_col = lookup_map.get(table_name)
    search_value = data.get('identifier')  # e.g., "Section 66F"

    if not lookup_col or not search_value:
        return jsonify({"message": "Cannot version this table or missing identifier"}), 400

    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        # Fetch ALL records (Active=True AND Active=False) matching that name
        # Ordered by version number so you see the timeline (v1, v2, v3...)
        query = f"SELECT * FROM {table_name} WHERE {lookup_col} = %s ORDER BY version_number DESC"
        cur.execute(query, (search_value,))
        versions = [dict(r) for r in cur.fetchall()]

        return jsonify({
            "table": table_name,
            "identifier": search_value,
            "total_versions": len(versions),
            "history": versions
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


if __name__ == '__main__':
    app.run(debug=True)