from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np
# This is required to make psycopg2 understand and correctly handle the VECTOR type
import pgvector.psycopg2

# --- Configuration: ⚠️ CONFIRM THESE DETAILS ⚠️ ---
DB_HOST = "localhost"
DB_NAME = "cyber_law_db"  # Must match your database name
DB_USER = "postgres"  # Your PostgreSQL username
DB_PASS = "Aswathim3844@postgresql"  # **REPLACE THIS WITH YOUR ACTUAL PASSWORD**

# Initialize the embedding model once. 'all-MiniLM-L6-v2' is small and fast.
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings_and_store():
    """Reads law text from cyber_laws, creates embeddings, and stores them in law_embeddings."""
    conn = None
    try:
        # 1. Connect to the database
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)

        # Register the pgvector type handler with the connection
        # This tells psycopg2 how to convert NumPy arrays to the PostgreSQL VECTOR type and vice versa
        pgvector.psycopg2.register_vector(conn)
        cur = conn.cursor()
        print("Database connection established and pgvector registered.")

        print("Clearing existing vector data from law_embeddings...")
        # TRUNCATE is fast and resets the table, ensuring clean inserts
        cur.execute("TRUNCATE TABLE law_embeddings;")
        conn.commit()
        print("Existing data cleared.")


        # 2. Extract data (The text we want to vectorize)
        # We join several descriptive columns to create one rich text chunk
        sql_fetch = """
        SELECT law_section_id, chapter, section_name, description
        FROM cyber_laws;
        """
        cur.execute(sql_fetch)
        records = cur.fetchall()

        # Prepare text and IDs for the model
        law_ids = [item[0] for item in records]

        # Combine the relevant text into a single string for better embedding context
        texts = [f"Chapter: {item[1]}. Section Name: {item[2]}. Details: {item[3]}" for item in records]

        # 3. Generate embeddings (The most time-consuming step initially)
        print(f"Generating embeddings for {len(texts)} law sections...")
        # The model converts the list of text strings into a list of 384-dimension NumPy arrays
        embeddings = model.encode(texts, convert_to_tensor=False)
        print("Embeddings generation complete.")

        # 4. Load data into law_embeddings table
        # We insert the Law ID, the full text chunk, and the NumPy array (embedding)
        insert_sql = "INSERT INTO law_embeddings (law_section_id, section_text, embedding) VALUES (%s, %s, %s);"

        insert_data = []
        for law_id, text, embedding in zip(law_ids, texts, embeddings):
            # The pgvector handler handles the direct insertion of the NumPy array
            insert_data.append((law_id, text, embedding))

        cur.executemany(insert_sql, insert_data)
        conn.commit()
        print(f"✅ Successfully inserted {len(insert_data)} vectors into 'law_embeddings'.")

    except Exception as e:
        print(f"❌ An error occurred during embedding: {e}")
        if conn: conn.rollback()
        # **Beginner Tip:** If you see 'No module named torch', see the error resolution below.
    finally:
        if conn: conn.close()


if __name__ == "__main__":
    generate_embeddings_and_store()