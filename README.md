# NeuroWeave

NeuroWeave is an AI-powered document intelligence application that allows users to:
1. Upload PDF documents.
2. Chat with the system to ask questions about the uploaded content using a Retrieval-Augmented Generation (RAG) pipeline backed by LightRAG and Supabase pgvector.
3. Request a highly structured, 20,000-word handbook generated iteratively from the PDF content using the xAI Grok API.

---

## Setup Instructions

### 1. Clone the repository
Clone this project to your local machine (or navigate to the extracted project folder).
```bash
git clone <repository-url>
cd NeuroWeave
```

### 2. Install dependencies
Ensure you have Python 3.9+ installed. Install the required packages via `pip`:
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy the provided `.env.example` file to create a standard `.env` file:
```bash
cp .env.example .env
```
Open the `.env` file and fill in your credentials:
- `GROK_API_KEY`: Your xAI Grok API key.
- `SUPABASE_URL`: Your Supabase PostgreSQL database URL.
- `SUPABASE_KEY`: Your Supabase Anon public key.
- `LIGHTRAG_WORKING_DIR`: (Optional) Local folder for LightRAG graph storage. Defaults to `./lightrag_storage`.

### 4. Run Supabase SQL
NeuroWeave utilizes `pgvector` for similarity search. Navigate to the SQL Editor in your Supabase dashboard and run the following commands:

```sql
-- Create vector extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the required table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT,
    content TEXT,
    chunk_index INTEGER,
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create a function to similarity search documents
CREATE OR REPLACE FUNCTION match_documents (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id uuid,
  filename text,
  content text,
  chunk_index int,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    documents.id,
    documents.filename,
    documents.content,
    documents.chunk_index,
    1 - (documents.embedding <=> query_embedding) AS similarity
  FROM documents
  WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
  ORDER BY documents.embedding <=> query_embedding
  LIMIT match_count;
$$;
```

### 5. Run the Application
Finally, start the Gradio UI application:
```bash
python app.py
```
Open the URL (typically `http://127.0.0.1:7860/`) returned in your terminal to interact with NeuroWeave!
