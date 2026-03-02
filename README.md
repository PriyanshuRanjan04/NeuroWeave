# 🧠 NeuroWeave

> Upload PDFs. Ask questions. Generate 20,000-word handbooks — all through conversation.

---

## 📌 What is NeuroWeave?

NeuroWeave is an AI-powered document intelligence application. Upload your PDF documents, have contextual conversations about the content, and generate comprehensive 20,000+ word structured handbooks — all through a simple chat interface powered by a hybrid Knowledge Graph + Vector Search RAG pipeline.

No complex UI. No manual summarization. Just upload, ask, and generate.

---

## ✨ Core Features

- 📄 **PDF Upload** — Upload one or multiple PDF files directly in the interface
- 💬 **Contextual Chat** — Ask questions and get answers grounded in your uploaded documents
- 📚 **Handbook Generation** — Generate a 20,000+ word structured handbook on any topic in your PDFs
- 🔗 **Knowledge Graph** — LightRAG builds an entity-relationship graph for deep, smart retrieval
- ⚡ **Vector Search** — Supabase pgvector powers fast semantic similarity search
- 📎 **Citations** — Generated handbooks reference content from your uploaded source materials

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Gradio | Chat interface and PDF upload |
| **PDF Processing** | pdfplumber | Extract and clean text from PDFs |
| **RAG Engine** | LightRAG | Build knowledge graph from document content |
| **Vector Database** | Supabase (pgvector) | Store and retrieve document embeddings |
| **LLM** | Grok 3 / 4 (xAI API) | Power chat responses and handbook generation |
| **Backend** | Python 3.9+ (async/await) | Core application logic |
| **Config** | python-dotenv | Manage API keys and environment variables |

---

## 📁 Project Structure

```
NeuroWeave/
├── app.py                        # 🚀 Main entry point — launches the Gradio UI
├── config.py                     # ⚙️  Loads and validates environment variables
├── requirements.txt              # 📦 Project dependencies
├── .env                          # 🔐 Your API keys (never commit this)
├── .env.example                  # 📋 Template showing required environment variables
├── README.md                     # 📖 Project documentation
│
├── pdf_processing/
│   └── extractor.py              # Handles PDF upload, text extraction and chunking
│
├── rag/
│   ├── lightrag_client.py        # Sets up LightRAG and builds knowledge graph
│   └── retriever.py              # Queries the knowledge graph for relevant context
│
├── database/
│   └── supabase_client.py        # Manages Supabase connection and vector operations
│
├── llm/
│   ├── grok_client.py            # Wrapper for Grok API calls (chat, stream, embed)
│   └── handbook_generator.py     # Orchestrates 20,000-word handbook generation
│
└── utils/
    └── helpers.py                # Shared utilities — text cleaning, chunking
```

---

## 🧠 How Handbook Generation Works

Standard LLMs struggle to generate very long documents in a single pass. NeuroWeave uses the **LongWriter / AgentWrite** technique:

1. **Plan** — Generate a full outline with 10–12 sections, each with a target word count (1,500–2,500 words)
2. **Write** — Generate each section individually using context retrieved from the knowledge graph
3. **Compile** — Assemble all sections into a structured document with a table of contents, executive summary, and references

This produces coherent, well-structured handbooks exceeding 20,000 words.

---

## 🔄 System Architecture

```mermaid
flowchart TD
    U(["👤 User"])

    subgraph Gradio["🖥️ Gradio Web Interface"]
        PU["📄 PDF Upload"]
        Q["💬 Question Input"]
        CR["📩 Chat Response"]
        HB["📚 20k Handbook"]
    end

    subgraph proc["📦 Processing Pipeline"]
        PP["🔧 PDF Processing\npdfplumber"]
        TC["📝 Text Chunks"]
    end

    subgraph storage["🗄️ Knowledge Storage"]
        SB[("🟢 Supabase pgvector\nVector Embeddings")]
        LR[("🟣 LightRAG\nKnowledge Graph")]
    end

    RAG["🔍 Hybrid RAG Retriever"]
    LLM["⚡ Grok LLM\nxAI API"]

    U --> PU --> PP --> TC
    TC --> SB
    TC --> LR
    SB --> RAG
    LR --> RAG
    U --> Q --> RAG
    RAG --> LLM
    LLM --> CR --> U
    LLM --> HB --> U
```

---

## 🔑 Environment Variables

Before running the app, you will need:

| Variable | Description |
|---|---|
| `GROK_API_KEY` | Your xAI API key for Grok |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Your Supabase anon/public key |
| `LIGHTRAG_WORKING_DIR` | Local folder path for LightRAG graph storage |

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
