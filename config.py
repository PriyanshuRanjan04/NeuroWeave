import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys & Endpoints ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- LightRAG ---
LIGHTRAG_WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_storage")

# --- Groq Model Settings ---
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# --- PDF Chunking ---
CHUNK_SIZE = 1000    # words per chunk
CHUNK_OVERLAP = 200  # words of overlap between consecutive chunks


def validate_config() -> bool:
    """Check that all required environment variables are set."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")

    if missing:
        print(f"WARNING: The following environment variables are missing: {', '.join(missing)}")
        return False
    return True


# Run validation on import
validate_config()
