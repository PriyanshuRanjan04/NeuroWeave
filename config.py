import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# LLM Provider Selection
# ---------------------------------------------------------------------------
# Set LLM_PROVIDER in your .env to switch between providers:
#   "groq" — Free tier, runs Llama 3.3-70B via Groq (default, used in development)
#   "grok" — xAI's native Grok model, paid, recommended for best quality
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# --- Groq settings (free — https://console.groq.com) ---
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_MODEL    = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# --- Grok / xAI settings (paid — https://console.x.ai) ---
GROK_API_KEY  = os.getenv("GROK_API_KEY")
GROK_MODEL    = "grok-3"
GROK_BASE_URL = "https://api.x.ai/v1"

# --- Convenience: active provider settings (used by grok_client.py) ---
if LLM_PROVIDER == "grok":
    ACTIVE_API_KEY  = GROK_API_KEY
    ACTIVE_MODEL    = GROK_MODEL
    ACTIVE_BASE_URL = GROK_BASE_URL
else:  # default: groq
    LLM_PROVIDER    = "groq"   # normalise any unknown value to groq
    ACTIVE_API_KEY  = GROQ_API_KEY
    ACTIVE_MODEL    = GROQ_MODEL
    ACTIVE_BASE_URL = GROQ_BASE_URL

# --- Other services ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- LightRAG ---
LIGHTRAG_WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_storage")

# --- PDF Chunking ---
CHUNK_SIZE    = 1000   # words per chunk
CHUNK_OVERLAP = 200    # words of overlap between consecutive chunks


def validate_config() -> bool:
    """Check that all required environment variables are set."""
    missing = []

    # Check the key for whichever provider is selected
    if LLM_PROVIDER == "grok" and not GROK_API_KEY:
        missing.append("GROK_API_KEY  (required when LLM_PROVIDER=grok)")
    elif LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        missing.append("GROQ_API_KEY  (required when LLM_PROVIDER=groq)")

    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")

    if missing:
        print(f"WARNING: The following environment variables are missing: {', '.join(missing)}")
        return False
    return True


# Run validation on import
print(f"[NeuroWeave] LLM provider: {LLM_PROVIDER.upper()}  |  model: {ACTIVE_MODEL}")
validate_config()
