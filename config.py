import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Required variables
GROK_API_KEY = os.getenv("GROK_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
LIGHTRAG_WORKING_DIR = os.getenv("LIGHTRAG_WORKING_DIR", "./lightrag_storage")

def validate_config():
    """Check if all required environment variables are set."""
    missing = []
    if not GROK_API_KEY:
        missing.append("GROK_API_KEY")
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
