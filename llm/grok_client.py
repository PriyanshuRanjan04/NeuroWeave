import os
import sys
import logging
from typing import List, Optional, Generator

from openai import OpenAI
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embedding fallback via sentence-transformers (Groq does not serve embeddings)
_SENTENCE_MODEL = None  # lazy-loaded

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set. Groq calls will fail.")

# Initialize the OpenAI-compatible Groq client
client = OpenAI(
    api_key=GROQ_API_KEY or "not-set",
    base_url=GROQ_BASE_URL,
)

logger.info("xAI (Grok) client initialized successfully.")


# -------------------------------------------------------------------
# Internal helper
# -------------------------------------------------------------------
def _build_messages(messages: List[dict], system_prompt: Optional[str]) -> List[dict]:
    """Prepend system prompt to message list if provided."""
    if system_prompt:
        return [{"role": "system", "content": system_prompt}] + list(messages)
    return list(messages)


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

async def chat(
    messages: List[dict],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7
) -> str:
    """
    Send messages to Groq and return the response text.

    Args:
        messages:      List of {"role": ..., "content": ...} dicts
        system_prompt: Optional system instruction prepended to messages
        temperature:   Sampling temperature (0.0 – 1.0)

    Returns:
        Response text as a string. Returns an error string on failure.
    """
    full_messages = _build_messages(messages, system_prompt)
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=full_messages,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        return text
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower() or "tokens per day" in err.lower():
            msg = (
                "⚠️ **Groq daily token limit reached.** "
                "Please wait ~2 hours for the quota to reset, or swap in a new "
                "`GROQ_API_KEY` in your `.env` file and restart the app."
            )
            logger.warning(f"Groq 429 rate limit hit: {e}")
            return msg
        logger.error(f"Error connecting to Groq API: {e}")
        return f"Error connecting to Groq API: {e}"


async def stream_chat(
    messages: List[dict],
    system_prompt: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Same as chat() but streams tokens as they arrive.

    Yields:
        Individual text chunks (strings) from the streaming response.
    """
    full_messages = _build_messages(messages, system_prompt)
    try:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=full_messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower() or "tokens per day" in err.lower():
            logger.warning(f"Groq 429 rate limit hit during streaming: {e}")
            yield (
                "⚠️ **Groq daily token limit reached.** "
                "Please wait ~2 hours for the quota to reset, or swap in a new "
                "`GROQ_API_KEY` in your `.env` file and restart the app."
            )
        else:
            logger.error(f"Error during Groq streaming: {e}")
            yield f"⚠️ Error during streaming: {e}"


async def embed(text: str) -> List[float]:
    """
    Generate a text embedding vector.

    Groq does not provide an embedding endpoint, so we use
    sentence-transformers as a local fallback (model: all-MiniLM-L6-v2,
    output dimension: 384, padded to 1536 to match Supabase table schema).

    Returns:
        List of floats representing the embedding vector (length 1536).
    """
    global _SENTENCE_MODEL

    try:
        if _SENTENCE_MODEL is None:
            from sentence_transformers import SentenceTransformer
            _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence-transformer model loaded for embeddings.")

        vector: List[float] = _SENTENCE_MODEL.encode(text).tolist()

        # Pad / truncate to 1536 dimensions to match Supabase VECTOR(1536) column
        if len(vector) < 1536:
            vector = vector + [0.0] * (1536 - len(vector))
        elif len(vector) > 1536:
            vector = vector[:1536]

        return vector

    except Exception as e:
        logger.error(f"Embedding failed: {e}. Returning zero vector.")
        return [0.0] * 1536


async def test_connection() -> bool:
    """
    Send a simple message to Groq to verify the API key is working.

    Returns:
        True if the connection succeeds, False otherwise.
    """
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
        )
        _ = response.choices[0].message.content
        return True
    except Exception as e:
        logger.error(f"Groq connection test failed: {e}")
        return False


# -------------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    async def _run_test():
        print("Testing Groq API connection...")
        success = await test_connection()
        if success:
            print("Groq connection successful ✅")
        else:
            print("Groq connection failed ❌ — check your GROQ_API_KEY in .env")

    asyncio.run(_run_test())
