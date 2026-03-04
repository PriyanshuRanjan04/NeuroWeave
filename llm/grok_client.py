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
# Configuration — reads from config.py so the provider is driven
# entirely by the LLM_PROVIDER environment variable.
# -------------------------------------------------------------------
from config import ACTIVE_API_KEY, ACTIVE_MODEL, ACTIVE_BASE_URL, LLM_PROVIDER

# Embedding fallback via sentence-transformers (neither Groq nor Grok serve embeddings)
_SENTENCE_MODEL = None  # lazy-loaded

if not ACTIVE_API_KEY:
    logger.warning(
        f"{LLM_PROVIDER.upper()}_API_KEY is not set. "
        "LLM calls will fail — set the correct key in your .env file."
    )

# Initialize the OpenAI-compatible client (works for both Groq and Grok)
client = OpenAI(
    api_key=ACTIVE_API_KEY or "not-set",
    base_url=ACTIVE_BASE_URL,
)

logger.info(f"LLM client initialised — provider: {LLM_PROVIDER.upper()}  model: {ACTIVE_MODEL}")


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
    Send messages to the configured LLM provider and return the response text.

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
            model=ACTIVE_MODEL,
            messages=full_messages,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        return text
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower() or "tokens per day" in err.lower():
            msg = (
                f"⚠️ **{LLM_PROVIDER.upper()} daily token/rate limit reached.** "
                "Please wait and retry, or swap in a new API key in your `.env` file and restart the app."
            )
            logger.warning(f"{LLM_PROVIDER.upper()} 429 rate limit hit: {e}")
            return msg
        logger.error(f"Error connecting to {LLM_PROVIDER.upper()} API: {e}")
        return f"Error connecting to {LLM_PROVIDER.upper()} API: {e}"


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
            model=ACTIVE_MODEL,
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
            logger.warning(f"{LLM_PROVIDER.upper()} 429 rate limit hit during streaming: {e}")
            yield (
                f"⚠️ **{LLM_PROVIDER.upper()} rate limit reached.** "
                "Please wait and retry, or swap in a new API key in your `.env` file and restart the app."
            )
        else:
            logger.error(f"Error during {LLM_PROVIDER.upper()} streaming: {e}")
            yield f"⚠️ Error during streaming: {e}"


async def embed(text: str) -> List[float]:
    """
    Generate a text embedding vector.

    Neither Groq nor Grok provide an embedding endpoint, so we use
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
    Send a simple message to verify the API key and provider are working.

    Returns:
        True if the connection succeeds, False otherwise.
    """
    try:
        response = client.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
        )
        _ = response.choices[0].message.content
        return True
    except Exception as e:
        logger.error(f"{LLM_PROVIDER.upper()} connection test failed: {e}")
        return False


# -------------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    async def _run_test():
        print(f"Testing {LLM_PROVIDER.upper()} API connection (model: {ACTIVE_MODEL})...")
        success = await test_connection()
        if success:
            print(f"{LLM_PROVIDER.upper()} connection successful ✅")
        else:
            print(
                f"{LLM_PROVIDER.upper()} connection failed ❌ — "
                f"check your {'GROK' if LLM_PROVIDER == 'grok' else 'GROQ'}_API_KEY in .env"
            )

    asyncio.run(_run_test())
