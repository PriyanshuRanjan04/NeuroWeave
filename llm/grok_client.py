import os
import sys
import logging
from typing import List, Dict, Any, AsyncGenerator
from openai import AsyncOpenAI

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROK_API_KEY

logger = logging.getLogger(__name__)

# Initialize xAI client using OpenAI SDK compatibility layer
client = None
if GROK_API_KEY:
    try:
        client = AsyncOpenAI(
            api_key=GROK_API_KEY,
            base_url="https://api.x.ai/v1",
        )
        logger.info("xAI (Grok) client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Grok client: {str(e)}")
else:
    logger.warning("GROK_API_KEY is missing. LLM operations will fail.")

# Use grok-3 or grok-beta, and eventually grok-4 when generally available
MODEL_NAME = "grok-3" 

async def chat(messages: List[Dict[str, str]], system_prompt: str = "") -> str:
    """Standard chat completion using Grok API."""
    if not client:
        return "Error: xAI client not initialized. Please check GROK_API_KEY."
    
    formatted_messages = []
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})
    formatted_messages.extend(messages)
    
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=formatted_messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Grok chat error: {str(e)}")
        return f"Error connecting to Grok API: {str(e)}"

async def stream_chat(messages: List[Dict[str, str]], system_prompt: str = "") -> AsyncGenerator[str, None]:
    """Streaming chat completion using Grok API."""
    if not client:
        yield "Error: xAI client not initialized. Please check GROK_API_KEY."
        return
        
    formatted_messages = []
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})
    formatted_messages.extend(messages)
    
    try:
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=formatted_messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"Grok stream chat error: {str(e)}")
        yield f"Error connecting to Grok API streaming endpoint: {str(e)}"

async def embed(text: str) -> List[float]:
    """
    Generate embeddings for vector storage.
    Note: xAI's native embedding endpoint or an OpenAI fallback model 'text-embedding-3-small' 
    if strictly relying on OpenAI compat logic with an external embedding provider.
    For this codebase, we execute assuming the /v1/embeddings route works.
    """
    if not client:
        return []
        
    try:
        response = await client.embeddings.create(
            input=text,
            model="text-embedding-3-small" # Generic 1536 dim fallback model string, you might adjust if xAI requires different wording.
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Grok embed error: {str(e)}")
        # In case API lacks embedding endpoints, returning empty list allows graceful failing upstream
        return []
