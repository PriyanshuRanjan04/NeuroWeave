import os
import sys
import logging
import asyncio
from typing import Dict, Any, List

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LIGHTRAG_WORKING_DIR, GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

logger = logging.getLogger(__name__)

if not os.path.exists(LIGHTRAG_WORKING_DIR):
    os.makedirs(LIGHTRAG_WORKING_DIR)


# -------------------------------------------------------------------
# LightRAG LLM function — uses Groq via OpenAI-compatible interface
# -------------------------------------------------------------------
async def llm_model_func(prompt, **kwargs) -> str:
    """LightRAG's internal LLM calls are routed to Groq."""
    return await openai_complete_if_cache(
        GROQ_MODEL,
        prompt,
        system_prompt=kwargs.get("system_prompt", ""),
        history_messages=kwargs.get("history_messages", []),
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        **kwargs
    )


# -------------------------------------------------------------------
# LightRAG Embedding function — uses sentence-transformers via grok_client
# -------------------------------------------------------------------
async def embedding_func(texts: List[str]):
    """Delegate embedding to our grok_client (sentence-transformers fallback)."""
    from llm.grok_client import embed
    import numpy as np

    results = []
    for text in texts:
        emb = await embed(text)
        if not emb:
            emb = [0.0] * 1536  # safe fallback
        results.append(emb)
    return np.array(results)


# -------------------------------------------------------------------
# Initialize LightRAG
# -------------------------------------------------------------------
try:
    rag = LightRAG(
        working_dir=LIGHTRAG_WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            func=embedding_func,
            embedding_dim=1536,
            max_token_size=8192
        ),
    )
    logger.info("LightRAG instance initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize LightRAG: {e}")
    rag = None


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

async def index_document(text: str, filename: str) -> bool:
    """Build knowledge graph from PDF text."""
    if not rag:
        logger.error("LightRAG not initialized. Cannot index document.")
        return False

    try:
        def insert_sync():
            rag.insert(text)

        await asyncio.to_thread(insert_sync)
        logger.info(f"Successfully indexed document to knowledge graph: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error indexing document '{filename}' in LightRAG: {e}")
        return False


def get_graph_stats() -> Dict[str, Any]:
    """Return number of nodes/edges in knowledge graph."""
    if not rag:
        return {"nodes": 0, "edges": 0, "status": "Not initialized"}

    try:
        graph_file = os.path.join(LIGHTRAG_WORKING_DIR, "graph_chunk_entity_relation.graphml")
        stats: Dict[str, Any] = {"status": "Active"}

        if os.path.exists(graph_file):
            import networkx as nx
            try:
                G = nx.read_graphml(graph_file)
                stats["nodes"] = G.number_of_nodes()
                stats["edges"] = G.number_of_edges()
            except Exception as read_err:
                stats["error"] = f"Could not parse graphml: {read_err}"
        else:
            stats["nodes"] = 0
            stats["edges"] = 0

        return stats
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        return {"error": str(e)}
