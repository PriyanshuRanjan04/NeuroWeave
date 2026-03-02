import os
import sys
import logging
import asyncio
from lightrag import QueryParam

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.lightrag_client import rag

logger = logging.getLogger(__name__)

async def query(question: str, mode: str = "hybrid") -> str:
    """
    Retrieve relevant context using LightRAG's dual querying capability.
    Mode can be 'naive' (vector), 'local' (graph), 'global' (graph), or 'hybrid' (vector+graph).
    """
    if not rag:
        return "Error: LightRAG system is not initialized."
        
    try:
        def query_sync():
            return rag.query(
                question,
                param=QueryParam(mode=mode)
            )
            
        response = await asyncio.to_thread(query_sync)
        return response
    except Exception as e:
        logger.error(f"Error querying LightRAG: {str(e)}")
        return f"Retrieval failed due to an error: {str(e)}"
