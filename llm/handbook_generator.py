import os
import sys
import logging
import asyncio
from typing import List, Dict, Any
import json
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.grok_client import chat
from rag.retriever import query

logger = logging.getLogger(__name__)

async def generate_outline(topic: str, context: str) -> List[Dict[str, Any]]:
    """Returns a structured plan for the handbook."""
    system_prompt = (
        "You are an expert technical writer and outline planner. "
        "Create a detailed structural outline for a comprehensive 20,000-word handbook on the given topic. "
        "The outline MUST contain exactly 10 to 12 major sections. "
        "For each section, provide a title, a brief description, and a target word count between 1500 and 2500 words. "
        "Output ONLY a valid JSON array of objects, where each object has keys: 'title', 'description', 'target_words'. "
        "Do NOT include any markdown formatting like ```json ...``` just output the raw JSON array."
    )
    
    user_prompt = f"Topic: {topic}\n\nAvailable Context:\n{context}\n\nGenerate the JSON outline."
    
    try:
        response = await chat([{"role": "user", "content": user_prompt}], system_prompt)
        
        # Clean up possible markdown or extra text
        json_str = response.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
            
        outline = json.loads(json_str.strip())
        logger.info(f"Generated outline with {len(outline)} sections.")
        return outline
    except Exception as e:
        logger.error(f"Failed to generate outline: {str(e)}\nRaw Response: {response if 'response' in locals() else 'None'}")
        # Return a fallback outline
        return [
            {"title": f"Section {i+1}", "description": "Auto-fallback section", "target_words": 1500} 
            for i in range(12)
        ]

async def generate_section(section_title: str, target_words: int, context: str, previous_summary: str) -> str:
    """Generate an individual section focusing on depth and detail."""
    system_prompt = (
        "You are an expert technical writer contributing a major section to a comprehensive handbook. "
        f"Your target length for this section is approximately {target_words} words. "
        "Write in a professional, authoritative tone. "
        "Be thorough and detailed, using the provided context. Exploit the provided context fully. "
        "Use subheadings, bullet points, and explanatory paragraphs to meet the length requirements. "
        "If the context is insufficient, extrapolate logically based on the core topic. "
        "Do not write the conclusion of the book, just complete this section."
    )
    
    user_prompt = (
        f"Section Title: {section_title}\n\n"
        f"Previous Section Summary Context (for flow and continuity):\n{previous_summary if previous_summary else 'This is the first section.'}\n\n"
        f"Source Context:\n{context}\n\n"
        f"Please write the complete section '{section_title}'."
    )
    
    try:
        response = await chat([{"role": "user", "content": user_prompt}], system_prompt)
        return response
    except Exception as e:
        logger.error(f"Error generating section '{section_title}': {str(e)}")
        return f"Error generating section {section_title}: {str(e)}"

def compile_handbook(title: str, sections_content: List[str], section_titles: List[str], exec_summary: str, conclusion: str, references: str) -> str:
    """Combine all sections with proper formatting."""
    handbook = []
    
    # Title
    handbook.append(f"# {title.upper()}\n\n")
    
    # Executive Summary
    handbook.append("## Executive Summary\n")
    handbook.append(f"{exec_summary}\n\n")
    
    # TOC
    handbook.append("## Table of Contents\n")
    for i, t in enumerate(section_titles):
        handbook.append(f"{i+1}. {t}\n")
    handbook.append("\n")
    
    # Sections
    for title_txt, content in zip(section_titles, sections_content):
        handbook.append(f"## {title_txt}\n")
        handbook.append(f"{content}\n\n")
        
    # Conclusion
    handbook.append("## Conclusion\n")
    handbook.append(f"{conclusion}\n\n")
    
    # References
    if references:
        handbook.append("## References\n")
        handbook.append(f"{references}\n\n")
        
    return "".join(handbook)

async def generate_handbook(topic: str, retrieved_context: str, update_fn=None) -> str:
    """Main orchestrator for the 20k word handbook generation."""
    if update_fn:
        update_fn("Planning: Generating detailed section outline...")
        
    # 1. Plan
    outline = await generate_outline(topic, retrieved_context)
    
    if update_fn:
        update_fn(f"Planning complete. Found {len(outline)} sections. Generating Executive Summary...")
        
    # 2. Exec Summary
    exec_summary_prompt = "You are an expert writer. Write a 500-word executive summary for a handbook."
    exec_summary = await chat(
        [{"role": "user", "content": f"Topic: {topic}\n\nContext:\n{retrieved_context}"}],
        system_prompt=exec_summary_prompt
    )
    
    # 3. Write Sections Iteratively
    sections_content = []
    section_titles = []
    previous_summary = ""
    
    for i, section in enumerate(outline):
        title = section.get('title', f"Section {i+1}")
        target = section.get('target_words', 1500)
        
        section_titles.append(title)
        
        if update_fn:
            update_fn(f"Writing Section {i+1}/{len(outline)}: {title} (Target: ~{target} words)...")
            
        # Get section specific context
        section_specific_context = await query(f"{topic} {title}", mode="hybrid")
        combined_context = f"Global Context:\n{retrieved_context}\n\nSection Specific Context:\n{section_specific_context}"
        
        content = await generate_section(title, target, combined_context, previous_summary)
        sections_content.append(content)
        
        # Briefly summarize this section to pass to the next for continuity
        # Instead of full LLM summary, basic truncation works for context sliding
        if len(content) > 600:
            previous_summary = "Previous section covered: " + content[:600] + "..."
        else:
            previous_summary = "Previous section covered: " + content
            
    if update_fn:
        update_fn("Generating Conclusion and formulating references...")
        
    conclusion_prompt = "You are an expert writer. Write a 500-word conclusion summarizing the key points of the handbook."
    conclusion = await chat(
        [{"role": "user", "content": f"Topic: {topic}"}],
        system_prompt=conclusion_prompt
    )
    
    references = "1. Uploaded Source PDF Documents via NeuroWeave.\n2. Generated via RAG Pipeline (LightRAG & xAI)."
    
    if update_fn:
        update_fn("Compiling final handbook...")
        
    # 4. Compile
    final_book = compile_handbook(topic, sections_content, section_titles, exec_summary, conclusion, references)
    
    if update_fn:
        update_fn("Handbook generation complete!")
        
    return final_book
