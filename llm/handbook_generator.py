import os
import sys
import logging
from typing import List, Optional, Callable, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Prompt Templates (exact AgentWrite plan.txt / write.txt)
# -----------------------------------------------------------------------

PLAN_PROMPT = (
    "I need you to help me break down the following long-form writing instruction "
    "into multiple subtasks. Each subtask will guide the writing of one paragraph "
    "in the essay, and should include the main points and word count requirements "
    "for that paragraph.\n"
    "The writing instruction is as follows:\n"
    "$INST$\n"
    "Please break it down in the following format, with each subtask taking up one line:\n"
    "Paragraph 1 - Main Point: [Describe the main point of the paragraph, in detail] "
    "- Word Count: [Word count requirement, e.g., 400 words]\n"
    "Paragraph 2 - Main Point: [Describe the main point of the paragraph, in detail] "
    "- Word Count: [word count requirement, e.g. 1000 words].\n"
    "...\n"
    "Make sure that each subtask is clear and specific, and that all subtasks cover "
    "the entire content of the writing instruction. Do not split the subtasks too "
    "finely; each subtask's paragraph should be no less than 200 words and no more "
    "than 1000 words. Do not output any other content."
)

WRITE_PROMPT = (
    "You are an excellent writing assistant. I will give you an original writing "
    "instruction and my planned writing steps. I will also provide you with the "
    "text I have already written. Please help me continue writing the next paragraph "
    "based on the writing instruction, writing steps, and the already written text.\n"
    "Writing instruction:\n"
    "$INST$\n"
    "Writing steps:\n"
    "$PLAN$\n"
    "Already written text:\n"
    "$TEXT$\n"
    "Please integrate the original writing instruction, writing steps, and the "
    "already written text, and now continue writing $STEP$. If needed, you can "
    "add a small subtitle at the beginning. Remember to only output the paragraph "
    "you write, without repeating the already written text. As this is an ongoing "
    "work, omit open-ended conclusions or other rhetorical hooks."
)


# -----------------------------------------------------------------------
# Function 1: generate_outline
# -----------------------------------------------------------------------

async def generate_outline(topic: str, context: str) -> List[str]:
    """
    PLAN step (plan.py equivalent).

    Calls Groq to break the handbook topic into numbered paragraph steps,
    each with a main point and target word count.

    Returns:
        List of step strings, e.g.:
        ["Paragraph 1 - Main Point: ... - Word Count: 500 words", ...]
    """
    from llm.grok_client import chat

    inst = topic + "\n\nContext from uploaded documents:\n" + context
    prompt = PLAN_PROMPT.replace("$INST$", inst)

    system_prompt = (
        "You are an excellent writing assistant helping plan long-form handbooks. "
        "Follow the format instructions exactly. Do not output anything else."
    )

    print("Generating outline...")
    try:
        response = await chat(
            [{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=0.7
        )
        if not response or response.strip() == "":
            # Retry once
            print("  Empty outline response, retrying...")
            response = await chat(
                [{"role": "user", "content": prompt}],
                system_prompt=system_prompt,
                temperature=0.7
            )

        if not response or response.strip() == "":
            raise Exception("Failed to generate outline — Groq returned empty response.")

        # Parse: one paragraph step per non-empty line
        steps = [line.strip() for line in response.splitlines() if line.strip()]
        # Keep only lines that look like plan steps (contain "Paragraph" or "-")
        steps = [s for s in steps if s.lower().startswith("paragraph") or " - " in s]

        if not steps:
            raise Exception("Failed to generate outline — no valid steps parsed from response.")

        print(f"  -> Outline ready: {len(steps)} sections planned")
        return steps

    except Exception as e:
        logger.error(f"generate_outline failed: {e}")
        raise Exception(f"Failed to generate outline: {e}")


# -----------------------------------------------------------------------
# Function 2: generate_section
# -----------------------------------------------------------------------

async def generate_section(
    inst: str,
    plan: str,
    text_so_far: str,
    current_step: str
) -> str:
    """
    WRITE step (write.py get_pred equivalent).

    Generates one paragraph by filling in the write.txt template and
    calling Groq. Retries once on empty response.

    Returns:
        Generated paragraph as a string (empty string if both attempts fail).
    """
    from llm.grok_client import chat

    prompt = (
        WRITE_PROMPT
        .replace("$INST$", inst)
        .replace("$PLAN$", plan)
        .replace("$TEXT$", text_so_far if text_so_far else "(none yet — this is the first paragraph)")
        .replace("$STEP$", current_step)
    )

    system_prompt = (
        "You are an excellent writing assistant. Write detailed, comprehensive "
        "paragraphs. Only output the paragraph content, nothing else."
    )

    try:
        response = await chat(
            [{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=0.7
        )

        if not response or response.strip() == "":
            print(f"    Empty response for step, retrying...")
            response = await chat(
                [{"role": "user", "content": prompt}],
                system_prompt=system_prompt,
                temperature=0.7
            )

        return response.strip() if response else ""

    except Exception as e:
        logger.error(f"generate_section failed for step '{current_step[:50]}...': {e}")
        return ""


# -----------------------------------------------------------------------
# Function 3: compile_handbook
# -----------------------------------------------------------------------

def compile_handbook(
    topic: str,
    steps: List[str],
    responses: List[str],
    source_filenames: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Assembles the final handbook document from parts.

    Returns:
        { "content": markdown_string, "word_count": int }
    """
    parts: List[str] = []

    # 1. Title
    parts.append(f"# {topic} — A Comprehensive Handbook\n\n")

    # 2. Table of Contents
    parts.append("## Table of Contents\n\n")
    for i, step in enumerate(steps):
        # Extract a short label from the step string
        label = step.split("Main Point:")[-1].split("- Word Count:")[0].strip()
        if not label:
            label = f"Section {i + 1}"
        parts.append(f"{i + 1}. {label}\n")
    parts.append("\n")

    # 3. Body — all paragraphs joined
    parts.append("---\n\n")
    for paragraph in responses:
        parts.append(paragraph)
        parts.append("\n\n")

    # 4. References
    parts.append("---\n\n## References\n\n")
    parts.append("1. Uploaded Source PDF Documents via NeuroWeave.\n")
    if source_filenames:
        for i, fname in enumerate(source_filenames, start=2):
            parts.append(f"{i}. {fname}\n")
    parts.append("3. Generated via RAG Pipeline (LightRAG & Groq llama-3.3-70b-versatile).\n")

    content = "".join(parts)
    word_count = len(content.split())

    return {"content": content, "word_count": word_count}


# -----------------------------------------------------------------------
# Function 4: generate_handbook  (main orchestrator)
# -----------------------------------------------------------------------

async def generate_handbook(
    topic: str,
    retrieved_context: str,
    source_filenames: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    # Legacy alias accepted for backwards compat with app.py
    update_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Main orchestrator — LongWriter AgentWrite technique.

    Flow:
      1. Build instruction string (inst)
      2. PLAN  — generate_outline()
      3. WRITE — generate_section() for each step, accumulating text
      4. COMPILE — compile_handbook()
    """
    # Support old update_fn kwarg from app.py
    cb = progress_callback or update_fn

    try:
        # Step 1: Build instruction
        # Guard: LightRAG can return None if query fails internally
        retrieved_context = retrieved_context or ""
        inst = topic + "\n\nContext from uploaded documents:\n" + retrieved_context

        # Step 2: Generate plan
        if cb:
            cb("Planning: generating detailed section outline…")

        steps = await generate_outline(topic, retrieved_context)

        if cb:
            cb(f"Outline ready — {len(steps)} sections planned. Starting writing…")

        # Step 3: Write each section (exact write.py loop)
        text = ""          # grows after every paragraph
        responses: List[str] = []

        for i, step in enumerate(steps):
            print(f"Writing section {i + 1}/{len(steps)}: {step[:60]}...")
            if cb:
                cb(f"Writing section {i + 1} of {len(steps)}: {step[:50]}…")

            response = await generate_section(inst, "\n".join(steps), text, step)

            if response == "":
                print(f"  Empty response for step {i + 1}, skipping…")
                continue

            responses.append(response)
            text += response + "\n\n"  # exact same accumulation as write.py

        if not responses:
            raise Exception("No sections were generated — all LLM responses were empty.")

        # Step 4: Compile
        print("Compiling final handbook…")
        if cb:
            cb("Compiling final handbook…")

        result = compile_handbook(topic, steps, responses, source_filenames)

        if result["word_count"] < 20000:
            logger.warning(
                f"Handbook word count ({result['word_count']:,}) is below the 20,000-word target. "
                "Consider uploading more source material or broadening the topic."
            )

        if cb:
            cb(f"Done! Handbook generated: {result['word_count']:,} words across {len(responses)} sections.")

        return {
            "content": result["content"],
            "word_count": result["word_count"],
            "sections_count": len(responses),
            "topic": topic,
        }

    except Exception as e:
        logger.error(f"generate_handbook failed: {e}")
        raise


# -----------------------------------------------------------------------
# Standalone test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio

    print("Handbook Generator ready")
    print("LongWriter AgentWrite technique loaded")
    print("Plan prompt template: LOADED")
    print("Write prompt template: LOADED")
    print()

    async def _dry_run():
        sample_steps = await generate_outline(
            "Introduction to Retrieval Augmented Generation",
            "Sample context about RAG systems"
        )
        print(f"Sample outline generated: {len(sample_steps)} sections")
        for i, step in enumerate(sample_steps):
            print(f"  {i + 1}. {step[:80]}...")

    asyncio.run(_dry_run())
