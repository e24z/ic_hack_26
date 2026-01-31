"""Paper summarization using LLM."""

import asyncio
import logging

from .semantic_scholar import PaperDetails
from .llm import OpenRouterAdapter, LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a research assistant. Summarize the given academic paper concisely.
Focus on: main contribution, methodology, key findings, and limitations.
Be precise and technical. Output 3-5 sentences."""


async def summarize_paper(
    paper: PaperDetails,
    provider: LLMProvider | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.3,
) -> str:
    """
    Summarize a single paper using LLM.

    Args:
        paper: Paper with full_text (or falls back to abstract)
        provider: LLM provider (defaults to OpenRouterAdapter)
        system_prompt: System prompt for summarization
        temperature: LLM temperature

    Returns:
        Summary string
    """
    # Use full text if available, otherwise fall back to abstract
    content = paper.full_text or paper.abstract or "No content available"

    # Truncate if too long (most models have context limits)
    max_chars = 30000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[Content truncated...]"

    prompt = f"""Paper: {paper.title}
Authors: {', '.join(a.name or 'Unknown' for a in paper.authors[:5])}
Year: {paper.year}

Content:
{content}

Provide a concise summary."""

    if provider:
        return await provider.complete(prompt, system_prompt, temperature)
    else:
        async with OpenRouterAdapter() as llm:
            return await llm.complete(prompt, system_prompt, temperature)


async def summarize_papers(
    papers: list[PaperDetails],
    provider: LLMProvider | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.3,
    parallel: bool = True,
) -> list[str]:
    """
    Summarize multiple papers, returning one summary per paper.

    Args:
        papers: List of papers with full_text (or abstracts as fallback)
        provider: LLM provider (defaults to OpenRouterAdapter)
        system_prompt: System prompt for summarization
        temperature: LLM temperature
        parallel: If True, summarize papers concurrently

    Returns:
        List of summaries, one per paper (same order as input)
    """
    if not papers:
        return []

    if provider:
        # Use provided provider
        if parallel:
            tasks = [
                summarize_paper(paper, provider, system_prompt, temperature)
                for paper in papers
            ]
            return await asyncio.gather(*tasks)
        else:
            return [
                await summarize_paper(paper, provider, system_prompt, temperature)
                for paper in papers
            ]
    else:
        # Create a single provider instance for all papers
        async with OpenRouterAdapter() as llm:
            if parallel:
                tasks = [
                    summarize_paper(paper, llm, system_prompt, temperature)
                    for paper in papers
                ]
                return await asyncio.gather(*tasks)
            else:
                return [
                    await summarize_paper(paper, llm, system_prompt, temperature)
                    for paper in papers
                ]
