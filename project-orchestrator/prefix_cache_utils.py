"""
DeepSeek Prefix Cache Optimization Utilities

DeepSeek V4 automatically caches prompt prefixes at block boundaries.
On cache hits, subsequent requests enjoy an 80-92% token discount on
repeated prefix portions (billed at only 10-20% of the original price).

Phase 0's serial document generation chain benefits naturally:
- Agent N's system prompt is identical to Agent N-1's
- Agent N loads prerequisite docs that heavily overlap with Agent N-1's
- Only the newest document and task description change each iteration

Usage:
    from prefix_cache_utils import build_cache_optimized_messages

    messages = build_cache_optimized_messages(
        system_prompt=prompt,
        context_docs=docs,      # prerequisite docs (in fixed order)
        task_description=task,  # changing task description (placed last)
    )

Message structure design (maximizes prefix cache hits):
    [system prompt] → [doc1] → [doc2] → ... → [docN] → [task description]
    |________________ cached prefix ________________|  |__ varying suffix __|
"""

from typing import Sequence


def build_cache_optimized_messages(
    system_prompt: str,
    context_docs: Sequence[str] = (),
    task_description: str = "",
    user_input: str = "",
) -> list[dict]:
    """Build prefix-cache-optimized message list.

    Principles:
    1. System prompt first (always identical → always hits cache)
    2. Prerequisite docs in fixed order (downstream Phase 0 agents more likely to hit cache)
    3. Task description / user input last (varying part, doesn't break prefix)

    Args:
        system_prompt: Agent system prompt string
        context_docs: List of prerequisite document contents (e.g. ["content of docs/PRD.md", ...])
        task_description: Specific task description
        user_input: Raw user input (provide one or both with task_description)

    Returns:
        List of messages in OpenAI API format
    """
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Prerequisite docs: each doc as a separate user message for clear block boundaries
    for i, doc_content in enumerate(context_docs):
        if doc_content:
            messages.append({
                "role": "user",
                "content": doc_content,
            })

    # Task description last — final segment before the cached suffix variation
    if task_description or user_input:
        content_parts = []
        if task_description:
            content_parts.append(f"## Task Requirements\n{task_description}")
        if user_input:
            content_parts.append(f"## User Input\n{user_input}")
        messages.append({
            "role": "user",
            "content": "\n\n".join(content_parts),
        })

    return messages


def estimate_cache_savings(
    num_agents: int = 6,
    avg_prompt_tokens: int = 30_000,
    cache_hit_ratio: float = 0.7,
) -> dict:
    """Estimate cost savings from prefix caching in the Phase 0 serial chain.

    Args:
        num_agents: Number of agents in Phase 0 (default 6)
        avg_prompt_tokens: Average prompt tokens per agent
        cache_hit_ratio: Estimated cache hit ratio (0-1)

    Returns:
        dict with cost estimates
    """
    # DeepSeek V4 Pro input price: $1.74 / 1M tokens
    # Cache hit price: 10-20% of original (using 15%)
    INPUT_PRICE_PER_M = 1.74
    CACHE_HIT_PRICE_PER_M = INPUT_PRICE_PER_M * 0.15

    total_prompt_tokens = num_agents * avg_prompt_tokens
    cached_tokens = total_prompt_tokens * cache_hit_ratio
    uncached_tokens = total_prompt_tokens - cached_tokens

    cost_without_cache = total_prompt_tokens / 1_000_000 * INPUT_PRICE_PER_M
    cost_with_cache = (
        uncached_tokens / 1_000_000 * INPUT_PRICE_PER_M
        + cached_tokens / 1_000_000 * CACHE_HIT_PRICE_PER_M
    )

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "cached_tokens": cached_tokens,
        "uncached_tokens": uncached_tokens,
        "cost_without_cache": round(cost_without_cache, 4),
        "cost_with_cache": round(cost_with_cache, 4),
        "savings_pct": round((1 - cost_with_cache / cost_without_cache) * 100, 1),
    }


if __name__ == "__main__":
    result = estimate_cache_savings()
    print("Phase 0 Prefix Cache Cost Estimation")
    print(f"  Total Prompt Tokens:   {result['total_prompt_tokens']:,}")
    print(f"  Cache Hit Tokens:      {result['cached_tokens']:,}")
    print(f"  Uncached Tokens:       {result['uncached_tokens']:,}")
    print(f"  Cost without Cache:    ${result['cost_without_cache']}")
    print(f"  Cost with Cache:       ${result['cost_with_cache']}")
    print(f"  Savings:               {result['savings_pct']}%")
