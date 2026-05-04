"""
DeepSeek 前缀缓存优化工具

DeepSeek V4 自动在块边界对 prompt 前缀进行缓存。缓存命中后，后续请求
中相同的前缀部分享受 80-92% 的 token 折扣（仅计费缓存命中 token 的 10-20%）。

Phase 0 的串行文档生成链路天然受益于前缀缓存：
- Agent N 的 system prompt 与 Agent N-1 完全相同
- Agent N 加载的前置文档与 Agent N-1 大部分重叠
- 只有最新的文档和任务描述是变化的部分

使用方法：
    from prefix_cache_utils import build_cache_optimized_messages

    messages = build_cache_optimized_messages(
        system_prompt=prompt,
        context_docs=docs,      # 前置文档列表（按固定顺序）
        task_description=task,  # 变化的任务描述（放在最后）
    )

消息结构设计（最大化前缀缓存命中）：
    [system prompt] → [文档1] → [文档2] → ... → [文档N] → [任务描述]
    |______________ 固定前缀（被缓存）______________|  |__ 变化后缀 __|
"""

from typing import Sequence


def build_cache_optimized_messages(
    system_prompt: str,
    context_docs: Sequence[str] = (),
    task_description: str = "",
    user_input: str = "",
) -> list[dict]:
    """构建前缀缓存优化的消息列表。

    原则：
    1. system prompt 放在最前（始终相同 → 始终命中缓存）
    2. 前置文档按固定顺序排列（Phase 0 下游文档更可能命中缓存）
    3. 任务描述/用户输入放在最后（变化部分，不破坏前缀）

    Args:
        system_prompt: Agent 系统提示词
        context_docs: 前置文档内容列表（例如 ["docs/PRD.md 的内容", ...]）
        task_description: 具体任务描述
        user_input: 用户原始输入（与 task_description 二选一或都提供）

    Returns:
        OpenAI API 格式的 messages 列表
    """
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # 前置文档：每个文档作为独立的 user message，确保块边界清晰
    for i, doc_content in enumerate(context_docs):
        if doc_content:
            messages.append({
                "role": "user",
                "content": doc_content,
            })

    # 任务描述放在最后——缓存后缀之前的最末一段
    if task_description or user_input:
        content_parts = []
        if task_description:
            content_parts.append(f"## 任务要求\n{task_description}")
        if user_input:
            content_parts.append(f"## 用户输入\n{user_input}")
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
    """估算 Phase 0 串行链路中前缀缓存节省的成本。

    Args:
        num_agents: Phase 0 的 Agent 数量（默认 6）
        avg_prompt_tokens: 每个 Agent 的平均 prompt token 数
        cache_hit_ratio: 估计的缓存命中比例

    Returns:
        包含成本估算的 dict
    """
    # DeepSeek V4 Pro 输入价格：$1.74 / 1M tokens
    # 缓存命中价格：原价 10-20%（取 15%）
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
    print("Phase 0 前缀缓存成本估算")
    print(f"  总 Prompt Tokens:   {result['total_prompt_tokens']:,}")
    print(f"  命中缓存 Tokens:     {result['cached_tokens']:,}")
    print(f"  未缓存 Tokens:       {result['uncached_tokens']:,}")
    print(f"  无缓存成本:          ${result['cost_without_cache']}")
    print(f"  有缓存成本:          ${result['cost_with_cache']}")
    print(f"  节省:                {result['savings_pct']}%")
