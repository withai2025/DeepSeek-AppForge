from pathlib import Path
from openai import OpenAI
from config import AGENT_REGISTRY, DEEPSEEK_API_KEY
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

_client = None


def _get_client():
    global _client
    if _client is None:
        if not DEEPSEEK_API_KEY:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. Please set DEEPSEEK_API_KEY in .env file"
            )
        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
    return _client


def load_agent_prompt(agent_name: str) -> str:
    """加载子 Agent 的系统提示词"""
    config = AGENT_REGISTRY[agent_name]
    prompt_path = Path(config["prompt_file"])
    if not prompt_path.exists():
        raise FileNotFoundError(f"Agent prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def load_context_docs(doc_paths: list[str]) -> str:
    """Load context documents and merge into a single string.

    Documents are ordered consistently (system prompt → prerequisite docs → task description)
    so that DeepSeek auto prefix caching shares document prefixes across agents.
    """
    parts = []
    for path in doc_paths:
        p = Path(path)
        if p.exists():
            content = p.read_text(encoding="utf-8")
            parts.append(f"\n\n---\n# Document: {path}\n\n{content}")
        else:
            parts.append(f"\n\n---\n# Document: {path}\n\n[File does not exist — complete prerequisite task first]")
    return "".join(parts)


def _build_extra_body(config: dict) -> dict:
    """Build thinking mode extra_body from agent config"""
    thinking_mode = config.get("thinking_mode")
    if thinking_mode:
        return {"thinking": {"type": "enabled"}}
    return {}


def run_worker(
    agent_name: str,
    task_description: str,
    extra_context: str = "",
    stream: bool = True,
) -> str:
    """执行子 Agent，返回完整输出内容"""
    config = AGENT_REGISTRY[agent_name]
    system_prompt = load_agent_prompt(agent_name)

    # 组装上下文文档
    context = load_context_docs(config.get("requires_docs", []))
    if extra_context:
        context += f"\n\n---\n# 额外上下文\n\n{extra_context}"

    user_message = f"""## Task Requirements
{task_description}

## Available Document Context
{context if context else "(No prerequisite documents)"}
"""

    thinking_label = f" + {config['thinking_mode']}" if config.get("thinking_mode") else ""
    console.print(
        Panel(
            f"[bold cyan]🤖 {config['display_name']} starting...[/bold cyan]\n"
            f"Model: {config['model']}{thinking_label}",
            border_style="cyan",
        )
    )

    full_response = ""
    extra_body = _build_extra_body(config)

    if stream:
        stream_obj = _get_client().chat.completions.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=True,
            extra_body=extra_body,
        )
        for chunk in stream_obj:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                full_response += delta.content
        print()  # 换行
    else:
        response = _get_client().chat.completions.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            extra_body=extra_body,
        )
        full_response = response.choices[0].message.content
        console.print(Markdown(full_response))

    return full_response


def save_agent_output(agent_name: str, content: str) -> str | None:
    """Save agent output to output document path, return saved path"""
    config = AGENT_REGISTRY[agent_name]
    output_path = config.get("output_doc")
    if not output_path:
        return None

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    console.print(f"[green]✅ Document saved: {output_path}[/green]")
    return output_path
