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
                "DEEPSEEK_API_KEY 未设置。请在 .env 文件中设置 DEEPSEEK_API_KEY"
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
        raise FileNotFoundError(f"Agent 提示词文件不存在: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def load_context_docs(doc_paths: list[str]) -> str:
    """加载上下文文档，合并为字符串

    文档按固定顺序排列（system prompt → 前置文档 → 任务描述），
    以便 DeepSeek 自动前缀缓存在多个 Agent 之间共享文档前缀。
    """
    parts = []
    for path in doc_paths:
        p = Path(path)
        if p.exists():
            content = p.read_text(encoding="utf-8")
            parts.append(f"\n\n---\n# 文档：{path}\n\n{content}")
        else:
            parts.append(f"\n\n---\n# 文档：{path}\n\n[文件不存在，请先完成前置任务]")
    return "".join(parts)


def _build_extra_body(config: dict) -> dict:
    """根据 Agent 配置构建 thinking mode extra_body"""
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

    user_message = f"""## 任务要求
{task_description}

## 可用文档上下文
{context if context else "（无前置文档）"}
"""

    thinking_label = f" + {config['thinking_mode']}" if config.get("thinking_mode") else ""
    console.print(
        Panel(
            f"[bold cyan]🤖 {config['display_name']} 启动中...[/bold cyan]\n"
            f"模型：{config['model']}{thinking_label}",
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
    """将 Agent 输出保存到对应文档路径，返回保存路径"""
    config = AGENT_REGISTRY[agent_name]
    output_path = config.get("output_doc")
    if not output_path:
        return None

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    console.print(f"[green]✅ 文档已保存：{output_path}[/green]")
    return output_path
