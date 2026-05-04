"""
跨文档一致性检查器

利用 DeepSeek V4 100 万 token 上下文窗口，将 Phase 0 全部 6 份文档
一次性加载到单个 API 调用中进行交叉审查，发现字段名不一致、逻辑矛盾、
接口与 Schema 不匹配等问题。

这是 Claude 200K 上下文窗口无法实现的功能。
"""

from pathlib import Path
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import DEEPSEEK_API_KEY

console = Console()

CHECKER_SYSTEM_PROMPT = """# 角色定义

你是一位资深 QA 审查专家，专精跨文档一致性检查。

# 任务

你将收到一整套 APP 开发文档（PRD、技术架构、编码规范、数据库 Schema、API 契约、任务书），请逐项审查以下维度：

## 审查维度

1. **字段名一致性**：同一实体在不同文档中的字段名是否一致？
   - 例如：PRD 中的"用户昵称"在 Schema 中是 `nickname` 还是 `display_name`？在 API 中是 `nickName` 还是 `nickname`？

2. **接口与 Schema 匹配**：API 契约中定义的请求/响应字段是否与 DB Schema 中的列一一对应？
   - 检查每个 endpoint 的 request body / response body 字段在数据库表中是否有对应列

3. **功能完整性**：PRD 中每个 P0 功能是否有对应的 API 接口？每个页面是否有对应的前端任务？

4. **编码规范一致性**：Schema 和 API 中的命名是否符合编码规范的命名约定？
   - 例如：规范要求 snake_case，Schema 中是否出现了 camelCase？

5. **数据类型一致性**：同一字段在不同文档中的类型定义是否一致？
   - 例如：PRD 说"年龄是整数"，Schema 中是 INTEGER，API 中是 string？

## 输出格式

请按照以下结构输出审查报告：

### 🔴 必须修复（阻塞问题）
- 具体问题描述
- 涉及文档及位置
- 修复建议

### 🟡 建议修复（潜在风险）
- 具体问题描述
- 涉及文档及位置
- 修复建议

### 🟢 可忽略（注释说明）
- 观察到的差异
- 为什么可忽略

### 📊 评分
- 字段一致性：X/10
- 接口-Schema 匹配度：X/10
- 功能完整性：X/10
- 编码规范一致性：X/10
- **综合评分：X/10**

## 关键原则
- 严格对照文档原文，不凭记忆推测
- 标注具体文档名和章节位置
- 修复建议必须具体到字段名或代码行
"""


def load_all_docs(docs_dir: str = "docs") -> dict[str, str]:
    """加载 Phase 0 全部产出文档"""
    doc_files = {
        "PRD.md": "产品需求文档",
        "TECH_ARCHITECTURE.md": "技术架构方案",
        "CODING_STANDARDS.md": "编码规范",
        "DB_SCHEMA.md": "数据库 Schema",
        "API_CONTRACT.md": "API 契约",
        "TASK_BOOK.md": "任务书",
    }

    docs = {}
    missing = []
    for filename, desc in doc_files.items():
        filepath = Path(docs_dir) / filename
        if filepath.exists():
            docs[filename] = filepath.read_text(encoding="utf-8")
        else:
            missing.append(filename)

    if missing:
        console.print(f"[yellow]⚠️  以下文档尚未生成，将跳过：{', '.join(missing)}[/yellow]")

    return docs


def run_cross_doc_check(docs_dir: str = "docs") -> dict:
    """执行跨文档一致性检查，返回审查报告"""
    docs = load_all_docs(docs_dir)

    if len(docs) < 3:
        console.print("[yellow]⚠️  文档不足 3 份，跳过跨文档审查[/yellow]")
        return {"error": "insufficient_docs", "score": None}

    # 构建审查消息：所有文档按顺序拼接
    doc_sections = []
    total_chars = 0
    for filename, content in docs.items():
        label = {
            "PRD.md": "产品需求文档",
            "TECH_ARCHITECTURE.md": "技术架构方案",
            "CODING_STANDARDS.md": "编码规范",
            "DB_SCHEMA.md": "数据库 Schema",
            "API_CONTRACT.md": "API 契约",
            "TASK_BOOK.md": "任务书",
        }.get(filename, filename)
        doc_sections.append(f"## {label} ({filename})\n\n{content}")
        total_chars += len(content)

    all_docs_text = "\n\n---\n\n".join(doc_sections)
    estimated_tokens = total_chars // 2  # 粗略估计：中文约 2 char/token

    console.print(
        Panel(
            f"[bold blue]🔍 跨文档一致性检查[/bold blue]\n\n"
            f"加载文档：{len(docs)} / 6 份\n"
            f"总字符数：{total_chars:,}\n"
            f"估算 Token：~{estimated_tokens:,} / 1,000,000\n"
            f"利用率：{estimated_tokens / 10000:.1f}%",
            border_style="blue",
        )
    )

    if estimated_tokens > 900_000:
        console.print("[yellow]⚠️  文档总大小接近上下文上限，部分文档可能被截断[/yellow]")

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

    console.print("[dim]🔄 正在审查（预计 1-3 分钟）...[/dim]")

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        max_tokens=8192,
        messages=[
            {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
            {"role": "user", "content": f"## 待审查文档集\n\n{all_docs_text}\n\n请逐项审查上述全部文档的跨文档一致性。"},
        ],
        extra_body={"thinking": {"type": "enabled"}},
    )

    report = response.choices[0].message.content

    # 保存审查报告
    report_path = Path(docs_dir) / "CROSS_DOC_REVIEW.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    full_report = f"# 跨文档一致性审查报告\n\n> 由 DeepSeek V4 Pro (thinking_max) 生成\n> 利用 100 万 token 上下文窗口一次性审查全部 {len(docs)} 份文档\n\n---\n\n{report}"
    report_path.write_text(full_report, encoding="utf-8")
    console.print(f"[green]✅ 审查报告已保存：{report_path}[/green]")

    return {"report": report, "path": str(report_path)}


def print_report_summary(report: str):
    """提取并展示审查摘要"""
    table = Table(title="📊 跨文档一致性评分")
    table.add_column("维度", style="cyan")
    table.add_column("评分", style="bold")

    # 从报告中提取评分
    import re
    dimensions = [
        ("字段一致性", r"字段一致性[：:]?\s*(\d+)/10"),
        ("接口-Schema 匹配度", r"接口[-\s]*Schema\s*匹配度[：:]?\s*(\d+)/10"),
        ("功能完整性", r"功能完整性[：:]?\s*(\d+)/10"),
        ("编码规范一致性", r"编码规范一致性[：:]?\s*(\d+)/10"),
        ("综合评分", r"综合评分[：:]?\s*(\d+)/10"),
    ]

    for name, pattern in dimensions:
        match = re.search(pattern, report)
        score = match.group(1) if match else "?"
        emoji = "🟢" if score != "?" and int(score) >= 8 else ("🟡" if score != "?" and int(score) >= 6 else "🔴")
        table.add_row(f"{emoji} {name}", f"{score}/10")

    console.print(table)


if __name__ == "__main__":
    result = run_cross_doc_check()
    if "report" in result:
        print_report_summary(result["report"])
