"""
Cross-Document Consistency Checker

Leverages the DeepSeek V4 1M token context window to load all 6 Phase 0
documents into a single API call for cross-review. Detects field name
inconsistencies, logical contradictions, API-Schema mismatches, and more.

This is a capability impossible with 200K context windows.
"""

from pathlib import Path
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import DEEPSEEK_API_KEY

console = Console()

CHECKER_SYSTEM_PROMPT = """# Role Definition

You are a senior QA review expert specializing in cross-document consistency checking.

# Task

You will receive a complete set of APP development documents (PRD, Tech Architecture, Coding Standards, DB Schema, API Contract, Task Book). Review each of the following dimensions:

## Review Dimensions

1. **Field Name Consistency**: Is the same entity consistently named across documents?
   - Example: Is "user nickname" in the PRD called `nickname` or `display_name` in the Schema? `nickName` or `nickname` in the API?

2. **API-Schema Alignment**: Do request/response fields defined in the API Contract map to corresponding columns in the DB Schema?
   - Check that every endpoint's request body / response body field has a matching column in the database tables

3. **Feature Completeness**: Does every P0 feature in the PRD have a corresponding API endpoint? Does every page have a corresponding frontend task?

4. **Coding Standards Consistency**: Do the Schema and API naming conventions match the Coding Standards?
   - Example: Standards mandate snake_case — does camelCase appear in the Schema?

5. **Data Type Consistency**: Is the same field defined with consistent types across documents?
   - Example: PRD says "age is an integer", Schema has INTEGER, but API has string?

## Output Format

Structure your review report as follows:

### 🔴 Must Fix (Blocking Issues)
- Specific problem description
- Documents and sections involved
- Fix recommendation

### 🟡 Should Fix (Potential Risks)
- Specific problem description
- Documents and sections involved
- Fix recommendation

### 🟢 Can Ignore (Noted)
- Observed difference
- Why it can be ignored

### 📊 Scoring
- Field Consistency: X/10
- API-Schema Alignment: X/10
- Feature Completeness: X/10
- Coding Standards Consistency: X/10
- **Overall Score: X/10**

## Key Principles
- Verify against document source text; do not rely on memory or assumptions
- Cite specific document names and section locations
- Fix recommendations must be specific down to field names or code lines
"""


def load_all_docs(docs_dir: str = "docs") -> dict[str, str]:
    """加载 Phase 0 全部产出文档"""
    doc_files = {
        "PRD.md": "Product Requirements Document",
        "TECH_ARCHITECTURE.md": "Technical Architecture",
        "CODING_STANDARDS.md": "Coding Standards",
        "DB_SCHEMA.md": "Database Schema",
        "API_CONTRACT.md": "API Contract",
        "TASK_BOOK.md": "Task Book",
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
        console.print(f"[yellow]⚠️  The following documents are not yet generated, skipping: {', '.join(missing)}[/yellow]")

    return docs


def run_cross_doc_check(docs_dir: str = "docs") -> dict:
    """执行跨文档一致性检查，返回审查报告"""
    docs = load_all_docs(docs_dir)

    if len(docs) < 3:
        console.print("[yellow]⚠️  Fewer than 3 documents available, skipping cross-document review[/yellow]")
        return {"error": "insufficient_docs", "score": None}

    # 构建审查消息：所有文档按顺序拼接
    doc_sections = []
    total_chars = 0
    for filename, content in docs.items():
        label = {
            "PRD.md": "Product Requirements Document",
            "TECH_ARCHITECTURE.md": "Technical Architecture",
            "CODING_STANDARDS.md": "Coding Standards",
            "DB_SCHEMA.md": "Database Schema",
            "API_CONTRACT.md": "API Contract",
            "TASK_BOOK.md": "Task Book",
        }.get(filename, filename)
        doc_sections.append(f"## {label} ({filename})\n\n{content}")
        total_chars += len(content)

    all_docs_text = "\n\n---\n\n".join(doc_sections)
    estimated_tokens = total_chars // 2  # rough estimate: ~2 chars per token for mixed content

    console.print(
        Panel(
            f"[bold blue]🔍 Cross-Document Consistency Check[/bold blue]\n\n"
            f"Documents loaded: {len(docs)} / 6\n"
            f"Total characters: {total_chars:,}\n"
            f"Estimated tokens: ~{estimated_tokens:,} / 1,000,000\n"
            f"Utilization: {estimated_tokens / 10000:.1f}%",
            border_style="blue",
        )
    )

    if estimated_tokens > 900_000:
        console.print("[yellow]⚠️  Total document size near context limit; some documents may be truncated[/yellow]")

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

    console.print("[dim]🔄 Reviewing (estimated 1-3 minutes)...[/dim]")

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        max_tokens=8192,
        messages=[
            {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
            {"role": "user", "content": f"## Document Set for Review\n\n{all_docs_text}\n\nPlease review all documents above for cross-document consistency across every dimension."},
        ],
        extra_body={"thinking": {"type": "enabled"}},
    )

    report = response.choices[0].message.content

    # 保存审查报告
    report_path = Path(docs_dir) / "CROSS_DOC_REVIEW.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    full_report = f"# Cross-Document Consistency Review Report\n\n> Generated by DeepSeek V4 Pro (thinking_max)\n> Single-pass review of all {len(docs)} documents using the 1M token context window\n\n---\n\n{report}"
    report_path.write_text(full_report, encoding="utf-8")
    console.print(f"[green]✅ Review report saved: {report_path}[/green]")

    return {"report": report, "path": str(report_path)}


def print_report_summary(report: str):
    """Extract and display review summary"""
    table = Table(title="📊 Cross-Document Consistency Scores")
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", style="bold")

    # Extract scores from report text
    import re
    dimensions = [
        ("Field Consistency", r"Field Consistency[：:]?\s*(\d+)/10"),
        ("API-Schema Alignment", r"API[- ]?Schema\s*Alignment[：:]?\s*(\d+)/10"),
        ("Feature Completeness", r"Feature\s*Completeness[：:]?\s*(\d+)/10"),
        ("Coding Standards", r"Coding\s*Standards\s*Consistency[：:]?\s*(\d+)/10"),
        ("Overall Score", r"Overall\s*Score[：:]?\s*(\d+)/10"),
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
