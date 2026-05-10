import json
from pathlib import Path
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import (
    DEEPSEEK_API_KEY,
    ORCHESTRATOR_MODEL,
    AGENT_REGISTRY,
    PHASE0_ORDER,
    PROJECT_NAME,
)
from state import (
    load_state,
    save_state,
    init_state,
    get_next_phase0_agent,
    is_phase0_complete,
    mark_agent_completed,
    mark_agent_failed,
    increment_retry,
    list_projects,
)
from worker import run_worker, save_agent_output

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

MAX_RETRIES = 2

# Orchestrator system prompt (loaded from external file)
ORCHESTRATOR_SYSTEM = Path("agents/orchestrator.md").read_text(encoding="utf-8")

# Tool definitions available to Orchestrator (DeepSeek/OpenAI Function Calling format)
ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_project_state",
            "description": "Read current project state and return JSON-formatted status data",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_agent",
            "description": "Route a task to a specified sub-agent for execution — the Orchestrator's core scheduling action",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "enum": list(AGENT_REGISTRY.keys()),
                        "description": "Name of the sub-agent to invoke",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Specific task description for the sub-agent",
                    },
                    "plan": {
                        "type": "object",
                        "description": "Scheduling plan including routing rationale and expected output",
                        "properties": {
                            "reason": {"type": "string"},
                            "expected_output": {"type": "string"},
                            "parallel_tasks": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "blockers": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["reason", "expected_output"],
                    },
                },
                "required": ["agent_name", "task_description", "plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_state",
            "description": "Update project state file",
            "parameters": {
                "type": "object",
                "properties": {
                    "updates": {
                        "type": "object",
                        "description": "State fields to merge/update",
                    }
                },
                "required": ["updates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents from the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"],
            },
        },
    },
]


def execute_tool(tool_name: str, tool_input: dict, state: dict) -> tuple[str, dict]:
    """Execute Orchestrator Tool call, return (result string, updated state)"""

    if tool_name == "read_project_state":
        return json.dumps(state, ensure_ascii=False, indent=2), state

    elif tool_name == "read_file":
        project_dir = Path(state["_project_dir"])
        path = project_dir / tool_input["path"]
        if path.exists():
            return path.read_text(encoding="utf-8"), state
        return f"[File not found: {tool_input['path']}]", state

    elif tool_name == "update_state":
        state.update(tool_input["updates"])
        save_state(state)
        return "State updated successfully", state

    elif tool_name == "route_to_agent":
        agent_name = tool_input["agent_name"]
        task_desc = tool_input["task_description"]
        plan = tool_input["plan"]

        # Show scheduling plan
        console.print(
            Panel(
                f"[bold yellow]🗺️ Scheduling Plan[/bold yellow]\n\n"
                f"**Route to**: {AGENT_REGISTRY[agent_name]['display_name']}\n"
                f"**Rationale**: {plan['reason']}\n"
                f"**Expected output**: {plan['expected_output']}",
                border_style="yellow",
            )
        )

        # Execute sub-agent (with retry)
        retry_count = state.get("retry_counts", {}).get(agent_name, 0)
        result = None
        error_msg = None

        project_dir = Path(state["_project_dir"])

        for attempt in range(MAX_RETRIES + 1):
            try:
                result = run_worker(agent_name, task_desc, project_dir, stream=True)

                # For Phase 0 doc generation, save output file
                saved_path = save_agent_output(agent_name, result, project_dir)
                if saved_path:
                    state = mark_agent_completed(state, agent_name)

                return (
                    f"✅ {AGENT_REGISTRY[agent_name]['display_name']} completed\n{result[:500]}...",
                    state,
                )

            except Exception as e:
                error_msg = str(e)
                console.print(f"[red]❌ Attempt {attempt + 1} failed: {error_msg}[/red]")
                if attempt < MAX_RETRIES:
                    console.print(f"[yellow]🔄 Retry attempt {attempt + 2}...[/yellow]")
                    increment_retry(state, agent_name)

        # All retries exhausted, execute degradation
        state = mark_agent_failed(state, agent_name, error_msg)
        return f"⚠️ {agent_name} execution failed (degraded), error: {error_msg}", state

    return f"Unknown tool: {tool_name}", state


def _serialize_tool_calls(tool_calls) -> list[dict]:
    """Serialize OpenAI tool_calls objects into a list of dicts appendable to messages"""
    result = []
    for tc in tool_calls:
        result.append({
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        })
    return result


def run_orchestrator_turn(user_input: str, state: dict) -> dict:
    """Execute one Orchestrator scheduling loop (DeepSeek/OpenAI Function Calling)"""

    state_summary = json.dumps(state, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        {
            "role": "user",
            "content": f"## Current Project State\n```json\n{state_summary}\n```\n\n## User Input\n{user_input}",
        },
    ]

    console.print("[dim]🧠 Orchestrator analyzing...[/dim]")

    # Orchestrator Function Calling loop
    while True:
        response = _get_client().chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            max_tokens=8192,
            messages=messages,
            tools=ORCHESTRATOR_TOOLS,
            extra_body={"thinking": {"type": "enabled"}},
        )

        msg = response.choices[0].message

        # Display text output
        if msg.content:
            console.print(Panel(msg.content, border_style="blue", title="📊 Orchestrator"))

        # Exit loop if no Tool calls
        if response.choices[0].finish_reason != "tool_calls":
            break

        # Append assistant message (with tool_calls) FIRST — required by API order.
        # DeepSeek thinking mode requires reasoning_content to be passed back.
        assistant_msg = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": _serialize_tool_calls(msg.tool_calls),
        }
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        messages.append(assistant_msg)

        # Then append tool result messages
        for tc in msg.tool_calls:
            console.print(f"[dim]⚙️  Calling tool: {tc.function.name}[/dim]")
            tool_input = json.loads(tc.function.arguments)
            result_text, state = execute_tool(tc.function.name, tool_input, state)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

    return state


class ProjectOrchestrator:
    def __init__(self, project_name: str | None = None):
        if project_name:
            self.state = init_state(project_name)
        else:
            # Load most recently updated project (backward compat)
            projects = list_projects()
            if projects:
                self.state = load_state(projects[0]["name"])
            else:
                self.state = init_state(PROJECT_NAME)

    def print_status(self):
        """Print current project status table"""
        table = Table(title="📊 Project Development Progress", show_header=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Document", style="white")
        table.add_column("Status", style="bold")

        doc_status_map = {
            "completed": "✅ Completed",
            "in_progress": "🔄 In Progress",
            "pending": "⏳ Pending",
            "failed": "❌ Failed (Degraded)",
        }

        for agent_name in PHASE0_ORDER:
            doc_info = self.state["phase0_documents"].get(agent_name, {})
            status = doc_status_map.get(doc_info.get("status", "pending"), "⏳ Pending")
            table.add_row(
                AGENT_REGISTRY[agent_name]["display_name"],
                doc_info.get("path", "-"),
                status,
            )

        console.print(table)

    def run(self, user_input: str):
        """Main execution entry point"""
        # Phase 0 auto-advance (check next pending document)
        if not is_phase0_complete(self.state):
            next_agent = get_next_phase0_agent(self.state)
            if next_agent:
                agent_display = AGENT_REGISTRY[next_agent]["display_name"]
                console.print(
                    f"\n[bold green]➡️  Phase 0 in progress: next → {agent_display}[/bold green]"
                )
            else:
                # PRD completed but awaiting user review
                prd_status = self.state["phase0_documents"]["prd_expert"]["status"]
                if prd_status == "completed" and not self.state.get("prd_reviewed", False):
                    console.print(
                        "\n[bold yellow]📋 PRD has been generated — please review docs/PRD.md[/bold yellow]"
                    )
                    console.print(
                        "[dim]Reply 'confirmed' to proceed, or describe what you'd like to change.[/dim]"
                    )

        self.state = run_orchestrator_turn(user_input, self.state)
