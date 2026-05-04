from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AgentConfig:
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    model: str = "deepseek-v4-pro"
    thinking_mode: str | None = None  # None | "thinking" | "thinking_max"


class BaseAgent:
    name: str = "base"
    description: str = ""
    system_prompt: str = ""
    config: AgentConfig = AgentConfig()

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self._base_url = base_url or "https://api.deepseek.com"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    """DeepSeek API Key not found.

To fix this:
  1. Copy the example env file: cp .env.example .env
  2. Add your key to .env: DEEPSEEK_API_KEY=sk-your-key-here
  3. Or export it: export DEEPSEEK_API_KEY=sk-your-key-here

See .env.example for reference:
  https://github.com/withAIx/DeepSeek-AppForge/blob/main/project-orchestrator/.env.example

Get a DeepSeek API key at:
  https://platform.deepseek.com/api_keys"""
                )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    def _build_messages(self, user_input: str) -> list[dict]:
        return [
            {"role": "user", "content": user_input},
        ]

    def _build_extra_body(self, thinking_mode: str | None = None) -> dict:
        mode = thinking_mode or self.config.thinking_mode
        if mode:
            return {"thinking": {"type": "enabled"}}
        return {}

    def run(self, user_input: str, **kwargs) -> str:
        cfg = self._merge_config(kwargs)
        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + self._build_messages(user_input)
        resp = self.client.chat.completions.create(
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg.get("top_p", 1.0),
            messages=messages,
            extra_body=self._build_extra_body(),
        )
        return resp.choices[0].message.content

    def run_stream(self, user_input: str, **kwargs) -> Iterator[str]:
        cfg = self._merge_config(kwargs)
        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + self._build_messages(user_input)
        stream = self.client.chat.completions.create(
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg.get("top_p", 1.0),
            messages=messages,
            stream=True,
            extra_body=self._build_extra_body(),
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def _merge_config(self, overrides: dict) -> dict:
        base = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        base.update({k: v for k, v in overrides.items() if k in base})
        return base

    def __repr__(self) -> str:
        return f"<Agent name={self.name}>"
