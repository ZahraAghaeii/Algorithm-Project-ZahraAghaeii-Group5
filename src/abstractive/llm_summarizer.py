from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.metisai.ir/openai/v1"
    temperature: float = 0.2
    max_tokens: int = 250

class MetisLLMSummarizer:

    def __init__(self, config: LLMConfig = LLMConfig(), api_key: Optional[str] = None) -> None:
        self.config = config
        key = api_key or os.getenv("METISAI_API_KEY")
        if not key:
            raise RuntimeError(
                "Missing METISAI_API_KEY. Set it as an environment variable before running."
            )

        self.client = OpenAI(api_key=key, base_url=config.base_url)

    def summarize(self, text: str, target_sentences: int = 5) -> str:
 
        text = (text or "").strip()
        if not text:
            return ""

        system_msg = (
            "You are a helpful assistant that writes concise, faithful summaries. "
            "Do not invent facts. Keep the summary readable."
        )

        user_msg = (
            f"Summarize the following text in about {target_sentences} sentences. "
            "Preserve key facts and avoid redundancy. "
            "Return only the summary text.\n\n"
            f"TEXT:\n{text}"
        )

        resp = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        out = resp.choices[0].message.content or ""
        return out.strip()
