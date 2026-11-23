"""Thin wrapper around the Anthropic SDK so the agent loop stays testable.

The agent only depends on `LLMClient.create(...)`; tests swap in a fake.
"""

from __future__ import annotations

from typing import Any

import anthropic

from ..utils.config import Settings
from ..utils.logger import get_logger

log = get_logger("llm")

# Server-side refusal fallback: if the primary model declines a request, the API
# re-runs it on a suitable fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMClient:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None):
        self.settings = settings
        # Credentials resolve from ANTHROPIC_API_KEY (or an `ant auth login` profile).
        self._client = client or anthropic.Anthropic()

    def create(self, *, system: str, messages: list[dict], tools: list[dict]) -> Any:
        log.debug("Calling %s with %d messages", self.settings.model, len(messages))
        response = self._client.beta.messages.create(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            # Adaptive thinking is on by default for this model; effort sets its depth.
            output_config={"effort": self.settings.effort},
            # Cache the growing conversation prefix: each loop turn re-sends the history.
            cache_control={"type": "ephemeral"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        usage = response.usage
        log.info(
            "stop_reason=%s in=%s out=%s cache_read=%s",
            response.stop_reason,
            usage.input_tokens,
            usage.output_tokens,
            getattr(usage, "cache_read_input_tokens", None),
        )
        return response
