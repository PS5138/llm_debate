"""No-network regression tests for provider token-limit routing."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llm_api.anthropic_llm import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    AnthropicChatModel,
)
from core.llm_api.llm import ModelAPI  # noqa: E402


class AnthropicTokenLimitRegressionTests(unittest.TestCase):
    @staticmethod
    def _model_api() -> ModelAPI:
        """Build the wrapper without constructing network clients."""
        anthropic = object.__new__(AnthropicChatModel)
        anthropic.num_threads = 1
        anthropic.print_prompt_and_response = False
        anthropic.available_requests = asyncio.BoundedSemaphore(1)

        api = object.__new__(ModelAPI)
        api._anthropic_chat = anthropic
        api.running_cost = 0
        api.model_timings = {}
        api.model_wait_times = {}
        return api

    def _effective_limit(self, requested_limit: int | None) -> int:
        observed_limits: list[int] = []

        async def fake_message_call(
            _self,
            _model_id,
            _system,
            _messages,
            max_tokens,
            **_kwargs,
        ):
            observed_limits.append(max_tokens)
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=0, output_tokens=0),
                content=[SimpleNamespace(type="text", text="ok")],
                stop_reason="end_turn",
            )

        with (
            patch.object(
                AnthropicChatModel,
                "_single_message_call",
                new=fake_message_call,
            ),
            patch.object(
                AnthropicChatModel,
                "_create_prompt_history_file",
                return_value="unused",
            ),
            patch.object(AnthropicChatModel, "_add_response_to_prompt_file"),
        ):
            responses = asyncio.run(
                self._model_api()(
                    "claude-sonnet-4-6",
                    [{"role": "user", "content": "test"}],
                    requested_limit,
                )
            )

        self.assertEqual([response.completion for response in responses], ["ok"])
        self.assertEqual(len(observed_limits), 1)
        return observed_limits[0]

    def test_none_uses_anthropic_adapter_default(self) -> None:
        self.assertEqual(self._effective_limit(None), DEFAULT_MAX_TOKENS)

    def test_explicit_limit_is_preserved(self) -> None:
        self.assertEqual(self._effective_limit(1234), 1234)


if __name__ == "__main__":
    unittest.main()
