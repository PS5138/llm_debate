"""Anthropic Messages-API adapter.

Rewritten from the upstream Khan et al. legacy-Completion-API adapter for
the medical sprint. Supports the Claude 4.x family (Opus 4.7, Sonnet 4.6,
Haiku 4.5) with real per-token pricing, BoN-via-parallel calls, and a
fast-fail path for credit / quota exhaustion that mirrors the OpenAI
adapter's behaviour (see core/llm_api/openai_llm.py).

The class signature (`AnthropicChatModel.__call__`) is unchanged — it
returns a list[LLMResponse] just like before, so core/llm_api/llm.py
doesn't need to know which backend it is talking to.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from traceback import format_exc
from typing import Optional, Union

import attrs
from anthropic import AsyncAnthropic, BadRequestError
from termcolor import cprint

from core.llm_api.base_llm import (
    PRINT_COLORS,
    LLMResponse,
    ModelAPIProtocol,
)
from core.llm_api.openai_llm import OAIChatPrompt
from core.utils import prompt_history_dir

# Known Anthropic IDs. The Messages API also accepts other valid IDs at
# runtime; this set is just for routing in core/llm_api/llm.py.
ANTHROPIC_MODELS = {
    # Claude 4.x family — the medical sprint targets these.
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    # Recent dated aliases that may be used.
    "claude-opus-4-7-20251201",
    "claude-sonnet-4-6-20251101",
    # Earlier 4.x snapshots still on the API (per Anthropic's models overview).
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-5-20251101",
    "claude-opus-4-1-20250805",
    # Legacy IDs kept so QuALITY-era reproduce scripts still route.
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1",
}

# Anthropic Messages API requires max_tokens. Per-round debate arguments
# are capped at ~150 words (~200 tokens) and answer-judge outputs are
# typically <500 tokens of final text, but adaptive-thinking models
# (Opus 4.7) and extended-thinking models (Sonnet 4.6 in oracle mode
# with full patient evidence in context) emit a lot of hidden reasoning
# tokens that count against this budget. 4096 was getting truncated on
# E3 oracle judging — bumped to 8192 so Sonnet 4.6 has room to finish.
DEFAULT_MAX_TOKENS = 8192

# Starting with Opus 4.7, Anthropic rejects any non-default `temperature`,
# `top_p`, or `top_k` with a 400. Those models control sampling internally
# via adaptive thinking; the migration guide tells callers to omit the
# sampling params entirely and to rely on prompting for behavioural
# variation. Add future models here as they ship (the Anthropic doc
# wording is "starting with Claude Opus 4.7", so any newer model is
# expected to follow the same rule).
_PREFIXES_WITHOUT_SAMPLING_PARAMS = (
    "claude-opus-4-7",
)


def _rejects_sampling_params(model_id: str) -> bool:
    mid = model_id.lower()
    return any(mid.startswith(p) for p in _PREFIXES_WITHOUT_SAMPLING_PARAMS)

LOGGER = logging.getLogger(__name__)


def count_tokens(prompt) -> int:
    """Rough word-based token count used by the rate-limit pre-check.

    Real cost is recorded post-hoc from `response.usage`.
    """
    if isinstance(prompt, list):
        text = " ".join(str(m.get("content", "")) for m in prompt)
    else:
        text = str(prompt)
    return len(text.split())


def price_per_token(model_id: str) -> tuple[float, float]:
    """Return (input, output) price per token in USD.

    List rates as of 2026 (per million tokens; from Anthropic's
    `platform.claude.com/docs/.../models/overview`):
      Opus 4.7        — $5  / $25
      Opus 4.6        — $5  / $25  (legacy generally available)
      Opus 4.5        — $5  / $25  (legacy)
      Opus 4.1        — $15 / $75  (legacy)
      Sonnet 4.6      — $3  / $15
      Haiku 4.5       — $1  / $5
      Claude 3.5 Sonnet — $3 / $15
      Claude 3.5 Haiku  — $0.80 / $4
      Claude 3 Opus     — $15 / $75
      Claude 3 Haiku    — $0.25 / $1.25
      Claude 2.1        — $8  / $24
    """
    mid = model_id.lower()
    if mid.startswith("claude-opus-4-1") or mid.startswith("claude-opus-4-20"):
        # Pre-4.5 Opus snapshots kept the old $15 / $75 pricing.
        return 15e-6, 75e-6
    if mid.startswith("claude-opus-4"):
        return 5e-6, 25e-6
    if mid.startswith("claude-sonnet-4") or mid.startswith("claude-3-5-sonnet"):
        return 3e-6, 15e-6
    if mid.startswith("claude-haiku-4"):
        return 1e-6, 5e-6
    if mid.startswith("claude-3-5-haiku"):
        return 0.8e-6, 4e-6
    if mid.startswith("claude-3-opus"):
        return 15e-6, 75e-6
    if mid.startswith("claude-3-sonnet"):
        return 3e-6, 15e-6
    if mid.startswith("claude-3-haiku"):
        return 0.25e-6, 1.25e-6
    if mid == "claude-2.1":
        return 8e-6, 24e-6
    return 0.0, 0.0


def _is_quota_error(exc: Exception) -> bool:
    """Detect Anthropic credit / quota exhaustion vs transient errors."""
    msg = str(exc).lower()
    if "credit balance is too low" in msg:
        return True
    if "insufficient_quota" in msg:
        return True
    if "billing" in msg and "issue" in msg:
        return True
    if "exceeded" in msg and "quota" in msg:
        return True
    return False


def _split_system_messages(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
    """Anthropic takes `system` as a top-level param, not a message role."""
    system_parts = []
    others = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            else:
                system_parts.append(str(content))
        else:
            others.append({"role": role, "content": content})
    system = "\n\n".join(s for s in system_parts if s) if system_parts else None
    return system, others


def _string_prompt_to_messages(prompt: str) -> list[dict]:
    """Best-effort parse of the legacy 'Human:/Assistant:' string format."""
    messages = []
    pattern = r"(Human|Assistant):\s*(.*?)(?=\n(?:Human|Assistant):|$)"
    for m in re.finditer(pattern, prompt, re.S):
        role = "user" if m.group(1) == "Human" else "assistant"
        content = m.group(2).strip()
        if content:
            messages.append({"role": role, "content": content})
    if not messages:
        messages = [{"role": "user", "content": prompt}]
    return messages


@attrs.define()
class AnthropicChatModel(ModelAPIProtocol):
    num_threads: int
    print_prompt_and_response: bool = False
    client: AsyncAnthropic = attrs.field(
        init=False, default=attrs.Factory(AsyncAnthropic)
    )
    available_requests: asyncio.BoundedSemaphore = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

    @staticmethod
    def _create_prompt_history_file(payload):
        filename = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}_prompt.txt"
        path = prompt_history_dir()
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, filename), "w") as f:
            try:
                json_str = json.dumps(payload, indent=4, default=str)
            except Exception:
                json_str = str(payload)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)
        return filename

    @staticmethod
    def _add_response_to_prompt_file(prompt_file, responses):
        try:
            with open(os.path.join(prompt_history_dir(), prompt_file), "a") as f:
                f.write("\n\n======RESPONSE======\n\n")
                payload = [
                    r.to_dict() if hasattr(r, "to_dict") else r.__dict__
                    for r in responses
                ]
                json_str = json.dumps(payload, indent=4, default=str)
                json_str = json_str.replace("\\n", "\n")
                f.write(json_str)
        except Exception:
            pass

    async def _single_message_call(
        self,
        model_id: str,
        system: Optional[str],
        messages: list[dict],
        max_tokens: int,
        **kwargs,
    ):
        # Allow-list of Anthropic Messages API parameters.
        allowed = {"temperature", "top_p", "top_k", "stop_sequences", "metadata"}
        call_kwargs = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if _rejects_sampling_params(model_id):
            dropped = [k for k in ("temperature", "top_p", "top_k") if k in call_kwargs]
            for k in dropped:
                call_kwargs.pop(k, None)
            if dropped:
                LOGGER.info(
                    f"Dropping {dropped} for {model_id}: this model rejects "
                    f"non-default sampling params and controls sampling "
                    f"internally via adaptive thinking. BoN diversity now "
                    f"comes from the model's own stochasticity + prompt "
                    f"variation, not from temperature."
                )
        elif "temperature" in call_kwargs and "top_p" in call_kwargs:
            # Older Claude 4.x rejects requests that specify both sampling
            # controls. The repo configs always set top_p=1.0, so keep the
            # intentional temperature setting and omit top_p.
            call_kwargs.pop("top_p")
        if system is not None:
            call_kwargs["system"] = system
        call_kwargs["max_tokens"] = max_tokens
        return await self.client.messages.create(
            model=model_id,
            messages=messages,
            **call_kwargs,
        )

    async def __call__(
        self,
        model_ids: list[str],
        prompt: Union[str, OAIChatPrompt],
        print_prompt_and_response: bool,
        max_attempts: int,
        n: int = 1,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        assert (
            len(model_ids) == 1
        ), "Anthropic adapter only supports one model at a time"
        model_id = model_ids[0]

        # Normalise the prompt to Anthropic Messages-API shape.
        if isinstance(prompt, str):
            messages = _string_prompt_to_messages(prompt)
            system = None
        else:
            system, messages = _split_system_messages(list(prompt))

        # Anthropic requires the conversation to start with user.
        if messages and messages[0].get("role") != "user":
            messages.insert(0, {"role": "user", "content": "[continue]"})

        # max_tokens handling. The shared wrapper historically used
        # Anthropic's legacy name, while the Messages API uses max_tokens.
        max_tokens = kwargs.pop("max_tokens", None)
        legacy_max_tokens = kwargs.pop("max_tokens_to_sample", None)
        if max_tokens is None:
            max_tokens = legacy_max_tokens
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        # Drop OpenAI-only params that won't apply.
        for stale in (
            "max_words",
            "min_words",
            "timeout",
            "logprobs",
            "top_logprobs",
            "num_candidates_per_completion",
        ):
            kwargs.pop(stale, None)

        prompt_file = self._create_prompt_history_file(
            {
                "system": system,
                "messages": messages,
                "model": model_id,
                "max_tokens": max_tokens,
                "n": n,
            }
        )

        async def attempt_one_call():
            async with self.available_requests:
                api_start = time.time()
                response = await self._single_message_call(
                    model_id, system, messages, max_tokens, **kwargs
                )
                return response, time.time() - api_start

        # Anthropic Messages API does not natively support n>1, so fan out
        # for BoN. The semaphore caps concurrency per the configured
        # num_threads.
        n_calls = max(1, int(n))

        results = None
        for i in range(max_attempts):
            try:
                results = await asyncio.gather(
                    *[attempt_one_call() for _ in range(n_calls)]
                )
            except Exception as e:
                if _is_quota_error(e):
                    raise RuntimeError(
                        "Anthropic credit / quota exhausted. Check "
                        "https://console.anthropic.com/settings/billing "
                        "and re-run once credit is available — cached "
                        "partial transcripts will be reused."
                    ) from e
                if isinstance(e, BadRequestError):
                    detail = getattr(e, "message", None) or str(e)
                    raise RuntimeError(
                        f"Anthropic rejected the request as invalid (model={model_id}): "
                        f"{detail}. This is usually a configuration issue (unknown "
                        f"model name, account lacks access, max_tokens too small, "
                        f"malformed parameters), not a transient API failure, so "
                        f"the call was not retried."
                    ) from e
                error_info = (
                    f"Exception Type: {type(e).__name__}, "
                    f"Error Details: {str(e)}, "
                    f"Traceback: {format_exc()}"
                )
                LOGGER.warn(
                    f"Encountered API error: {error_info}.\n"
                    f"Retrying now. (Attempt {i})"
                )
                await asyncio.sleep(1.5**i)
            else:
                break

        if results is None:
            raise RuntimeError(
                f"Failed to get a response from the Anthropic API after "
                f"{max_attempts} attempts."
            )

        in_price, out_price = price_per_token(model_id)
        total_duration = time.time() - start
        responses: list[LLMResponse] = []
        for response, api_duration in results:
            usage = getattr(response, "usage", None)
            in_tokens = int(getattr(usage, "input_tokens", 0)) if usage else 0
            out_tokens = int(getattr(usage, "output_tokens", 0)) if usage else 0
            cost = in_tokens * in_price + out_tokens * out_price

            completion = ""
            for block in getattr(response, "content", []) or []:
                btype = getattr(block, "type", None)
                if btype == "text":
                    completion += getattr(block, "text", "") or ""
                elif hasattr(block, "text"):
                    completion += block.text or ""

            stop_reason = getattr(response, "stop_reason", None)
            responses.append(
                LLMResponse(
                    model_id=model_id,
                    completion=completion,
                    stop_reason=str(stop_reason) if stop_reason else "",
                    duration=total_duration,
                    api_duration=api_duration,
                    cost=cost,
                )
            )

        self._add_response_to_prompt_file(prompt_file, responses)

        if self.print_prompt_and_response or print_prompt_and_response:
            try:
                preview = json.dumps(
                    {"system": system, "messages": messages},
                    indent=2,
                    default=str,
                )
                cprint(preview, "yellow")
                for r in responses:
                    cprint(f"Response ({r.model_id}):", "white")
                    cprint(
                        r.completion,
                        PRINT_COLORS.get("assistant", "white"),
                        attrs=["bold"],
                    )
                print()
            except Exception:
                pass

        LOGGER.debug(
            f"Completed call to {model_id} in {total_duration:.2f}s "
            f"(n={n_calls})"
        )
        return responses
