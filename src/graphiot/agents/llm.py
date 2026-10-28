"""LLM providers behind one interface: Ollama, Google Gemini, Anthropic Claude.

A :class:`ChatSession` holds one agent conversation in the provider's *native*
history format (so provider-specific blocks — e.g. Claude's thinking blocks —
are passed back unchanged), and exposes the same three operations everywhere:

* :meth:`ChatSession.step` — send the history, get a normalized :class:`Turn`
  (text, tool calls, stop reason, the model that actually served the call);
* :meth:`ChatSession.add_tool_results` — append the results of the tool calls;
* :meth:`ChatSession.add_user` — append a user nudge.

API keys are resolved through :mod:`graphiot.utils.secrets` from the
environment-variable *name* in the config; they are never logged.

Provider notes (verified against the installed SDKs: anthropic 2.3.0,
google-genai 2.33.0, ollama 0.6.3):

* Claude: Messages API with client tools (``input_schema``); the full
  ``response.content`` is appended each turn; no sampling parameters (removed in
  API version 2026-10-21); effort via ``output_config``; optional server-side
  refusal fallback (``fallbacks="default"``, beta ``server-side-fallback-2026-07-01``).
* Gemini: function declarations with ``parameters_json_schema``; automatic
  function calling disabled so the agent loop executes every call itself.
* Ollama: ``/api/chat`` tools; tool results as ``role: tool`` messages.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..config.schema import LLMAgent
from ..utils.secrets import STORE, SecretStore

_ids = itertools.count(1)


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall]
    stop: str                      # tool_use | end | max_tokens | refusal
    model: str                     # model that actually served the call
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class ChatSession(ABC):
    def __init__(self, cfg: LLMAgent, system: str, tools: list[dict]):
        self.cfg = cfg
        self.system = system
        self.tools = tools

    @abstractmethod
    async def step(self) -> Turn: ...

    @abstractmethod
    def add_tool_results(self, results: list[tuple[ToolCall, Any, bool]]) -> None: ...

    @abstractmethod
    def add_user(self, text: str) -> None: ...


def _result_text(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str)


def recover_text_tool_calls(text: str, tool_names: set[str]) -> list[ToolCall]:
    """Some local models emit a tool call as JSON in their reply instead of the
    provider's tool-call format (seen with qwen2.5-coder on Ollama). Recover
    ``{"name": ..., "arguments"|"parameters": {...}}`` objects (or a list of them,
    possibly in a ```json fence) when the name is one of our tools."""
    import re
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    starts = [i for i, ch in enumerate(s) if ch in "[{"]
    for i in starts[:3]:
        try:
            obj, _ = json.JSONDecoder().raw_decode(s[i:])
        except json.JSONDecodeError:
            continue
        items = obj if isinstance(obj, list) else [obj]
        calls = []
        for it in items:
            if not isinstance(it, dict) or it.get("name") not in tool_names:
                continue
            args = it.get("arguments", it.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if isinstance(args, dict):
                calls.append(ToolCall(id=f"call_{next(_ids)}", name=it["name"], args=args))
        if calls:
            return calls
    return []


# --------------------------------------------------------------------------- #
class OllamaSession(ChatSession):
    def __init__(self, cfg: LLMAgent, system: str, user: str, tools: list[dict]):
        super().__init__(cfg, system, tools)
        import ollama
        self._client = ollama.AsyncClient(host=cfg.base_url, timeout=cfg.request_timeout_s)
        self._tools = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system},
                                     {"role": "user", "content": user}]

    async def step(self) -> Turn:
        t0 = time.monotonic()
        try:
            resp = await self._client.chat(
                model=self.cfg.model, messages=self.messages, tools=self._tools,
                options={"temperature": self.cfg.temperature, "num_predict": self.cfg.max_tokens})
        except Exception as exc:
            raise LLMError(f"ollama: {exc}") from exc
        msg = resp.message
        calls = []
        for tc in (msg.tool_calls or []):
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_unparsed": args}
            calls.append(ToolCall(id=f"call_{next(_ids)}", name=tc.function.name, args=dict(args or {})))
        meta: dict = {}
        if not calls and msg.content:
            calls = recover_text_tool_calls(msg.content, {t["name"] for t in self.tools})
            if calls:
                meta["recovered_text_tool_call"] = True
        self.messages.append({"role": "assistant", "content": msg.content or "",
                              **({"tool_calls": [{"function": {"name": c.name, "arguments": c.args}}
                                                 for c in calls]} if calls else {})})
        done = getattr(resp, "done_reason", None)
        stop = "tool_use" if calls else ("max_tokens" if done == "length" else "end")
        return Turn(text=msg.content or "", tool_calls=calls, stop=stop, model=resp.model or self.cfg.model,
                    usage={"input_tokens": getattr(resp, "prompt_eval_count", None),
                           "output_tokens": getattr(resp, "eval_count", None)},
                    latency_s=time.monotonic() - t0, meta=meta)

    def add_tool_results(self, results):
        for call, result, is_error in results:
            content = _result_text({"error": result} if is_error else result)
            self.messages.append({"role": "tool", "tool_name": call.name, "content": content})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


# --------------------------------------------------------------------------- #
class AnthropicSession(ChatSession):
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, cfg: LLMAgent, system: str, user: str, tools: list[dict],
                 secrets: SecretStore):
        super().__init__(cfg, system, tools)
        import anthropic
        key = secrets.get(cfg.api_key_env)
        if not key:
            raise LLMError(f"Claude API key not set (environment variable {cfg.api_key_env})")
        self._anthropic = anthropic
        self._client = anthropic.AsyncAnthropic(api_key=key, timeout=cfg.request_timeout_s)
        self._tools = [{"name": t["name"], "description": t["description"],
                        "input_schema": t["parameters"]} for t in tools]
        self.messages: list[dict] = [{"role": "user", "content": user}]

    async def step(self) -> Turn:
        t0 = time.monotonic()
        kwargs: dict[str, Any] = dict(model=self.cfg.model, max_tokens=self.cfg.max_tokens,
                                      system=self.system, tools=self._tools, messages=self.messages,
                                      output_config={"effort": self.cfg.effort})
        if self.cfg.fallbacks == "default":
            kwargs.update(betas=[self.FALLBACK_BETA], fallbacks="default")
        a = self._anthropic
        try:
            resp = await self._client.beta.messages.create(**kwargs)
        except a.BadRequestError as exc:
            raise LLMError(f"claude bad request: {exc.message}") from exc
        except (a.AuthenticationError, a.PermissionDeniedError) as exc:
            raise LLMError(f"claude auth: {exc.message}") from exc
        except a.RateLimitError as exc:
            raise LLMError("claude rate limited (after SDK retries)") from exc
        except a.APIStatusError as exc:
            raise LLMError(f"claude HTTP {exc.status_code}: {exc.message}") from exc
        except a.APIConnectionError as exc:
            raise LLMError(f"claude connection: {exc}") from exc
        # Append the full content (thinking, fallback and tool_use blocks) unchanged.
        self.messages.append({"role": "assistant", "content": resp.content})
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, args=dict(b.input or {}))
                 for b in resp.content if b.type == "tool_use"]
        meta: dict = {}
        fb = [{"from": b.from_.model, "to": b.to.model} for b in resp.content if b.type == "fallback"]
        if fb:
            meta["fallbacks"] = fb
        if resp.stop_reason == "refusal":
            sd = getattr(resp, "stop_details", None)
            meta["refusal"] = {"category": getattr(sd, "category", None),
                               "explanation": getattr(sd, "explanation", None)}
        stop = {"tool_use": "tool_use", "end_turn": "end", "max_tokens": "max_tokens",
                "refusal": "refusal"}.get(resp.stop_reason, "end")
        if calls and stop != "refusal":
            stop = "tool_use"
        u = resp.usage
        return Turn(text=text, tool_calls=calls if stop == "tool_use" else [], stop=stop,
                    model=resp.model, usage={"input_tokens": u.input_tokens,
                                             "output_tokens": u.output_tokens},
                    latency_s=time.monotonic() - t0, meta=meta)

    def add_tool_results(self, results):
        # All results of one turn go back in a single user message.
        blocks = [{"type": "tool_result", "tool_use_id": call.id,
                   "content": _result_text(result), **({"is_error": True} if is_error else {})}
                  for call, result, is_error in results]
        self.messages.append({"role": "user", "content": blocks})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


# --------------------------------------------------------------------------- #
class GeminiSession(ChatSession):
    def __init__(self, cfg: LLMAgent, system: str, user: str, tools: list[dict],
                 secrets: SecretStore):
        super().__init__(cfg, system, tools)
        from google import genai
        from google.genai import types
        key = secrets.get(cfg.api_key_env)
        if not key:
            raise LLMError(f"Gemini API key not set (environment variable {cfg.api_key_env})")
        self._types = types
        self._client = genai.Client(api_key=key)
        decls = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                           parameters_json_schema=t["parameters"]) for t in tools]
        self._config = types.GenerateContentConfig(
            system_instruction=system, temperature=cfg.temperature,
            max_output_tokens=cfg.max_tokens, tools=[types.Tool(function_declarations=decls)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        self.contents: list = [types.Content(role="user", parts=[types.Part(text=user)])]

    async def step(self) -> Turn:
        t0 = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self.cfg.model, contents=self.contents, config=self._config),
                timeout=self.cfg.request_timeout_s)
        except Exception as exc:
            raise LLMError(f"gemini: {exc}") from exc
        cand = (resp.candidates or [None])[0]
        if cand is None or cand.content is None:
            reason = str(getattr(resp, "prompt_feedback", "") or "")
            return Turn(text="", tool_calls=[], stop="refusal", model=self.cfg.model,
                        latency_s=time.monotonic() - t0, meta={"refusal": {"category": reason}})
        self.contents.append(cand.content)
        calls = [ToolCall(id=fc.id or f"call_{next(_ids)}", name=fc.name, args=dict(fc.args or {}))
                 for fc in (resp.function_calls or [])]
        text = "".join(p.text for p in (cand.content.parts or []) if getattr(p, "text", None))
        finish = str(getattr(cand, "finish_reason", "") or "")
        stop = "tool_use" if calls else ("max_tokens" if "MAX_TOKENS" in finish else
                                         "refusal" if "SAFETY" in finish else "end")
        um = getattr(resp, "usage_metadata", None)
        return Turn(text=text, tool_calls=calls, stop=stop,
                    model=getattr(resp, "model_version", None) or self.cfg.model,
                    usage={"input_tokens": getattr(um, "prompt_token_count", None),
                           "output_tokens": getattr(um, "candidates_token_count", None)},
                    latency_s=time.monotonic() - t0)

    def add_tool_results(self, results):
        t = self._types
        parts = [t.Part.from_function_response(
            name=call.name, response={"error": result} if is_error else {"result": result})
            for call, result, is_error in results]
        self.contents.append(t.Content(role="user", parts=parts))

    def add_user(self, text: str) -> None:
        t = self._types
        self.contents.append(t.Content(role="user", parts=[t.Part(text=text)]))


# --------------------------------------------------------------------------- #
def new_session(cfg: LLMAgent, *, system: str, user: str, tools: list[dict],
                secrets: SecretStore = STORE) -> ChatSession:
    if cfg.provider == "ollama":
        return OllamaSession(cfg, system, user, tools)
    if cfg.provider == "anthropic":
        return AnthropicSession(cfg, system, user, tools, secrets)
    if cfg.provider == "gemini":
        return GeminiSession(cfg, system, user, tools, secrets)
    raise ValueError(f"unknown provider {cfg.provider!r}")
