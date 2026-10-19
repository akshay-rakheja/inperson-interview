"""Single model-provider boundary: OpenRouter via the OpenAI-compatible Python SDK.

`ModelClient.generate(task=..., payload=..., schema=...)` returns a validated Pydantic object or
raises `LLMError` with a user-facing `kind`. There is no tool execution, no automatic model or
provider fallback, and OPENAI_API_KEY is never consulted.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from . import config


class LLMError(Exception):
    """kind: missing_key | auth | credits | refusal | timeout | unsupported_schema | malformed | request_failed"""

    def __init__(self, kind: str, message: str, raw: str | None = None):
        super().__init__(f"[{kind}] {message}")
        self.kind = kind
        self.message = message
        self.raw = raw  # unparsed model output, kept locally for diagnosis


class ModelClient(Protocol):
    provider: str
    base_url: str
    model: str

    def generate(self, *, task: str, payload: dict, schema: type[BaseModel]) -> BaseModel: ...


def strict_json_schema(schema: type[BaseModel]) -> dict:
    """Pydantic JSON schema adjusted for strict structured outputs.

    Every object lists all properties as required and forbids extras; titles/defaults dropped.
    """
    raw = copy.deepcopy(schema.model_json_schema())

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            # Strip schema annotations only; a *property* named "title" lives under "properties".
            node.pop("title", None)
            node.pop("default", None)
            if "const" in node:
                node["enum"] = [node.pop("const")]
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"].keys())
                node["additionalProperties"] = False
                for prop_schema in node["properties"].values():
                    walk(prop_schema)
            for key, value in node.items():
                if key != "properties":
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(raw)
    return raw


def _strip_fences(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


class OpenRouterClient:
    provider = config.PROVIDER
    base_url = config.OPENROUTER_BASE_URL

    def __init__(self, *, api_key: str, model: str, sdk_client: Any = None, timeout: float = 600.0,
                 max_tokens: int = 32000):
        if not api_key:
            raise LLMError("missing_key", f"Set {config.KEY_VAR} in {config.ENV_PATH}")
        self.model = model
        self.max_tokens = max_tokens
        self.last_metadata: dict = {}
        self.request_count = 0
        if sdk_client is None:
            from openai import OpenAI

            sdk_client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout, max_retries=1)
        self._sdk = sdk_client

    def build_request(self, *, task: str, payload: dict, schema: type[BaseModel]) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": payload["system"]},
                {"role": "user", "content": payload["user"]},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "strict": True, "schema": strict_json_schema(schema)},
            },
            "max_tokens": payload.get("max_tokens", self.max_tokens),
            # Reject providers that would silently drop response_format instead of honouring it.
            "extra_body": {"provider": {"require_parameters": True}},
        }

    def generate(self, *, task: str, payload: dict, schema: type[BaseModel]) -> BaseModel:
        import openai

        request = self.build_request(task=task, payload=payload, schema=schema)
        self.request_count += 1
        started = time.monotonic()
        self.last_metadata = {"task": task, "provider": self.provider, "base_url": self.base_url,
                              "requested_model": self.model}
        try:
            response = self._sdk.chat.completions.create(**request)
        except openai.AuthenticationError as exc:
            raise LLMError("auth", f"OpenRouter rejected the key (401). {exc.message}") from exc
        except openai.PermissionDeniedError as exc:
            raise LLMError("auth", f"OpenRouter denied the request (403). {exc.message}") from exc
        except openai.APITimeoutError as exc:
            raise LLMError("timeout", "OpenRouter request timed out.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("request_failed", f"Could not reach OpenRouter: {exc}") from exc
        except openai.APIStatusError as exc:
            status = exc.status_code
            message = str(exc.message)
            if status == 402:
                raise LLMError("credits", f"OpenRouter reports insufficient credits (402). {message}") from exc
            lowered = message.lower()
            if status in (400, 404) and any(k in lowered for k in ("schema", "response_format", "structured",
                                                                   "no endpoints", "parameter")):
                raise LLMError("unsupported_schema",
                               f"Model {self.model} did not accept the structured-output request ({status}). "
                               f"{message}") from exc
            raise LLMError("request_failed", f"OpenRouter error {status}: {message}") from exc

        elapsed = round(time.monotonic() - started, 1)
        usage = getattr(response, "usage", None)
        self.last_metadata.update({
            "returned_model": getattr(response, "model", None),
            "upstream_provider": getattr(response, "provider", None),
            "response_id": getattr(response, "id", None),
            "latency_s": elapsed,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
        })
        if not getattr(response, "choices", None):
            raise LLMError("malformed", "Response contained no choices.")
        choice = response.choices[0]
        self.last_metadata["finish_reason"] = choice.finish_reason
        message = choice.message
        if getattr(message, "refusal", None):
            raise LLMError("refusal", f"Model refused: {message.refusal}")
        if choice.finish_reason == "length":
            raise LLMError("malformed", "Output was truncated at the token limit; result discarded.")
        content = message.content
        if not content:
            raise LLMError("malformed", "Response had no content.")
        try:
            data = json.loads(_strip_fences(content))
        except json.JSONDecodeError as exc:
            raise LLMError("malformed", f"Response was not valid JSON: {exc}", raw=content) from exc
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMError("malformed", f"Response did not match {schema.__name__}: "
                                        f"{exc.error_count()} errors; first: {exc.errors()[0]['msg']} "
                                        f"at {exc.errors()[0]['loc']}", raw=content) from exc


def client_from_env() -> OpenRouterClient | None:
    """Build the configured client, or None when no key is configured. Makes no request."""
    settings = config.get_settings()
    if not settings.has_key:
        return None
    return OpenRouterClient(api_key=config.get_api_key(), model=settings.model)
