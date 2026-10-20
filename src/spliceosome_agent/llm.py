"""OpenAI-compatible JSON transport with bounded attempts and outbound redaction."""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import httpx

from .credentials import resolve_api_key
from .io import object_hash, read_json, write_json


class ModelUnavailable(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class PrivacyGuard:
    def __init__(self, identifiers=(), secrets=()):
        self.identifiers = sorted({str(x) for x in identifiers if str(x)}, key=len, reverse=True)
        self.secrets = [x for x in secrets if x]

    def clean(self, text):
        text = str(text)
        for secret in self.secrets:
            text = text.replace(secret, "[SECRET]")
        text = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{8,}|Bearer\s+\S+)", "[SECRET]", text)
        for identifier in self.identifiers:
            text = re.sub(r"(?<![\w])" + re.escape(identifier) + r"(?![\w])", "[SUBJECT]", text)
        return text

    def clean_value(self, value):
        if isinstance(value, str):
            return self.clean(value)
        if isinstance(value, dict):
            return {self.clean(str(key)): self.clean_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean_value(item) for item in value]
        return value


def contract_hint(contract):
    """Exact top-level reply fields for the system prompt.

    Some OpenAI-compatible gateways accept ``response_format: json_schema`` but do
    not enforce it, so the field names must also be present in the prompt itself.
    """
    schema = contract.model_json_schema()
    required = set(schema.get("required", []))
    parts = []
    for name, spec in schema.get("properties", {}).items():
        if "enum" in spec:
            kind = "|".join(str(value) for value in spec["enum"])
        elif "anyOf" in spec:
            kinds = []
            for option in spec["anyOf"]:
                if "enum" in option:
                    kinds.append("|".join(str(value) for value in option["enum"]))
                elif option.get("type") == "array":
                    kinds.append("array")
                else:
                    kinds.append(option.get("type") or option.get("$ref", "object").split("/")[-1].lower())
            kind = "|".join(dict.fromkeys(kinds))
        elif spec.get("type") == "array":
            kind = "array"
        else:
            kind = spec.get("type", "object")
        parts.append(f"{name}: {kind}" + ("" if name in required else " (optional)"))
    return ("Reply with exactly one JSON object using these top-level fields and no others: "
            + "; ".join(parts) + ".")


def failure_note(exc):
    """Short, content-free failure location for audit records and user-facing errors."""
    errors = exc.errors() if hasattr(exc, "errors") else None
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.get("loc", ())) or "(root)"
        return f"{type(exc).__name__}: {location}: {first.get('msg', first.get('type', ''))}"
    return type(exc).__name__


class ModelClient:
    def __init__(self, config, audit_dir, token_budget=100_000, identifiers=(), transport=None):
        self.config = config
        self.audit_dir = Path(audit_dir)
        self.budget = token_budget
        self.lock = threading.RLock()
        self.transport = transport
        secrets = []
        for p in config.profiles.values():
            value = resolve_api_key(p)
            if value:
                secrets.append(value)
        self.guard = PrivacyGuard(identifiers, secrets)
        self.metrics_path = self.audit_dir / "usage.json"
        self.metrics = read_json(self.metrics_path) if self.metrics_path.exists() else {"tokens": 0, "requests": 0}

    def profile(self, role):
        name = self.config.roles.get(role, self.config.default)
        if not name or name not in self.config.profiles:
            raise ModelUnavailable("No configured model; natural-language planning unavailable")
        profile = self.config.profiles[name]
        if profile.kind == "remote" and not resolve_api_key(profile):
            raise ModelUnavailable(f"No API key for '{name}'; save one in the OS keychain or set {profile.api_key_env or 'an API key env var'}")
        return profile

    def available(self, role="Orchestrator"):
        try:
            self.profile(role)
            return True
        except ModelUnavailable:
            return False

    def json(self, role, system, payload, contract):
        # One lock spans budget reservation and response accounting across parallel roles.
        with self.lock:
            profile = self.profile(role)
            safe_payload = json.dumps(self.guard.clean_value(payload), ensure_ascii=False, allow_nan=False)
            messages = [{"role": "system", "content": system + "\n" + contract_hint(contract)},
                        {"role": "user", "content": safe_payload}]
            mode = profile.format_mode
            last = None
            for attempt in range(3):
                remaining = self.budget - self.metrics["tokens"]
                estimate = max(1, sum(len(m["content"].encode()) for m in messages) // 3)
                if remaining <= estimate + 100:
                    raise BudgetExceeded("Model token budget exhausted; partial artifacts retained")
                limit = min(profile.max_tokens, remaining - estimate)
                body = {"model": profile.model, "messages": messages, "max_tokens": limit, "stream": False}
                if mode in {"auto", "schema"}:
                    body["response_format"] = {"type": "json_schema", "json_schema": {
                        "name": contract.__name__, "strict": True, "schema": contract.model_json_schema()}}
                elif mode == "json":
                    body["response_format"] = {"type": "json_object"}
                headers = {"Content-Type": "application/json"}
                api_key = resolve_api_key(profile)
                if api_key:
                    headers["Authorization"] = "Bearer " + api_key
                self.metrics["requests"] += 1
                sequence = self.metrics["requests"]
                try:
                    with httpx.Client(timeout=profile.timeout, transport=self.transport,
                                      follow_redirects=False, trust_env=profile.kind != "local") as client:
                        response = client.post(profile.base_url.rstrip("/") + "/chat/completions", json=body, headers=headers)
                    if response.status_code == 400 and profile.format_mode == "auto" and mode != "plain" and any(
                            word in response.text.lower() for word in ["response_format", "schema", "unsupported", "json_object"]):
                        mode = "json" if mode in {"schema", "auto"} else "plain"
                        self._record(sequence, role, body, {"status": "format_fallback", "http_status": 400})
                        continue
                    if response.status_code in {429, 500, 502, 503, 504}:
                        raise httpx.TransportError(f"Transient HTTP {response.status_code}")
                    if response.status_code >= 400:
                        raise ModelUnavailable(f"Model HTTP {response.status_code}; check endpoint, model and credentials")
                    try:
                        data = response.json()
                    except ValueError:
                        self.metrics["tokens"] += estimate + limit
                        self._save_metrics()
                        raise
                    usage = data.get("usage", {})
                    charged = int(usage.get("total_tokens") or estimate + limit)
                    self.metrics["tokens"] += charged
                    self._save_metrics()
                    if self.metrics["tokens"] > self.budget:
                        raise BudgetExceeded("Provider reported usage exceeding the approved token budget")
                    choice = data["choices"][0]
                    if choice.get("finish_reason") == "length":
                        raise ValueError("Model output truncated")
                    content = choice["message"]["content"] or ""
                    if content.startswith("```"):
                        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
                    value = contract.model_validate(self.guard.clean_value(json.loads(content)))
                    self._record(sequence, role, body, {"status": "validated", "response": value.model_dump(), "usage": usage})
                    return value
                except (BudgetExceeded, ModelUnavailable):
                    self._save_metrics()
                    raise
                except (httpx.TransportError, ValueError, KeyError, IndexError, TypeError) as exc:
                    last = failure_note(exc)
                    self._record(sequence, role, body, {"status": "failed", "error_type": type(exc).__name__, "error": last})
                    if isinstance(exc, httpx.TransportError):
                        time.sleep(min(2 ** attempt, 4))
                    else:
                        messages.append({"role": "user", "content": "Return a complete JSON object matching the supplied "
                                         "schema; no markdown or extra fields. " + contract_hint(contract)})
            raise ModelUnavailable(f"Model response failed validation/transport after 3 attempts ({last})")

    def _save_metrics(self):
        write_json(self.metrics_path, self.metrics)

    def _record(self, seq, role, body, result):
        self._save_metrics()
        # Never persist request bodies, credentials or raw clinical inputs.
        record = {"role": role, "request_hash": object_hash(body), **result}
        write_json(self.audit_dir / f"{seq:05d}_{role}.json", record)
