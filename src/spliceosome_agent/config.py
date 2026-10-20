"""Non-secret model profiles; configuration never contains API key values."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from platformdirs import user_config_dir
from pydantic import Field, model_validator

from .io import load_document, write_json
from .models import Contract


class ModelProfile(Contract):
    name: str = Field(min_length=1)
    base_url: str
    model: str = Field(min_length=1)
    kind: str = "remote"
    api_key_env: str | None = "OPENAI_API_KEY"
    timeout: float = Field(default=120, gt=0)
    max_tokens: int = Field(default=4000, ge=100)
    format_mode: str = "auto"

    @model_validator(mode="after")
    def valid_endpoint(self):
        url = urlparse(self.base_url)
        if url.username or url.password or url.query or url.fragment or not url.hostname:
            raise ValueError("Use a base URL without credentials, query, or fragment")
        local = url.hostname in {"localhost", "127.0.0.1", "::1"}
        if self.kind not in {"remote", "local"}:
            raise ValueError("kind must be remote or local")
        if self.kind == "local" and not local:
            raise ValueError("Local profiles must use a loopback address")
        if url.scheme != "https" and not (local and url.scheme == "http"):
            raise ValueError("Remote endpoints require HTTPS")
        if self.format_mode not in {"auto", "schema", "json", "plain"}:
            raise ValueError("format_mode must be auto/schema/json/plain")
        if self.api_key_env and not self.api_key_env.replace("_", "").isalnum():
            raise ValueError("Invalid environment variable name")
        return self


class ModelConfig(Contract):
    default: str | None = None
    profiles: dict[str, ModelProfile] = Field(default_factory=dict)
    roles: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def references_exist(self):
        for name in [self.default, *self.roles.values()]:
            if name is not None and name not in self.profiles:
                raise ValueError(f"Unknown model profile: {name}")
        return self


PRESETS = {
    "openai": {"base_url": "https://api.openai.com/v1", "kind": "remote", "api_key_env": "OPENAI_API_KEY"},
    "ollama": {"base_url": "http://127.0.0.1:11434/v1", "kind": "local", "api_key_env": None},
    "lmstudio": {"base_url": "http://127.0.0.1:1234/v1", "kind": "local", "api_key_env": None},
    "vllm": {"base_url": "http://127.0.0.1:8000/v1", "kind": "local", "api_key_env": None},
}


def config_path(explicit=None):
    return Path(explicit or os.environ.get("SPLICEOSOME_AGENT_CONFIG") or
                Path(user_config_dir("spliceosome-agent")) / "models.json").expanduser().resolve()


def load_config(path=None):
    path = config_path(path)
    return ModelConfig.model_validate(load_document(path)) if path.exists() else ModelConfig()


def profile_ready(profile):
    """A remote profile needs a usable API key; a local profile only needs an endpoint."""
    from .credentials import resolve_api_key
    if profile.kind == "local":
        return True
    return bool(resolve_api_key(profile))


def readiness(config):
    """Return (ready, profile_or_None, reason) for the default model profile."""
    name = config.default
    if not name or name not in config.profiles:
        return False, None, "No default model profile is configured"
    profile = config.profiles[name]
    if profile.kind == "remote":
        from .credentials import resolve_api_key
        if not resolve_api_key(profile):
            return False, profile, f"No API key for '{name}'; add one or set {profile.api_key_env or 'an API key env var'}"
    return True, profile, ""


def config_ready(config):
    return readiness(config)[0]


def save_config(config, path=None):
    config = ModelConfig.model_validate(config.model_dump())
    path = config_path(path)
    write_json(path, config)
    path.chmod(0o600)
    return path
