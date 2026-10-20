"""API keys are kept in the OS keychain; configuration files never hold secret values.

Only the *name* of the environment variable is written to models.json. The value is
resolved at runtime from the environment first (explicit user intent) and then from the
operating system credential store (macOS Keychain, Linux/WSL Secret Service).
"""
from __future__ import annotations

import os

SERVICE = "spliceosome-agent"

# Test hook: a keyring-like object with get/set/delete_password. None means auto-detect.
_backend = None


class KeychainUnavailable(RuntimeError):
    """No usable OS credential backend; callers must fall back to an environment variable."""


def set_backend(backend):
    """Inject a keyring-like backend (tests); pass None to restore auto-detection."""
    global _backend
    _backend = backend


def backend():
    if _backend is not None:
        return _backend
    try:
        import keyring
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
        raise KeychainUnavailable(
            "No OS keychain support (install the 'keyring' package) or set the API key environment variable"
        ) from exc
    implementation = keyring.get_keyring()
    if getattr(implementation, "priority", 0) <= 0:
        raise KeychainUnavailable(
            "No usable OS keychain backend was detected; set the API key environment variable instead"
        )
    return keyring


def get_api_key(account):
    """Return the stored key, or None when no entry exists. Raises KeychainUnavailable."""
    if not account:
        return None
    try:
        value = backend().get_password(SERVICE, account)
    except KeychainUnavailable:
        raise
    except Exception:
        # A missing entry, a locked keychain or a backend error all mean "no usable key".
        return None
    return value or None


def set_api_key(account, value):
    if not account or not value:
        raise ValueError("An account name and a non-empty key are required")
    backend().set_password(SERVICE, account, value)


def delete_api_key(account):
    if not account:
        return
    try:
        backend().delete_password(SERVICE, account)
    except Exception:
        return


def resolve_api_key(profile):
    """Environment variable first, then the OS keychain. Never logs the value."""
    if profile.api_key_env:
        value = os.environ.get(profile.api_key_env)
        if value:
            return value
    try:
        return get_api_key(profile.name)
    except KeychainUnavailable:
        return None


def has_api_key(profile):
    return bool(resolve_api_key(profile))


def describe_backend():
    """Short, non-secret description of the credential backend for diagnostics."""
    try:
        implementation = backend()
    except KeychainUnavailable as exc:
        return {"available": False, "detail": str(exc)}
    return {"available": True, "backend": type(implementation).__name__}
