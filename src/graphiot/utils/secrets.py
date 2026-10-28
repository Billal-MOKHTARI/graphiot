"""In-memory secret store: API keys from .env or the GUI, never written anywhere.

Values come from the project's ``.env`` (loaded once, read-only) or from the
GUI's key field, and live only in this process. Configs name the *environment
variable* (``api_key_env``); :meth:`SecretStore.get` resolves it. A logging
filter replaces every known secret value with ``***`` in any log record, and
:func:`redact` does the same for strings about to be exported.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


class SecretStore:
    def __init__(self):
        self._values: dict[str, str] = {}
        self._lock = threading.Lock()

    def load_dotenv(self, path: str | Path | None = None) -> list[str]:
        """Read KEY=VALUE lines from .env into memory (does not modify os.environ).
        Returns the variable names loaded (never the values)."""
        from dotenv import dotenv_values
        path = Path(path) if path else _ROOT / ".env"
        if not path.exists():
            return []
        loaded = []
        for k, v in dotenv_values(path).items():
            if v:
                self.set(k, v)
                loaded.append(k)
        return loaded

    def set(self, name: str, value: str) -> None:
        with self._lock:
            self._values[name] = value

    def clear(self, name: str) -> None:
        with self._lock:
            self._values.pop(name, None)

    def get(self, name: str | None) -> str | None:
        if not name:
            return None
        with self._lock:
            if name in self._values:
                return self._values[name]
        return os.environ.get(name)

    def names(self) -> list[str]:
        """Which variables are available (for the GUI), never the values."""
        with self._lock:
            return sorted(self._values)

    def known_values(self) -> list[str]:
        with self._lock:
            vals = list(self._values.values())
        return [v for v in vals if v and len(v) >= 6]


STORE = SecretStore()


def redact(text: str, store: SecretStore = STORE) -> str:
    for v in store.known_values():
        text = text.replace(v, "***")
    return text


class RedactingFilter(logging.Filter):
    """Attach to handlers: scrubs secret values from every formatted record."""

    def __init__(self, store: SecretStore = STORE):
        super().__init__()
        self.store = store

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        clean = redact(msg, self.store)
        if clean != msg:
            record.msg, record.args = clean, None
        return True
