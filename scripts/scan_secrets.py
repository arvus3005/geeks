#!/usr/bin/env python3
"""Scan tracked source files for likely credential literals.

Prints filenames only — never the matching secret value.
Exits 1 if any credential is found.

Detected patterns:
  - Pinecone Starter keys:    pcsk_...
  - Sarvam API keys:          sarvam-...  / SARVAM_API_KEY=<value>
  - Groq API keys:            gsk_...
  - Gemini/Google API keys:   AIzaSy...  / GOOGLE_API_KEY=<value>
  - ElevenLabs keys:          sk_...  (ElevenLabs specific)
  - Generic key assignments:  API_KEY="..."  TOKEN="..."  SECRET="..."
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Patterns that indicate credential literals.
# Each tuple: (pattern_name, compiled_regex)
# Regex must NOT capture the secret value — only detect its presence.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Pinecone Starter key (pcsk_ prefix, ~40+ chars)
    ("pinecone_key", re.compile(r"pcsk_[A-Za-z0-9]{20,}", re.IGNORECASE)),
    # Groq API key (gsk_ prefix)
    ("groq_key", re.compile(r"gsk_[A-Za-z0-9]{20,}", re.IGNORECASE)),
    # Google / Gemini API key
    ("google_key", re.compile(r"AIzaSy[A-Za-z0-9_-]{25,}", re.IGNORECASE)),
    # ElevenLabs key (sk_ prefix followed by long token, distinct from short npm tokens)
    ("elevenlabs_key", re.compile(r"(?i)elevenlabs[^\n]*sk_[A-Za-z0-9]{20,}")),
    # Sarvam API key
    ("sarvam_key", re.compile(r"sarvam-[A-Za-z0-9_-]{16,}", re.IGNORECASE)),
    # Generic assignment patterns: KEY="value" or KEY=value (non-empty, not placeholder)
    (
        "generic_secret_assignment",
        re.compile(
            r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)"
            r'\s*[=:]\s*["\']?(?!<|your|placeholder|changeme|example|xxx|abc|test|none|false|true)'
            r"[A-Za-z0-9+/=_-]{20,}",
        ),
    ),
]

# File extensions to scan
_SCAN_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".sh",
    ".bash",
    ".env",
    ".md",
    ".txt",
    ".cfg",
    ".ini",
    ".conf",
}

# Paths to skip entirely (directories or files)
_SKIP_PREFIXES = {
    ".venv",
    ".git",
    "reference",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "node_modules",
    "uv.lock",
    ".env",
}


def _should_skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = set(rel.parts)
    return bool(parts & _SKIP_PREFIXES)


def scan_file(path: Path) -> list[str]:
    """Return list of matched pattern names (no secret values)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found = []
    for name, pat in _PATTERNS:
        if pat.search(text):
            found.append(name)
    return found


def main() -> None:
    root = Path(os.getcwd())
    flagged: list[tuple[Path, list[str]]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        # Prune skipped directories in-place
        dirnames[:] = [d for d in dirnames if not _should_skip(dp / d, root)]
        for fname in filenames:
            fpath = dp / fname
            if fpath.suffix not in _SCAN_EXTENSIONS and fpath.name not in {".env", ".env.example"}:
                continue
            if _should_skip(fpath, root):
                continue
            hits = scan_file(fpath)
            if hits:
                flagged.append((fpath, hits))

    if flagged:
        print("CREDENTIAL SCAN: POTENTIAL SECRETS FOUND", file=sys.stderr)
        print("Filenames with matches (values NOT printed):", file=sys.stderr)
        for fpath, patterns in flagged:
            rel = fpath.relative_to(root)
            print(f"  {rel}  [{', '.join(patterns)}]", file=sys.stderr)
        print(
            "\nRemediate before committing. Rotate any exposed credentials immediately.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print("Credential scan: no secrets detected in tracked source files.")
        sys.exit(0)


if __name__ == "__main__":
    main()
