import re
from dataclasses import dataclass

GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|namaste|namaskar|नमस्ते|হ্যালো|হেলো|bonjour|ola|hola|salut|ciao)\b",
    re.IGNORECASE,
)
CURRENT_STATE_PATTERNS = re.compile(
    r"\b(weather|temperature|stock price|share price|cricket score|latest news|current|today|right now|live score)\b",
    re.IGNORECASE,
)
UNSAFE_PATTERNS = re.compile(
    r"\b(bomb|kill|hack|exploit|malware|ransomware|cvv|ssn|credit card|password|api key|private key)\b",
    re.IGNORECASE,
)
CREDENTIAL_PATTERNS = re.compile(
    r"(password|passwd|secret|api[_\-]?key|token|bearer|ssh[_\-]?key)",
    re.IGNORECASE,
)
INJECTION_PATTERNS = re.compile(
    r"(ignore previous|forget instructions|you are now|disregard|jailbreak|DAN mode)",
    re.IGNORECASE,
)


@dataclass
class GuardResult:
    allowed: bool
    reason_code: str
    message: str


def check_input(text: str, max_chars: int = 512) -> GuardResult:
    if not text or not text.strip():
        return GuardResult(False, "empty_input", "Input is empty.")
    if len(text) > max_chars:
        return GuardResult(False, "input_too_long", f"Input exceeds {max_chars} chars.")
    stripped = text.strip()
    if len(set(stripped)) <= 3 and len(stripped) > 10:
        return GuardResult(False, "degenerate_input", "Input appears repetitive or gibberish.")
    if UNSAFE_PATTERNS.search(text):
        return GuardResult(False, "unsafe", "Request appears unsafe.")
    if CREDENTIAL_PATTERNS.search(text):
        return GuardResult(False, "credential_request", "Request asks for credentials.")
    if INJECTION_PATTERNS.search(text):
        return GuardResult(False, "prompt_injection", "Prompt injection attempt detected.")
    if GREETING_PATTERNS.match(stripped):
        return GuardResult(False, "greeting", "Greeting — no retrieval needed.")
    if CURRENT_STATE_PATTERNS.search(text):
        return GuardResult(False, "current_state", "Cannot answer live/current-state questions from static corpus.")
    return GuardResult(True, "allowed", "")
