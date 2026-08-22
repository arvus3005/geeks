"""Optional config the eval harness (rag-local-eval-loop) reads
defensively via getattr() with its own fallback if a name is missing —
see TARGET_INTERFACE.md in that suite's repo. Not used by this project's
own code; exists only for the eval harness's report cosmetics and a
concurrency safety clamp.

eval/judge.py's own docstring assumes importing this module has the side
effect of loading .env (its exact words: "this suite's original target
project does [load .env], via python-dotenv") -- this project's own
settings (hhgoa_rag.config.settings.Settings) load .env too, but only into
its own declared pydantic fields, never into os.environ, so OPENAI_API_KEY/
ANTHROPIC_API_KEY sitting in .env were never actually visible to the judge
even with a real key present there (confirmed 2026-08-22: two full eval
runs after adding OPENAI_API_KEY to .env still reported judge SKIPPED).
This is the one place that assumption actually needs to hold.
"""
from dotenv import load_dotenv

load_dotenv()

# GENERATION_BACKEND is deliberately left unset: it only exists to
# auto-clamp --workers to 1 for a shared local GPU/LLM model under
# concurrent-call contention risk. This project's generator is in-process
# extractive answering over ONNX Runtime (thread-safe for concurrent
# inference by design) + stateless SentencePiece encode() -- no shared
# mutable state a concurrent eval run would corrupt, so the eval suite can
# run with its normal multi-worker default instead of being forced to 1.

# This project's real end-to-end latency target (task spec + CLAUDE.md),
# not the eval suite's own generic 50ms default.
LATENCY_BUDGET_MS = 200

GENERATION_MODEL = "hhgoa-local-hybrid-extractive-v1"
