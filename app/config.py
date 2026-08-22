"""Optional config the eval harness (rag-local-eval-loop) reads
defensively via getattr() with its own fallback if a name is missing —
see TARGET_INTERFACE.md in that suite's repo. Not used by this project's
own code; exists only for the eval harness's report cosmetics and a
concurrency safety clamp.
"""

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
