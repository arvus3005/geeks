# Indexing Guide for Contributors

You've been asked to help build part of a large multilingual search index
for HH Goa 2026 Task 2. This doc is everything you need — no other context
required. Should take about 15 minutes of setup, then the indexing itself
runs unattended for several hours.

## Before you start — do you have what's needed?

**You need an Apple Silicon Mac** (M1, M2, M3, or M4 — any variant). This
uses the Mac's GPU (via `MPS`/Metal) for speed; it will not work correctly
on Intel Macs, Windows, or Linux. If you don't have Apple Silicon, let the
person who sent you this know — you can still help in other ways, but not
this particular task.

You'll also want:
- At least **20-30GB of free disk space** per language you're assigned
  (see the sizing table below for specifics).
- A few hours where your Mac can stay on, plugged in, and not go to sleep
  (see the "keep it running" note below).

## 1. Get the code

```bash
git clone https://github.com/arvus3005/geeks.git
cd geeks
git checkout feat/self-hosted-hybrid-retrieval
```

## 2. Install `uv` (if you don't have it)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 3. Install dependencies, including the GPU extras

```bash
uv sync --extra gpu-index
```

This installs everything including `torch` (needed for the GPU embedding
path — it's deliberately kept out of the base install since the production
API doesn't use it).

## 4. Which language(s) are you indexing?

You'll be told which 2-letter language code(s) to run — something like
`gu` (Gujarati), `ta` (Tamil), `mr` (Marathi), etc. A natural split is 3
languages per contributor; here's the full list of what's available and
roughly how big each one is (every language shares the same underlying
English passage set, so English isn't counted per-language — see the
"Why this is small" note below):

| Code | Language | Train split | Validation split |
|---|---|---|---|
| `as` | Assamese | yes | yes |
| `bn` | Bengali | yes | yes | (already being done — don't pick this one)
| `gu` | Gujarati | yes | yes |
| `hi` | Hindi | yes | yes | (already being done — don't pick this one)
| `kn` | Kannada | yes | yes |
| `ml` | Malayalam | yes | yes |
| `mr` | Marathi | yes | yes |
| `ne` | Nepali | yes | yes |
| `or` | Odia | yes | yes |
| `pa` | Punjabi | yes | yes |
| `sa` | Sanskrit | yes | yes |
| `ta` | Tamil | yes | yes |
| `te` | Telugu | **no train split** (confirmed missing upstream) | yes |
| `ur` | Urdu | yes | yes |

If you're assigned `te` (Telugu), you'll get much less data than the
others — that's expected, not a bug on your end; the source dataset simply
doesn't have a Telugu training split.

## 5. Run it

Say you're assigned Gujarati, Tamil, and Marathi:

```bash
uv run python -m scripts.build_full_local_index --configs gu ta mr
```

Pass all your assigned languages in one command (space-separated) — don't
run them as separate invocations, since each language's translated
passages plus one shared English pool all need to land in the same output
directory (`artifacts/full_local_index/`).

**Do not pass `--max-rows-per-config`.** That flag is only for quick smoke
tests; leaving it off is what gets you the real, full data this project
needs.

### What you'll see

It logs progress every 2000 source rows, something like:
```
INFO gu/train: 2000 source rows, 32784 passages indexed so far (47.0s elapsed, 697.0 passages/sec)
```
On an M4 Pro this ran at ~700-715 passages/sec sustained. Other Apple
Silicon chips (M1/M2/M3, or non-Pro/Max variants) will likely be somewhat
slower — there's no exact number for your specific hardware yet, but this
is the ballpark to expect.

### Keep it running

This is a multi-hour job (see timing below). Your Mac needs to stay awake
and not sleep the whole time:

```bash
caffeinate -i uv run python -m scripts.build_full_local_index --configs gu ta mr
```

(`caffeinate -i` prevents idle sleep for as long as the command runs — safe
to prefix any of the commands in this doc with it.)

### It's resumable

If it crashes, your laptop restarts, or you need to stop it, just re-run
the exact same command — it picks up from the last completed checkpoint
(every ~8,192 passages) instead of starting over. Progress is saved in
`artifacts/full_index_checkpoints/`.

### Roughly how long will it take?

Nobody has run this exact workload on a non-M4-Pro chip yet, so treat this
as a rough guide, not a promise — check your own logged rate after the
first few progress lines and do the math for your assigned languages.
As a reference point: 3 full languages (train+val) at the M4 Pro's ~700
passages/sec was projected around 6-10 hours. If your Mac is running
noticeably slower or faster than ~700 passages/sec once it's warmed up,
scale that estimate accordingly.

## 6. When it finishes

Two things confirm it's actually done (not just paused):
- The log prints a final `Done: {...}` line with a `manifest.json` summary.
- `artifacts/full_index_checkpoints/` has a `status: "complete"` entry for
  every `(language, split)` pair you ran.

Then export the embedding vectors (needed for merging later — the raw
vectors aren't stored anywhere else in a portable format):

```bash
uv run python -m scripts.export_local_index_vectors artifacts/full_local_index
```

This creates `artifacts/full_local_index/embeddings.npy` alongside the
other output files.

## 7. Send your results back

You need to send back the whole `artifacts/full_local_index/` directory:
`passages.jsonl`, `bm25_tokens.jsonl`, `hnsw.usearch`, `embeddings.npy`,
and `manifest.json`. Depending on your language(s), this could be several
GB to a few tens of GB — check the actual size first:

```bash
du -sh artifacts/full_local_index/
```

If you're physically together at the hackathon venue: **AirDrop is
probably fastest** for this size (faster than uploading over shared venue
WiFi and someone else downloading it). Zip the directory first if AirDrop
struggles with many small files:

```bash
cd artifacts
tar -czf full_local_index_<yourlanguages>.tar.gz full_local_index/
# e.g. full_local_index_gu_ta_mr.tar.gz
```

If you're not together: cloud storage (Google Drive, Dropbox, WeTransfer)
works fine for tens of GB — just budget upload time on top of the indexing
time above.

**Rename the tarball/folder to include your language codes** before
sending — with several people sending files back, an unlabeled
`full_local_index.tar.gz` is not identifiable.

## Why this whole thing is smaller than you'd expect

Every MSMARCO-XI language is a translation of the *same* underlying
English query/passage set — not independent corpora. So your English
passages, whatever language you're assigned, will very likely be
duplicates of what everyone else's runs also produce (this is expected and
correct — the merge step on the receiving end deduplicates it, so it's
only stored once in the final combined index; don't worry about it on your
end, just send everything you produced).

## If something looks wrong

- **"MPS not available on this machine"**: you're not on Apple Silicon, or
  your macOS/PyTorch version doesn't support it. This script won't work
  for you as-is — flag it rather than trying to force it through.
- **A parquet download fails or times out**: the dataset (`ai4bharat/MSMARCO-XI`
  on HuggingFace) is public and needs no login, but large files (~3-4GB
  each) over a slow connection can be flaky. Just re-run the same command;
  HuggingFace's download cache resumes partial downloads.
- **Anything else**: don't try to work around it silently — flag it back
  with the exact error message. This is exactly the kind of thing worth
  a quick message rather than guessing.
