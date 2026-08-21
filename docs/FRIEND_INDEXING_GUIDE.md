# Indexing Guide — For a Friend Helping Out on Windows

This guide is written for **one person, on one Windows PC, indexing one
language**. If you're reading this, someone on the team asked you to help
build part of a large multilingual search index. This document has
everything you need — no other context required, and it's written so
either you or an AI coding assistant (like Claude Code or Copilot) can
follow it directly.

---

## Quick reference (read this first)

| | |
|---|---|
| **What you're doing** | Running one Python script that downloads a language's text data and builds a search index from it |
| **Your OS** | Windows |
| **Time needed (yours)** | ~15 minutes of setup, then it runs by itself for a few hours |
| **Time needed (machine)** | A few hours of continuous running — keep the PC on and plugged in |
| **What you need to tell the team first** | Which language code you're assigned (Step 2) |
| **What you send back at the end** | One compressed file, a few GB to a few dozen GB |
| **Can it be interrupted?** | Yes — safe to stop and resume any time, see Step 6 |

If anything on your screen doesn't match what this guide says, **stop and
ask** rather than guessing — a quick question now is much cheaper than
redoing hours of work later.

### Why this is needed (the actual problem being solved)

The project this supports needs a search index built over an entire
multilingual dataset — not a small sample of it — across many Indian
languages. That's too much data and compute for one machine to finish
alone in the time available. So the work is split: different volunteers
each index one or two languages on their own computers, and all the
results get combined at the end into one final index. Your part is one
language.

---

## Step 1: Install the tools

Open **PowerShell** (search for it in the Start menu).

**1a. Install Git** (skips if already installed):
```powershell
winget install --id Git.Git -e --source winget
```
Close and reopen PowerShell after this finishes.

**1b. Install `uv`** (the tool this project uses to manage Python and
dependencies — you do not need Python installed separately, `uv` handles
that):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Close and reopen PowerShell again after this finishes.

## Step 2: Get the code and pick your language

```powershell
git clone https://github.com/arvus3005/geeks.git
cd geeks
git checkout main
```

**Tell the team which language code you're taking, and wait for a
confirmation before you start running anything.** This matters — if two
people accidentally pick the same language, that work is wasted.

Pick one (or two) from this table — **anything except `hi`, `bn`, `gu`,
`mr`, `ta`**, which are already being handled elsewhere. English is not
a separate pick — every language's data already includes a matching set
of English text, so you don't need to (and can't) pick English on its own.

| Code | Language | Has training data | Has validation data |
|---|---|---|---|
| `as` | Assamese | yes | yes |
| `kn` | Kannada | yes | yes |
| `ml` | Malayalam | yes | yes |
| `ne` | Nepali | yes | yes |
| `or` | Odia | yes | yes |
| `pa` | Punjabi | yes | yes |
| `sa` | Sanskrit | yes | yes |
| `ur` | Urdu | yes | yes |
| `te` | Telugu | **no** (missing from the source data — not a bug, real gap) | yes |

## Step 3: Install project dependencies

**If you have an NVIDIA graphics card** and want to use it (faster):
```powershell
uv sync --extra gpu-index
```

**If you're not sure, or you know you don't have one:**
```powershell
uv sync
```
This still works completely fine, just slower. The script automatically
detects what your PC has (NVIDIA GPU → CPU fallback) — you don't need to
configure anything either way.

## Step 4: Check your free disk space and memory

**Disk** — budget roughly **120GB** for your one assigned language. Check
what you have free:
```powershell
Get-PSDrive C
```
If you're short on space, tell the team before starting, not after.

**Memory (RAM)** — check yours:
```powershell
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
```
(divide the number by 1073741824 to get GB). If you have 8GB or 16GB
rather than 24GB+, flag this before starting — a smaller setting may be
needed.

## Step 5: Run it

Replace `XX` below with your assigned language code (e.g. `kn`):

```powershell
uv run python -m scripts.build_full_local_index --configs XX
```

Also go to **Settings → System → Power & battery** and turn off automatic
sleep for as long as you expect this to run — otherwise Windows may put
the machine to sleep partway through and pause the job.

**Do not add a `--max-rows-per-config` flag.** That's for quick internal
tests only — leaving it off is what gets the real, complete data needed.

### What normal output looks like

Right at the start:
```
INFO Auto-detected device: cuda
```
(or `cpu` if you don't have an NVIDIA GPU — both are fine, just different
speeds.)

Then, repeating progress lines:
```
INFO kn/train: 2000 source rows, 32784 passages indexed so far (47.0s elapsed, 697.0 passages/sec)
```
Speed varies a lot by hardware — just watch that the numbers keep
climbing, not stuck.

You'll also see lines mentioning "segment" and "Finalized segment" — this
is the script periodically saving safe progress to disk. Normal, not a
problem.

## Step 6: It's safe to interrupt and resume

If you need to close the laptop, it crashes, or you stop it for any
reason: that's fine. **Run the exact same command again** to continue —
it automatically picks up from the last safe checkpoint instead of
starting over. You'll lose at most a few minutes of progress, never more.

## Step 7: When it's finished

You'll see a line starting with `Done.`, and PowerShell will return to a
normal prompt.

Your results will be split across many folders (this is intentional, not
a problem) — e.g. `artifacts/full_local_index/kn_train_segment_0000/`,
`kn_train_segment_0001/`, and so on.

Prepare everything for sending back:
```powershell
Get-ChildItem artifacts/full_local_index -Directory | ForEach-Object {
  uv run python -m scripts.export_local_index_vectors $_.FullName
}
```

## Step 8: Send your results back

Check the total size:
```powershell
Get-ChildItem artifacts/full_local_index -Recurse | Measure-Object -Property Length -Sum
```

Compress it into one file:
```powershell
cd artifacts
Compress-Archive -Path full_local_index -DestinationPath full_local_index_<yourname>_<language>.zip
# example: full_local_index_prasun_kn.zip
```

**Always include your name and language code in the filename** — with
multiple people sending files, an unlabeled file can't be identified.

Send it via whatever's fastest available: a shared drive/cable transfer
if you're in person, or Google Drive / Dropbox / WeTransfer if remote
(budget extra time for the upload — these files are large).

## Why your results will be smaller than you might expect

Every language in this dataset is a translation of the *same* original
set of English text — one shared dataset translated many ways, not many
separate ones. So the English portion of your results will overlap with
everyone else's. That's expected — it gets merged down to one copy when
everything is combined at the end. You don't need to do anything
differently because of this; just send everything your run produces.

## If something looks wrong

- **Script can't find a GPU when you're sure you have one**: it still
  works, just falls back to CPU (slower, not broken). Tell the team
  anyway so it can be looked into.
- **A download fails, times out, or seems stuck**: source files are a
  few GB each — a slow connection can make this take a while or fail
  partway. Run the exact same command again — it resumes downloads
  instead of starting over.
- **Anything else you're not sure about**: don't guess or force past it.
  Send the exact error message (copy-paste the text) and which step you
  were on.

---

## Structured summary (for AI agents / automation)

If you're an AI assistant helping run this rather than a human following
it step by step, here is the same guide as an unambiguous checklist:

```yaml
task: index_one_language
target_os: windows
prerequisites:
  - git installed (winget install --id Git.Git -e --source winget)
  - uv installed (irm https://astral.sh/uv/install.ps1 | iex, via PowerShell -ExecutionPolicy ByPass)
  - free disk space >= 120GB on the drive containing the repo
  - language code confirmed with the team, not already in the excluded set
excluded_language_codes: [hi, bn, gu, mr, ta]
available_language_codes: [as, kn, ml, ne, or, pa, sa, ur, te]
setup_commands:
  - git clone https://github.com/arvus3005/geeks.git
  - cd geeks
  - git checkout main
  - uv sync --extra gpu-index   # if NVIDIA GPU present, else: uv sync
run_command_template: "uv run python -m scripts.build_full_local_index --configs {LANGUAGE_CODE}"
forbidden_flags: ["--max-rows-per-config"]
success_signal: "a line starting with 'Done.' appears, process exits 0"
interrupted_run_recovery: "re-run the exact same run_command_template — it resumes from last checkpoint automatically, no data loss beyond a few minutes"
post_run_commands:
  - "for each dir in artifacts/full_local_index/*/: uv run python -m scripts.export_local_index_vectors <dir>"
  - "Compress-Archive -Path artifacts/full_local_index -DestinationPath artifacts/full_local_index_<name>_<language>.zip"
output_naming_convention: "full_local_index_<contributor_name>_<language_code>.zip"
failure_modes:
  gpu_not_detected: "non-fatal, falls back to CPU automatically, notify team"
  download_stuck_or_failed: "re-run the exact same run_command_template, downloads resume"
  unrecognized_error: "do not guess a fix; capture exact error text and current step, report to team"
do_not:
  - do not pick a language already in excluded_language_codes
  - do not add --max-rows-per-config to the real run
  - do not let the machine sleep mid-run
  - do not rename output folders manually before running export_local_index_vectors
```
