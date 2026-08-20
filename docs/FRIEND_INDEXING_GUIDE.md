# Indexing Guide for Contributors

Hi — you've been asked to help build part of a large search index for HH
Goa 2026 Task 2. This document has everything you need. No other context
is required. Setup takes about 15 minutes. After that, the indexing runs
by itself for a long time (likely several hours) — you don't need to
watch it the whole time.

If anything here doesn't match what you actually see on your screen, stop
and ask rather than guessing. A quick question now is much cheaper than
redoing hours of work later.

## Step 0: Before you start

**Tell the team which language code(s) you're picking, and wait for a
thumbs-up before you actually start running anything.** This is the most
important step. If two people accidentally pick the same language, that
work is wasted — we need to know who's doing what so nothing overlaps.
The list of available codes is in Step 4 below.

**This works on Windows and Mac.** The script automatically detects what
your computer has:
- An NVIDIA graphics card → uses it (fastest)
- An Apple Silicon Mac (M1/M2/M3/M4 chip) → uses its built-in GPU (fastest)
- Neither of the above → uses your regular processor (CPU) instead. This
  still works completely fine, just slower. There's nothing to configure —
  the script figures this out on its own when it starts.

You don't need to know which of these applies to you — just run the
setup steps below and the first few lines of output will tell you what
was detected.

You'll also want:
- Your computer plugged into power for the whole run.
- A few hours where you don't need to shut it down. It's fine to let the
  screen lock or step away — just don't put it to sleep or shut it off
  (see "keep it awake" in Step 6).
- Free disk space — see Step 5.

## Step 1: Get the code

**On Mac**: open the Terminal app (search for "Terminal" with Spotlight,
the magnifying glass in your menu bar).
**On Windows**: open PowerShell (search for "PowerShell" in the Start menu).

Then paste these commands one at a time:

```bash
git clone https://github.com/arvus3005/geeks.git
cd geeks
git checkout feat/self-hosted-hybrid-retrieval
```

If `git` isn't installed, your system will likely prompt you to install it
the first time — say yes. On Windows, if that doesn't happen, download it
from [git-scm.com](https://git-scm.com/download/win) first.

## Step 2: Install `uv` (a tool this project uses)

**On Mac/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows (in PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal afterward so it takes effect.

## Step 3: Install the project's dependencies

If you have an NVIDIA GPU or an Apple Silicon Mac and want to use it:
```bash
uv sync --extra gpu-index
```

If you're not sure, or you know you're on CPU only, this also works fine
(smaller download, and the script will use the slower CPU path):
```bash
uv sync
```

Either is fine to start with — if you picked wrong, just re-run whichever
command you needed later, nothing breaks.

## Step 4: Which language(s) are you doing?

Pick from the list below (anything except `hi` and `bn`, which are
already being handled elsewhere) and **message the team before you start**
(see Step 0). 2-3 languages per person is reasonable. Every language is
roughly similar in size to index — more on why at the bottom of this doc.

| Code | Language | Has training data | Has validation data |
|---|---|---|---|
| `as` | Assamese | yes | yes |
| `gu` | Gujarati | yes | yes |
| `kn` | Kannada | yes | yes |
| `ml` | Malayalam | yes | yes |
| `mr` | Marathi | yes | yes |
| `ne` | Nepali | yes | yes |
| `or` | Odia | yes | yes |
| `pa` | Punjabi | yes | yes |
| `sa` | Sanskrit | yes | yes |
| `ta` | Tamil | yes | yes |
| `ur` | Urdu | yes | yes |
| `te` | Telugu | **no** (missing from the source data — not our bug) | yes |

If you pick `te` (Telugu), expect noticeably less output than the others —
that's expected, not something wrong with your setup.

## Step 5: Disk space and memory

**Disk**: budget roughly **120GB for your first language, plus ~60GB for
each additional one** if you run them together in one command (the shared
English portion only gets stored once per run, which is why extra
languages cost less than the first). So: 1 language ≈ 120GB, 2 ≈ 180GB,
3 ≈ 240GB. This is measured from a real run, not a guess, but still budget
some extra margin. Check your free space first:

**Mac/Linux:**
```bash
df -h /
```
**Windows (PowerShell):**
```powershell
Get-PSDrive C
```

If you're short on space, tell us before starting rather than after — we
can adjust how many languages you take on.

**Memory (RAM)**: the script saves progress in chunks so it doesn't use
unlimited memory, but each chunk is held fully in memory while it's being
built. If your computer has 8GB or 16GB of RAM (common on lower-end
laptops) rather than 24GB+, flag this to us before starting — we may need
to give you a smaller chunk-size setting. Check yours:
- **Mac**: Apple menu → About This Mac → "Memory"
- **Windows**: Settings → System → About → "Installed RAM"

## Step 6: Run it

Say you're assigned Gujarati and Tamil:

**Mac/Linux** (stays awake automatically while running):
```bash
caffeinate -i uv run python -m scripts.build_full_local_index --configs gu ta
```

**Windows:**
```powershell
uv run python -m scripts.build_full_local_index --configs gu ta
```
On Windows, also go to Settings → System → Power & battery, and turn off
automatic sleep for as long as you expect this to run — otherwise Windows
may sleep the machine partway through.

Swap `gu ta` for whichever codes you were assigned.

**Do not add a `--max-rows-per-config` option.** If you see that flag
mentioned elsewhere in this project, it's for quick internal tests only —
leaving it off is what gets you the real, complete data we need.

### What normal output looks like

Right at the start, one line tells you what was detected:
```
INFO Auto-detected device: cuda
```
(or `mps`, or `cpu` — whichever your computer has). That's just
informational, no action needed.

Then, every couple thousand rows, a progress line:
```
INFO gu/train: 2000 source rows, 32784 passages indexed so far (47.0s elapsed, 697.0 passages/sec)
```
Speed varies a lot by hardware — there's no single "correct" number, just
watch that it's steadily climbing and not stuck.

You'll also see lines mentioning "segment" and "Finalized segment" — the
script periodically saves a safe checkpoint to disk. This is normal and
means it's working correctly, not a problem.

### It's safe to interrupt and resume

If you need to close the laptop, it crashes, or you stop it for any
reason: that's fine. Run the **exact same command again** to continue —
it automatically resumes from the last safe checkpoint instead of
starting over. You'll lose at most a few minutes of progress, never more.

### How long will this take?

We don't have solid numbers yet for most hardware — watch your own logged
speed for the first few minutes and use that as your best guide. As a
rough reference point, one Apple Silicon Mac (M4 Pro) processed about
500-700 passages per second; a language with training and validation data
combined has roughly 1.5-2 million passages, so do the math with your own
observed rate for a real estimate.

## Step 7: When it's finished

You'll see a line starting with `Done.`, and the program will end on its
own (return you to a normal prompt).

**Your results will be split across many folders**, not one — this is
intentional (it's how the script keeps memory use safe), so you'll see
something like `artifacts/full_local_index/gu_train_segment_0000/`,
`gu_train_segment_0001/`, and so on, one per chunk of progress. That's
correct — don't worry that it looks fragmented.

Then run this to prepare everything for sending back:

**Mac/Linux:**
```bash
for dir in artifacts/full_local_index/*/; do
  uv run python -m scripts.export_local_index_vectors "$dir"
done
```

**Windows (PowerShell):**
```powershell
Get-ChildItem artifacts/full_local_index -Directory | ForEach-Object {
  uv run python -m scripts.export_local_index_vectors $_.FullName
}
```

## Step 8: Send your results back

Check the total size:
```bash
du -sh artifacts/full_local_index/
```
(Windows: `Get-ChildItem artifacts/full_local_index -Recurse | Measure-Object -Property Length -Sum`)

Compress the whole folder into one file:

**Mac/Linux:**
```bash
cd artifacts
tar -czf full_local_index_<yourname>_<languages>.tar.gz full_local_index/
# example: full_local_index_prasun_gu_ta.tar.gz
```

**Windows (PowerShell):**
```powershell
cd artifacts
Compress-Archive -Path full_local_index -DestinationPath full_local_index_<yourname>_<languages>.zip
```

**Always include your name and language codes in the filename** — with
multiple people sending files, an unlabeled file is impossible to identify.

How to send it:
- **In person together**: a shared drive, cable transfer, or local network
  transfer is usually much faster than uploading over shared WiFi.
- **Remote**: Google Drive, Dropbox, or WeTransfer all handle files this
  size fine — just budget extra time for the upload.

## Why this ends up smaller than it looks

Every language in this dataset is a translation of the *same* original
set of English questions and passages — it's one shared dataset translated
many ways, not many separate ones. So the English portion of your results
will almost certainly overlap with what everyone else produces too. That's
expected — when everything gets combined at the end, duplicate English is
automatically merged down to one copy. You don't need to do anything
differently because of this; just send us everything your run produces.

## If something looks wrong

- **Script says it can't find a GPU when you're sure you have one**: it
  will still work — it just falls back to CPU (slower, not broken). Tell
  us anyway so we can look into why it wasn't detected.
- **A download fails, times out, or seems stuck**: the dataset files are a
  few GB each, so a slow connection can make this take a while or
  occasionally fail partway. Run the exact same command again — it
  resumes downloads instead of starting over.
- **Anything else you're not sure about**: don't guess or try to force
  past it. Send us the exact error message (copy-paste the text) and
  which step you were on — that's much faster for us to help with than a
  description of what happened.
