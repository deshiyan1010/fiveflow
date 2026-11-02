# fiveflow

A real-time speech-to-text overlay for macOS. Hold a hotkey, speak, release — your words appear wherever your cursor is. Built on Whisper for transcription and Gemma for punctuation correction and voice commands.

> **Platform support:** macOS 14+ on Apple silicon only. Windows and Linux support coming soon.

## Widget states

A floating pill widget lives at the bottom of your screen. Drag it anywhere. It stays out of the way until you need it.

| State | Appearance |
|-------|-----------|
| Idle | Small black pill with gray outline |
| Hover | Pill expands smoothly, shows "History" label |
| Recording | Animated bars that scale with mic level |
| Processing | Dark pill with status label (Transcribing / Formatting / Thinking…) |

## Features

- **Transcribe mode** — Hold `Fn`, speak, release. Whisper transcribes, Gemma fixes punctuation and stitches sentences together naturally, result is pasted at your cursor.
- **Command mode** — Hold `Shift+Fn`, speak a command, release. If you have text selected, Gemma applies your command to it (e.g. "make this formal", "translate to French", "convert to bullet points"). If nothing is selected, Gemma generates content freely from the command.
- **History panel** — Hover over the idle pill, click to open a history panel showing all past transcriptions and command exchanges. Each entry has a copy (⧉) button. Command entries show the voice input and output separately with colour-coded badges.
- **Draggable pill** — Click and drag the pill to reposition it anywhere on screen.
- **Parallel model loading** — Whisper (MPS) and Gemma (MLX) load simultaneously at startup.
- **Clipboard preservation** — Keeps images, rich text, and multiple clipboard items intact during copy/paste. A newer copy made during processing is preserved.
- **Assignable hotkey** — Use “Assign Key” in History to choose a trigger; Escape cancels assignment.
- **Safe updates** — Updates fast-forward clean checkouts and preserve local edits/commits. Shell setup preserves unrelated configuration and symlinked dotfiles.
- **Fully transparent overlay** — Click-through everywhere except the pill itself; never steals focus.
- **Stuck-recording protection** — Recovers both macOS event-tap disable notifications, checks hardware key state every 50 ms for dropped releases, and limits a capture to 120 seconds. Audio-device errors reset the controller so the next recording still works.
- **Processing recovery** — Recordings are processed one at a time. Whisper requests also time out after 120 seconds; a stalled native call stays isolated until it finishes or the app restarts. Formatting failures or responses that alter/drop words preserve the raw Whisper transcript; formatter requests time out after 120 seconds instead of waiting forever. Wait for the pill to return to idle before recording again.

## Command capabilities

Gemma handles a wide range of voice commands in command mode:

| Category | Example commands |
|----------|-----------------|
| Tone & style | "make this formal", "make this casual", "make this assertive" |
| Editing | "fix the grammar", "fix spelling", "improve clarity" |
| Length | "summarise this", "expand this", "make it shorter" |
| Format | "convert to bullet points", "convert to a numbered list", "make it a table" |
| Translation | "translate to French", "translate to Spanish" |
| Code | "explain this code", "add comments", "refactor this", "fix the bug" |
| Extraction | "extract the action items", "list the key points", "extract all dates" |
| Generation | "write a follow-up email", "draft a summary" (no selection needed) |

## Requirements

- macOS 14 or later on Apple silicon (required by [MLX](https://ml-explore.github.io/mlx/build/html/install.html))
- Python 3.13
- Accessibility permission (`System Settings → Privacy & Security → Accessibility`)
- Microphone permission (prompted on first recording)

## Installation

Run this one-liner in your terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/deshiyan1010/fiveflow/master/install.sh | bash
```

This will:
1. Clone the repo to `~/.fiveflow`
2. Create a Python 3.13 virtual environment and install all dependencies
3. Add a `fiveflow` command to your shell (`~/.zshrc` / `~/.bash_profile`)

Then reload your shell and you're done:

```bash
source ~/.zshrc   # or ~/.bash_profile if using bash
fiveflow
```

The install script registers three commands in your shell:

| Command | What it does |
|---------|-------------|
| `fiveflow` | Start the app |
| `fiveflow-update` | Pull the latest version and update dependencies |
| `fiveflow-remove` | Completely uninstall — deletes `~/.fiveflow` and removes all shell commands |

On first run, if the models are not yet cached, you will be prompted for a Hugging Face token. A token is optional but significantly speeds up the download.

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Grant Accessibility permission when prompted, then wait for the pill to show `idle` (~30–60 s for first-time model loading).

## Manual setup (alternative)

```bash
git clone https://github.com/deshiyan1010/fiveflow.git
cd fiveflow
python3.13 -m venv stt_env
source stt_env/bin/activate
pip install -r requirements.txt
stt_env/bin/python stt_pipeline.py
```

## Hotkeys

| Hotkey | Mode |
|--------|------|
| Hold `Fn` → release | Transcribe and paste |
| Hold `Shift+Fn` → release | Command mode (acts on selected text or generates freely) |

## Models

| Model | Purpose | Runtime |
|-------|---------|---------|
| `openai/whisper-large-v3-turbo` | Speech recognition | PyTorch / MPS |
| `mlx-community/gemma-4-e2b-it-4bit` | Text correction & commands | MLX |

## Architecture

`stt_pipeline.py` starts the `fiveflow` package — no build step. Recording and
model orchestration live in `fiveflow/pipeline.py`.

```
Main thread          NSApplication run loop, CGEventTap callbacks
_load_whisper        Loads Whisper on startup (daemon thread)
_gemma_worker        Loads Gemma and handles all inference — MLX GPU
                     streams are thread-local so load and infer stay
                     on this one thread
_recording_controller Serializes stream start/abort/close; survives device errors
_process_audio       Grabs selection after capture stops; resets UI on errors
transcribe_and_paste Whisper → Gemma punctuation fix → paste
command_and_paste    Whisper → Gemma command execution → paste
_watch_trigger_release Re-enables the tap, recovers releases, enforces capture limit
```

## Regression tests

```bash
stt_env/bin/python -m unittest discover -s tests -v
```

Tests replace macOS, microphone, and model I/O with fakes; only NumPy is needed.
They cover dropped/coalesced release events, disabled taps, concurrent trigger
updates, quick/empty recordings, device errors, command-mode selection failures,
model loading/inference failures and timeouts, rich clipboard restoration,
selection detection, safe updates, terminal prompts, installation, and UI geometry. A physical Fn/microphone check on macOS is still
needed to verify device-specific behavior.

## Recovery behavior

Model-load failures appear in the pill with a Quit action; see the terminal for
the cause. Initial loading has a 30-minute deadline to allow first-time downloads.
Whisper and Gemma inference each have a 120-second deadline. A native GPU call
cannot safely be killed from Python: after a timeout, that model rejects further
requests until the call finishes or Fiveflow is restarted. Dictation falls back
to the raw Whisper transcript when formatting fails.

The update commands refuse dirty or divergent checkouts. Commit or stash local
work before retrying; updates no longer discard files with a hard reset.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
