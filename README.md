# fiveflow

A real-time speech-to-text overlay for macOS. Hold a hotkey, speak, release — your words appear wherever your cursor is. Built on Whisper for transcription and Gemma for punctuation correction and voice commands.

> **Platform support:** macOS only (Apple Silicon recommended). Windows and Linux support coming soon.

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
- **Fully transparent overlay** — Click-through everywhere except the pill itself; never steals focus.
- **Stuck-recording protection** — CGEventTap is automatically re-enabled if macOS times it out, and a watchdog force-stops any recording stuck for over 2 minutes.

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

- macOS 12 or later (Apple Silicon recommended)
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

Running `fiveflow` again in the future will always start the app. Re-running the install script at any time will update to the latest version.

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

Everything runs in `stt_pipeline.py` — no build step.

```
Main thread          NSApplication run loop, CGEventTap callbacks
_load_whisper        Loads Whisper on startup (daemon thread)
_gemma_worker        Loads Gemma and handles all inference — MLX GPU
                     streams are thread-local so load and infer stay
                     on this one thread
stop_recording       Stops audio stream, grabs selected text in command mode
transcribe_and_paste Whisper → Gemma punctuation fix → paste
command_and_paste    Whisper → Gemma command execution → paste
_watchdog            Monitors recording duration; force-stops if stuck >120 s
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
