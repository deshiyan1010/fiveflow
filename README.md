# fiveflow

A real-time speech-to-text overlay for macOS. Hold a hotkey, speak, release — your words appear wherever your cursor is. Built on Whisper for transcription and Gemma for punctuation correction and voice commands.

## Demo

A floating pill widget lives at the bottom of your screen. It stays out of the way until you need it.

| State | Appearance |
|-------|-----------|
| Idle | Small gray pill |
| Recording | Animated bars that scale with mic level |
| Processing | Dark pill with status label |

Click the idle pill to open a **transcription history** panel showing all past transcriptions with copy buttons.

## Features

- **Transcribe mode** — Hold `Fn`, speak, release. Whisper transcribes, Gemma corrects punctuation, result is pasted at your cursor.
- **Command mode** — Hold `Shift+Fn`, speak a command, release. If you have text selected, Gemma applies your command to it (e.g. "make this formal"). If nothing is selected, Gemma executes the command freely.
- **History panel** — Click the idle pill to see all past transcriptions and command exchanges, each with individual copy buttons.
- **Parallel model loading** — Whisper (MPS) and Gemma (MLX) load simultaneously at startup.
- Fully transparent, click-through overlay — never steals focus.

## Requirements

- macOS (Apple Silicon recommended — Gemma runs on MLX, Whisper on MPS)
- Python 3.13
- Accessibility permission (`System Settings → Privacy & Security → Accessibility`)
- Microphone permission (prompted on first recording)

## Setup

```bash
python3.13 -m venv stt_env
source stt_env/bin/activate
pip install -r requirements.txt
```

On first run, if the models are not yet cached, you will be prompted for a Hugging Face token. A token is optional but significantly speeds up the download.

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Running

```bash
stt_env/bin/python stt_pipeline.py
```

Grant Accessibility permission when prompted, then wait for the pill to show `idle` (~30–60 s for first-time model loading).

## Models

| Model | Purpose | Runtime |
|-------|---------|---------|
| `openai/whisper-large-v3-turbo` | Speech recognition | PyTorch / MPS |
| `mlx-community/gemma-4-e2b-it-4bit` | Text correction & commands | MLX |

## Hotkeys

| Hotkey | Mode |
|--------|------|
| Hold `Fn` → release | Transcribe and paste |
| Hold `Shift+Fn` → release | Command mode (acts on selected text or generates freely) |

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
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
