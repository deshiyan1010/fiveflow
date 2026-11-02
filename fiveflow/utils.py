import os
import sys
import subprocess
import tty
import termios
import getpass
import select

from Foundation import NSDictionary
from ApplicationServices import AXIsProcessTrustedWithOptions

from .config import WHISPER_MODEL_ID, GEMMA_MODEL_ID


def request_accessibility():
    opts = NSDictionary.dictionaryWithObject_forKey_(True, "AXTrustedCheckOptionPrompt")
    if not AXIsProcessTrustedWithOptions(opts):
        print("Accessibility permission required. Grant it in System Settings and restart.")
        sys.exit(1)


def prompt_hf_token_if_needed():
    from pathlib import Path
    if os.environ.get('HF_TOKEN') or not sys.stdin.isatty():
        return
    cache = Path(os.environ.get('HF_HUB_CACHE',
                 str(Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'hub')))

    def is_cached(model_id):
        directory = cache / ("models--" + model_id.replace("/", "--")) / 'snapshots'
        return any(directory.glob('*/config.json'))

    whisper_ok = is_cached(WHISPER_MODEL_ID)
    gemma_ok   = is_cached(GEMMA_MODEL_ID)

    if whisper_ok and gemma_ok:
        return

    missing = []
    if not whisper_ok:
        missing.append(f"  - {WHISPER_MODEL_ID}")
    if not gemma_ok:
        missing.append(f"  - {GEMMA_MODEL_ID}")

    print("\nThe following models are not cached and need to be downloaded:")
    for m in missing:
        print(m)
    print()
    print("A Hugging Face token (huggingface.co/settings/tokens) speeds up the download.")
    print("If you skip this, the download will be very slow (no rate-limit bypass).")
    print()
    try:
        token = getpass.getpass("HF token [press Enter to skip]: ").strip()
    except EOFError:
        return
    if token:
        os.environ["HF_TOKEN"] = token
        print()
    else:
        print("Warning: downloading without a token will be very slow.\n")


def ask_yes_no_arrow(prompt):
    print(prompt)
    choice = False

    def draw():
        sys.stdout.write("\r\033[K")
        if choice:
            sys.stdout.write("  \033[1m\033[36m> Yes\033[0m    No ")
        else:
            sys.stdout.write("    Yes  \033[1m\033[36m> No\033[0m  ")
        sys.stdout.flush()

    if not sys.stdin.isatty():
        return False
    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except (OSError, ValueError, AttributeError):
        return False

    try:
        tty.setraw(fd)
        draw()
        while True:
            ch1 = os.read(fd, 1).decode('ascii', errors='ignore')
            if ch1 == '\x03':
                choice = False
                break
            if not ch1:
                choice = False
                break
            if ch1 in ('\r', '\n'):
                break
            elif ch1.lower() == 'y':
                choice = True
                draw()
            elif ch1.lower() == 'n':
                choice = False
                draw()
            elif ch1 == '\033':
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    choice = False
                    break
                ch2 = os.read(fd, 1).decode('ascii', errors='ignore')
                if ch2 == '[':
                    if not select.select([sys.stdin], [], [], 0.1)[0]:
                        choice = False
                        break
                    ch3 = os.read(fd, 1).decode('ascii', errors='ignore')
                    if ch3 in ('A', 'D'):
                        choice = True
                    elif ch3 in ('B', 'C'):
                        choice = False
                    draw()
    except (KeyboardInterrupt, EOFError):
        choice = False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    print()
    return choice


def check_for_updates():
    from pathlib import Path
    from .updates import available_update, apply_update
    root = Path(__file__).resolve().parents[1]
    # Background/GUI launches must not block on terminal prompts.
    if not sys.stdin.isatty():
        return
    try:
        target = available_update(root)
        if target and ask_yes_no_arrow("[fiveflow] An update is available. Upgrade now?"):
            apply_update(root, target)
            os.execv(sys.executable, [sys.executable, str(root / 'stt_pipeline.py'), *sys.argv[1:]])
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[fiveflow] Could not update: {exc}")
