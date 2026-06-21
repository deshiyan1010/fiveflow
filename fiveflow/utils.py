import os
import sys
import subprocess
import tty
import termios

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
    cache = Path.home() / ".cache" / "huggingface" / "hub"

    def is_cached(model_id):
        return (cache / ("models--" + model_id.replace("/", "--"))).exists()

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
    token = input("HF token [press Enter to skip]: ").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        print()
    else:
        print("Warning: downloading without a token will be very slow.\n")


def ask_yes_no_arrow(prompt):
    print(prompt)
    choice = True

    def draw():
        sys.stdout.write("\r\033[K")
        if choice:
            sys.stdout.write("  \033[1m\033[36m> Yes\033[0m    No ")
        else:
            sys.stdout.write("    Yes  \033[1m\033[36m> No\033[0m  ")
        sys.stdout.flush()

    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except termios.error:
        try:
            res = input(" (Yes/No) [y/N]: ").strip().lower()
            return res in ('y', 'yes')
        except (KeyboardInterrupt, EOFError):
            return False

    try:
        tty.setraw(fd)
        draw()
        while True:
            ch1 = sys.stdin.read(1)
            if ch1 in ('\r', '\n'):
                break
            elif ch1.lower() == 'y':
                choice = True
                draw()
            elif ch1.lower() == 'n':
                choice = False
                draw()
            elif ch1 == '\033':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
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
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(os.path.join(script_dir, ".git")):
        return

    print("[fiveflow] Checking for updates...")
    try:
        subprocess.run(["git", "-C", script_dir, "fetch", "origin", "--quiet"],
                       timeout=5.0, capture_output=True, check=True)

        local_hash = subprocess.run(["git", "-C", script_dir, "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=True).stdout.strip()
        remote_hash = subprocess.run(["git", "-C", script_dir, "rev-parse", "origin/master"],
                                     capture_output=True, text=True, check=True).stdout.strip()

        if local_hash != remote_hash:
            behind = subprocess.run(["git", "-C", script_dir, "log", f"{local_hash}..{remote_hash}", "--oneline"],
                                    capture_output=True, text=True, check=True).stdout.strip()
            if behind:
                prompt = "\n[fiveflow] A new update is available! Would you like to upgrade now?"
                if ask_yes_no_arrow(prompt):
                    print("[fiveflow] Upgrading to the latest version...")
                    subprocess.run(["git", "-C", script_dir, "reset", "--hard", "origin/master"], check=True)

                    pip_path = os.path.join(os.path.dirname(sys.executable), "pip")
                    req_path = os.path.join(script_dir, "requirements.txt")
                    if os.path.exists(pip_path) and os.path.exists(req_path):
                        print("[fiveflow] Updating dependencies...")
                        subprocess.run([pip_path, "install", "-q", "-r", req_path], check=True)

                    print("[fiveflow] Upgrade complete! Restarting...")
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                else:
                    print("[fiveflow] Update skipped.")
    except Exception as e:
        print(f"[fiveflow] Could not check for updates: {e}")
