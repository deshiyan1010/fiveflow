#!/bin/bash
set -e

INSTALL_DIR="$HOME/.fiveflow"
REPO_URL="https://github.com/deshiyan1010/fiveflow.git"
HF_CACHE="$HOME/.cache/huggingface/hub"
WHISPER_CACHE="$HF_CACHE/models--openai--whisper-large-v3-turbo"
GEMMA_CACHE="$HF_CACHE/models--mlx-community--gemma-4-e2b-it-4bit"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${BLUE}[fiveflow]${NC} $1"; }
success() { echo -e "${GREEN}[fiveflow]${NC} $1"; }
warn()    { echo -e "${YELLOW}[fiveflow]${NC} $1"; }

echo ""
echo "  ███████╗██╗██╗   ██╗███████╗███████╗██╗      ██████╗ ██╗    ██╗"
echo "  ██╔════╝██║██║   ██║██╔════╝██╔════╝██║     ██╔═══██╗██║    ██║"
echo "  █████╗  ██║██║   ██║█████╗  █████╗  ██║     ██║   ██║██║ █╗ ██║"
echo "  ██╔══╝  ██║╚██╗ ██╔╝██╔══╝  ██╔══╝  ██║     ██║   ██║██║███╗██║"
echo "  ██║     ██║ ╚████╔╝ ███████╗██║     ███████╗╚██████╔╝╚███╔███╔╝"
echo "  ╚═╝     ╚═╝  ╚═══╝  ╚══════╝╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝ "
echo ""

# ── Check macOS ───────────────────────────────────────────────────────────────
if [[ "$(uname)" != "Darwin" ]]; then
    echo "fiveflow currently supports macOS only. Windows and Linux support coming soon."
    exit 1
fi

# MLX runs on Apple silicon with macOS 14 or newer.
if [[ "$(uname -m)" != "arm64" ]] || [[ "$(sw_vers -productVersion | cut -d. -f1)" -lt 14 ]]; then
    warn "Fiveflow requires Apple silicon and macOS 14 or newer. Use a native arm64 terminal."
    exit 1
fi

# ── Check Python 3.13 ─────────────────────────────────────────────────────────
if ! command -v python3.13 &>/dev/null; then
    echo ""
    warn "Python 3.13 is required but not found."
    echo "  Install it from https://www.python.org/downloads/ or via Homebrew:"
    echo "  brew install python@3.13"
    echo ""
    exit 1
fi

if [[ "$(python3.13 -c 'import platform; print(platform.machine())')" != "arm64" ]]; then
    warn "Python 3.13 must run natively on arm64, not under Rosetta."
    exit 1
fi

# ── Clone or sync to latest ───────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    info "Syncing to latest version..."
    git -C "$INSTALL_DIR" fetch origin --quiet
    if [ -n "$(git -C "$INSTALL_DIR" status --porcelain --untracked-files=normal)" ]; then
        warn "Local changes in $INSTALL_DIR. Commit or stash them before reinstalling."
        exit 1
    fi
    git -C "$INSTALL_DIR" merge --ff-only origin/master --quiet
else
    info "Cloning fiveflow into $INSTALL_DIR ..."
    git clone --quiet "$REPO_URL" "$INSTALL_DIR"
fi

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "$INSTALL_DIR/stt_env" ]; then
    info "Creating Python 3.13 virtual environment..."
    python3.13 -m venv "$INSTALL_DIR/stt_env"
fi

info "Installing dependencies (this may take a moment on first run)..."
"$INSTALL_DIR/stt_env/bin/pip" install -q --upgrade pip
"$INSTALL_DIR/stt_env/bin/pip" install -q -r "$INSTALL_DIR/requirements.txt"

# ── Detect shell config ───────────────────────────────────────────────────────
detect_shell_rc() {
    local shell_name
    shell_name="$(basename "$SHELL")"
    case "$shell_name" in
        zsh)  echo "$HOME/.zshrc" ;;
        bash) echo "$HOME/.bash_profile" ;;
        *)    echo "$HOME/.profile" ;;
    esac
}
SHELL_RC="$(detect_shell_rc)"

# ── Check if SHELL_RC is writable ─────────────────────────────────────────────
IS_WRITABLE=true
if [ -f "$SHELL_RC" ] && [ ! -w "$SHELL_RC" ]; then
    IS_WRITABLE=false
elif [ ! -f "$SHELL_RC" ] && [ ! -w "$(dirname "$SHELL_RC")" ]; then
    IS_WRITABLE=false
fi

# ── Register safely quoted shell functions without replacing user dotfiles ──
if [ "$IS_WRITABLE" = true ]; then
    "$INSTALL_DIR/stt_env/bin/python" "$INSTALL_DIR/fiveflow/shell_config.py" --install "$SHELL_RC" "$INSTALL_DIR" "$WHISPER_CACHE" "$GEMMA_CACHE"
else
    warn "$SHELL_RC is not writable. Writing configuration with sudo..."
    sudo "$INSTALL_DIR/stt_env/bin/python" "$INSTALL_DIR/fiveflow/shell_config.py" --install "$SHELL_RC" "$INSTALL_DIR" "$WHISPER_CACHE" "$GEMMA_CACHE"
fi

info "Added fiveflow / fiveflow-update / fiveflow-remove to $SHELL_RC"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
success "Installation complete!"
echo ""
echo "  Reload your shell config:"
echo "    source $SHELL_RC"
echo ""
echo "  Start fiveflow:"
echo "    fiveflow"
echo ""
echo "  Update to the latest version:"
echo "    fiveflow-update"
echo ""
echo "  Completely uninstall (removes app + downloaded models):"
echo "    fiveflow-remove"
echo ""
echo "  On first launch, grant Accessibility permission when prompted."
echo "  The pill will appear once models finish loading (~30–60 s first time)."
echo ""
