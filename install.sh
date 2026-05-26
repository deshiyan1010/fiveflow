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

# ── Check Python 3.13 ─────────────────────────────────────────────────────────
if ! command -v python3.13 &>/dev/null; then
    echo ""
    warn "Python 3.13 is required but not found."
    echo "  Install it from https://www.python.org/downloads/ or via Homebrew:"
    echo "  brew install python@3.13"
    echo ""
    exit 1
fi

# ── Clone or sync to latest ───────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    info "Syncing to latest version..."
    git -C "$INSTALL_DIR" fetch origin --quiet
    git -C "$INSTALL_DIR" reset --hard origin/master --quiet
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

# ── Remove any existing fiveflow block (sentinel-based) ──────────────────────
if grep -q "# >>> fiveflow >>>" "$SHELL_RC" 2>/dev/null; then
    sed -i '' '/# >>> fiveflow >>>/,/# <<< fiveflow <<</d' "$SHELL_RC"
    info "Replacing existing fiveflow functions in $SHELL_RC"
fi

# ── Write shell functions with sentinels ─────────────────────────────────────
cat >> "$SHELL_RC" << SHELL_BLOCK

# >>> fiveflow >>>
fiveflow() {
    "$INSTALL_DIR/stt_env/bin/python" "$INSTALL_DIR/stt_pipeline.py"
}
fiveflow-update() {
    echo "[fiveflow] Fetching latest version..."
    git -C "$INSTALL_DIR" fetch origin --quiet
    git -C "$INSTALL_DIR" reset --hard origin/master --quiet
    echo "[fiveflow] Updating dependencies..."
    "$INSTALL_DIR/stt_env/bin/pip" install -q --upgrade pip
    "$INSTALL_DIR/stt_env/bin/pip" install -q -r "$INSTALL_DIR/requirements.txt"
    echo "[fiveflow] Done. Run: fiveflow"
}
fiveflow-remove() {
    echo "[fiveflow] Uninstalling..."
    # App files
    rm -rf "$INSTALL_DIR"
    # Hugging Face model caches
    rm -rf "$WHISPER_CACHE"
    rm -rf "$GEMMA_CACHE"
    # Shell functions
    sed -i '' '/# >>> fiveflow >>>/,/# <<< fiveflow <<</d' "$SHELL_RC"
    echo "[fiveflow] Fully removed. Run: source $SHELL_RC"
}
# <<< fiveflow <<<
SHELL_BLOCK

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
