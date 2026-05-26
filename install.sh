#!/bin/bash
set -e

INSTALL_DIR="$HOME/.fiveflow"
REPO_URL="https://github.com/deshiyan1010/fiveflow.git"

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

# ── Clone or update ───────────────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating existing installation in $INSTALL_DIR ..."
    git -C "$INSTALL_DIR" pull --quiet
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

# ── Shell config ──────────────────────────────────────────────────────────────
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

# ── Shell functions ───────────────────────────────────────────────────────────
FUNC_BLOCK='
# fiveflow — voice STT overlay
fiveflow() {
    "'"$INSTALL_DIR"'/stt_env/bin/python" "'"$INSTALL_DIR"'/stt_pipeline.py"
}
fiveflow-update() {
    echo "[fiveflow] Pulling latest changes..."
    git -C "'"$INSTALL_DIR"'" pull
    echo "[fiveflow] Updating dependencies..."
    "'"$INSTALL_DIR"'/stt_env/bin/pip" install -q --upgrade pip
    "'"$INSTALL_DIR"'/stt_env/bin/pip" install -q -r "'"$INSTALL_DIR"'/requirements.txt"
    echo "[fiveflow] Update complete. Run: fiveflow"
}
fiveflow-remove() {
    echo "[fiveflow] Removing installation at '"$INSTALL_DIR"'..."
    rm -rf "'"$INSTALL_DIR"'"
    # Remove the fiveflow block from this shell config
    perl -i -0pe '"'"'s/\n# fiveflow.*?(?=\n[^}]|\z)//s'"'"' "'"$SHELL_RC"'"
    echo "[fiveflow] Uninstalled. Run: source '"$SHELL_RC"'"
}'

# Remove any existing fiveflow block, then write fresh
if grep -q "# fiveflow" "$SHELL_RC" 2>/dev/null; then
    perl -i -0pe 's/\n# fiveflow — voice STT overlay\nfiveflow\(\).*?^}.*?^}.*?^}//ms' "$SHELL_RC" 2>/dev/null || true
    info "Replacing existing fiveflow functions in $SHELL_RC"
fi

echo "$FUNC_BLOCK" >> "$SHELL_RC"
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
echo "  Completely uninstall:"
echo "    fiveflow-remove"
echo ""
echo "  On first launch, grant Accessibility permission when prompted."
echo "  The pill will appear once models finish loading (~30–60 s first time)."
echo ""
