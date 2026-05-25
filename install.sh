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

# Shell function (handles paths with spaces, easier to update later)
FUNC_BLOCK="
# fiveflow — voice STT overlay
fiveflow() {
    \"$INSTALL_DIR/stt_env/bin/python\" \"$INSTALL_DIR/stt_pipeline.py\"
}"

if grep -q "fiveflow()" "$SHELL_RC" 2>/dev/null; then
    # Remove old block and rewrite
    perl -i -0pe 's/\n# fiveflow.*?fiveflow\(\) \{.*?\}//s' "$SHELL_RC"
    info "Updated fiveflow command in $SHELL_RC"
fi

if ! grep -q "fiveflow()" "$SHELL_RC" 2>/dev/null; then
    echo "$FUNC_BLOCK" >> "$SHELL_RC"
    info "Added fiveflow command to $SHELL_RC"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
success "Installation complete!"
echo ""
echo "  Reload your shell config with:"
echo "    source $SHELL_RC"
echo ""
echo "  Then launch fiveflow anytime with:"
echo "    fiveflow"
echo ""
echo "  On first launch, grant Accessibility permission when prompted."
echo "  The pill will appear once models finish loading (~30–60 s first time)."
echo ""
