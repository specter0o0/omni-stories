#!/usr/bin/env bash

# Omni-Stories v1.1.0 Setup Wizard
# Universal, Zero-Failure, Fully Automated.
# "It just works."

set -e

# ==============================================================================
# 0. Utilities & Constants
# ==============================================================================

# Terminal Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m'
BOLD='\033[1m'

ROOT_NAME="omni-stories"
DATA_NAME=".omni-stories-data"
REPO_URL="https://github.com/specter0o0/omni-stories"
LOCAL_BIN="$HOME/.local/bin"

BANNER="
${CYAN}╔═══════════════════════════════════════════════════════════╗
║  ${MAGENTA} ██████╗ ███╗   ███╗███╗   ██╗██╗${CYAN}                        ║
║  ${MAGENTA}██╔═══██╗████╗ ████║████╗  ██║██║${CYAN}                        ║
║  ${MAGENTA}██║   ██║██╔████╔██║██╔██╗ ██║██║${CYAN}                        ║
║  ${MAGENTA}██║   ██║██║╚██╔╝██║██║╚██╗██║██║${CYAN}                        ║
║  ${MAGENTA}╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║${CYAN}                        ║
║  ${MAGENTA} ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝${CYAN}                        ║
║  ${WHITE}███████╗████████╗ ██████╗ ██████╗ ██╗███████╗███████╗${CYAN}    ║
║  ${WHITE}██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██║██╔════╝██╔════╝${CYAN}    ║
║  ${WHITE}███████╗   ██║   ██║   ██║██████╔╝██║█████╗  ███████╗${CYAN}    ║
║  ${WHITE}╚════██║   ██║   ██║   ██║██╔══██╗██║██╔══╝  ╚════██║${CYAN}    ║
║  ${WHITE}███████║   ██║   ╚██████╔╝██║  ██║██║███████╗███████║${CYAN}    ║
║  ${WHITE}╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝${CYAN}    ║
╚═══════════════════════════════════════════════════════════╝${NC}
"

log_info() { echo -e "${CYAN}[I] $1${NC}"; }
log_ok()   { echo -e "${GREEN}[✔] $1${NC}"; }
log_warn() { echo -e "${YELLOW}[!] $1${NC}"; }
log_err()  { echo -e "${RED}[✖] $1${NC}"; }

# Retry mechanism for network operations (3 attempts)
retry_cmd() {
    local -i retries=5
    local -i count=0
    until "$@"; do
        exit_code=$?
        count=$((count + 1))
        if [ $count -lt $retries ]; then
            log_warn "Command failed (Attempt $count/$retries). Retrying..."
            sleep 2
        else
            log_err "Command failed after $retries attempts."
            return $exit_code
        fi
    done
    return 0
}

# Check if a specific binary exists in PATH
has_cmd() {
    command -v "$1" &> /dev/null
}

clear
echo -e "$BANNER"
log_info "Initializing Omni-Stories Installer..."

# Parse Flags
SILENT=false
API_KEYS=""
for arg in "$@"; do
    if [[ "$arg" == "-s" || "$arg" == "--silent" ]]; then SILENT=true; fi
    if [[ "$arg" != "-"* && -n "$arg" ]]; then API_KEYS="$arg"; fi
done

# ==============================================================================
# 1. System Dependency Resolution
# ==============================================================================

log_info "Step 1/6: Checking System Dependencies..."

install_pkg_mgr() {
    local sys_deps="git ffmpeg python3 espeak-ng unzip"
    local install_cmd=""
    
    # Pre-check: What's actually missing?
    local missing=""
    for bin in git ffmpeg python3 espeak-ng unzip; do
        has_cmd "$bin" || missing="$missing $bin"
    done
    
    # If nothing missing, skip pkg manager entirely
    if [ -z "$missing" ]; then
        return 0
    fi

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        has_cmd brew || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        install_cmd="brew install $missing"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        log_warn "Windows support is experimental. Ensure Git, FFmpeg, and Python are in PATH."
        return 0
    elif has_cmd apt-get; then
        sudo apt-get update >/dev/null 2>&1
        install_cmd="sudo apt-get install -y $missing libass-dev"
    elif has_cmd dnf; then
        install_cmd="sudo dnf install -y $missing"
    elif has_cmd pacman; then
        install_cmd="sudo pacman -S --noconfirm $missing"
    elif has_cmd zypper; then
        install_cmd="sudo zypper install -y $missing"
    fi

    if [ -n "$install_cmd" ]; then
        if ! eval "$install_cmd" >/dev/null 2>&1; then
             # Only warn if something is STILL missing after the attempt
             local still_missing=""
             for bin in $missing; do
                 has_cmd "$bin" || still_missing="$still_missing $bin"
             done
             
             if [ -n "$still_missing" ]; then
                log_warn "Package manager encountered errors. Required items still missing: $still_missing"
             fi
        fi
    fi
}

# Run install attempt
install_pkg_mgr

# STRICT VERIFICATION: If binary checks fail after install attempt, then we actually fail
MISSING_CRITICAL=""
for bin in git ffmpeg python3 espeak-ng; do
    if ! has_cmd "$bin"; then 
        MISSING_CRITICAL="$MISSING_CRITICAL $bin"
    fi
done

if [ -n "$MISSING_CRITICAL" ]; then
    log_err "Critical Failure: The following dependencies are missing and could not be installed:"
    log_err "   -> $MISSING_CRITICAL"
    log_err "Please install them manually."
    exit 1
fi
log_ok "System dependencies verified."

# ==============================================================================
# 2. Workspace Provisioning
# ==============================================================================

log_info "Step 2/6: Setting up Workspace..."

if [ -d "$DATA_NAME" ] && [ -f "$DATA_NAME/main.py" ]; then
    # Already inside repo
    ABSOLUTE_ROOT="$(pwd)"
    log_ok "Using existing directory."
else
    # Need to deploy
    if [ -d "$ROOT_NAME" ]; then
        log_warn "Directory '$ROOT_NAME' exists. Updating..."
        cd "$ROOT_NAME"
        if [ -d ".git" ]; then
            git pull >/dev/null 2>&1 || true
        fi
    else
        log_info "Downloading Omni-Stories..."
        if has_cmd git; then
            git clone --depth 1 "$REPO_URL.git" "$ROOT_NAME" -q || { log_err "Git clone failed."; exit 1; }
            cd "$ROOT_NAME"
        else
            mkdir -p "$ROOT_NAME"
            curl -L -s "$REPO_URL/archive/refs/heads/main.zip" -o "os.zip"
            unzip -q "os.zip" -d "os-tmp"
            mv "os-tmp/omni-stories-main/"* "$ROOT_NAME/"
            rm -rf "os-tmp" "os.zip"
            cd "$ROOT_NAME"
        fi
    fi
    ABSOLUTE_ROOT="$(pwd)"
fi

# ==============================================================================
# 3. Python Environment Optimization
# ==============================================================================

log_info "Step 3/6: configuring Python Runtime..."

# Find best Python version (prefer 3.12 > 3.11 > 3.10 > 3.9)
PYTHON_CMD=""
for ver in 3.12 3.11 3.10 3.9 3; do
    if has_cmd "python$ver"; then
        PYTHON_CMD="python$ver"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    log_err "No suitable Python 3 version found."
    exit 1
fi

log_ok "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Smart PIP Install
pip_install_args="--user"
if $PYTHON_CMD -m pip help install | grep -q 'break-system-packages'; then
    pip_install_args="--user --break-system-packages"
fi

log_info "Installing Dependencies (this may take a moment)..."
retry_cmd $PYTHON_CMD -m pip install --upgrade pip $pip_install_args >/dev/null 2>&1
retry_cmd $PYTHON_CMD -m pip install -r "$DATA_NAME/requirements.txt" $pip_install_args >/dev/null 2>&1

log_ok "Python environment ready."

# ==============================================================================
# 4. Configuration
# ==============================================================================

log_info "Step 4/6: Configuring Credentials..."

if [ "$SILENT" = false ] && [ -z "$API_KEYS" ]; then
    echo -en "${YELLOW}Enter ElevenLabs API Key(s) (Optional, comma-separated): ${NC}"
    read -r API_KEYS < /dev/tty || true
fi

echo "ELEVENLABS_API_KEYS='$API_KEYS'" > .env
log_ok "Configuration saved."

# ==============================================================================
# 5. Asset Provisioning (Self-Healing)
# ==============================================================================

log_info "Step 5/6: Provisioning Assets..."

mkdir -p "$DATA_NAME/models/kokoro"
mkdir -p "$DATA_NAME/background_videos"

# Download Kokoro Model
# Download Kokoro Model
MODEL_PATH="$DATA_NAME/models/kokoro/model.onnx"
VOCAB_PATH="$DATA_NAME/models/kokoro/tokenizer.json"
VOICE_PT_PATH="$DATA_NAME/models/kokoro/am_adam.pt"
VOICE_BIN_PATH="$DATA_NAME/models/kokoro/am_adam.bin"

# Validation: Check if existing model is valid (>100MB)
if [ -f "$MODEL_PATH" ]; then
    fsize=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null || echo 0)
    if [ "$fsize" -lt 100000000 ]; then
        log_warn "Corrupted model detected ($fsize bytes). Redownloading..."
        rm -f "$MODEL_PATH"
    fi
fi

if [ ! -s "$MODEL_PATH" ]; then
    log_info "Downloading Kokoro TTS Model (330MB)..."
    retry_cmd curl -L --connect-timeout 10 -# "https://huggingface.co/hexgrad/kLegacy/resolve/main/v0.19/kokoro-v0_19.onnx" --output "$MODEL_PATH"
fi

if [ ! -s "$VOCAB_PATH" ]; then
     log_info "Downloading Tokenizer Config..."
     retry_cmd curl -L --connect-timeout 10 -s "https://huggingface.co/hexgrad/kLegacy/resolve/main/v0.19/config.json" --output "$VOCAB_PATH"
fi

if [ ! -s "$VOICE_BIN_PATH" ]; then
    if [ ! -s "$VOICE_PT_PATH" ]; then
         log_info "Downloading Voice Model (am_adam)..."
         retry_cmd curl -L --connect-timeout 10 -s "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/am_adam.pt" --output "$VOICE_PT_PATH"
    fi
    
    log_info "Converting Voice Model to Binary..."
    $PYTHON_CMD -c "
import torch
import numpy as np
try:
    data = torch.load('$VOICE_PT_PATH', map_location='cpu', weights_only=True)
    # Extract weight tensor (handle both raw tensor and dict checkpoint formats)
    tensor = data if isinstance(data, torch.Tensor) else data.get('weight', data)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    tensor.astype(np.float32).tofile('$VOICE_BIN_PATH')
    print('Conversion successful.')
except Exception as e:
    print(f'Conversion failed: {e}')
" || log_warn "Voice conversion failed. TTS may fallback or fail."
fi

# Cache Whisper Model
log_info "Caching Whisper Model..."
$PYTHON_CMD -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-base.en')" > /dev/null 2>&1 || true

# Download Background Video
if [ -z "$(ls -A "$DATA_NAME/background_videos")" ]; then
    log_info "Downloading Sample Background Footage (this may take a minute)..."
    if ! $PYTHON_CMD "$DATA_NAME/main.py" --dl_video -u "https://www.youtube.com/watch?v=n_Dv4JH_G_E" -r "1080p"; then
        log_warn "Background download failed. You can download one later with: omni-stories --dl_video"
    else
        log_ok "Background footage ready."
    fi
fi

log_ok "Assets provisioned."

# ==============================================================================
# 6. Global Command Setup
# ==============================================================================

log_info "Step 6/6: Finalizing..."

mkdir -p "$LOCAL_BIN"
SHIM="$LOCAL_BIN/omni-stories"

cat <<EOF > "$SHIM"
#!/usr/bin/env bash
PROJECT_DIR="$ABSOLUTE_ROOT"
export PYTHONPATH="\$PROJECT_DIR/\$DATA_NAME:\$PYTHONPATH"
exec "$PYTHON_CMD" "\$PROJECT_DIR/$DATA_NAME/main.py" "\$@"
EOF
chmod +x "$SHIM"

# Shell Injection
inject_path() {
    local rc="$1"
    local line="export PATH=\"\$HOME/.local/bin:\$PATH\""
    if [ -f "$rc" ]; then
        if ! grep -Fq "$line" "$rc"; then
            echo -e "\n# Omni-Stories Path\n$line" >> "$rc"
            log_ok "Added to $rc"
        fi
    fi
}
inject_path "$HOME/.bashrc"
inject_path "$HOME/.zshrc"
inject_path "$HOME/.profile"
inject_path "$HOME/.bash_profile"

echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e " ${BOLD}INSTALLATION COMPLETE${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e " You can now run: ${CYAN}omni-stories${NC}"
echo -e " To check status: ${CYAN}omni-stories --doctor${NC}"
echo ""
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
    echo -e "${YELLOW}NOTE: You may need to restart your terminal.${NC}"
fi
exit 0
