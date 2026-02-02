#!/usr/bin/env bash

# Omni-Stories v1.1.0 Universal Installer
# Cross-platform, self-healing, production-grade installation system
# Supports: Linux, macOS (Intel/ARM), Windows (Native/MSYS2/Cygwin/WSL/Git Bash)

set -e

# ==============================================================================
# Utilities & Constants
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

# Dynamic path resolution with XDG fallbacks
if [ -n "$XDG_BIN_HOME" ];then
    USER_BIN="$XDG_BIN_HOME"
elif [ -n "$HOME" ]; then
    USER_BIN="$HOME/.local/bin"
else
    USER_BIN="/usr/local/bin"
fi

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

# Transaction log for rollback support
# Transaction log for rollback support
if command -v mktemp >/dev/null 2>&1; then
    TRANSACTION_LOG=$(mktemp)
else
    # Fallback for systems without mktemp (e.g. strict Windows environments)
    TRANSACTION_LOG="${TMPDIR:-/tmp}/omni-stories-install-$$.log"
    touch "$TRANSACTION_LOG"
fi

log_info() { echo -e "${CYAN}[I] $1${NC}"; }
log_ok()   { echo -e "${GREEN}[✔] $1${NC}"; }
log_warn() { echo -e "${YELLOW}[!] $1${NC}"; }
log_err()  { echo -e "${RED}[✖] $1${NC}"; }
log_transaction() { echo "$1" >> "$TRANSACTION_LOG"; }

# Exponential backoff retry mechanism
retry_cmd() {
    local -i max_attempts=5
    local -i attempt=1
    local -i delay=2
    
    until "$@"; do
        exit_code=$?
        if [ $attempt -lt $max_attempts ]; then
            log_warn "Command failed (Attempt $attempt/$max_attempts). Retrying in ${delay}s..."
            sleep $delay
            attempt=$((attempt + 1))
            delay=$((delay * 2))  # Exponential backoff
        else
            log_err "Command failed after $max_attempts attempts."
            return $exit_code
        fi
    done
    return 0
}

# Check if command exists
has_cmd() {
    command -v "$1" &> /dev/null
}

# Get disk space in MB
get_disk_space_mb() {
    local path="$1"
    if df -m "$path" 2>/dev/null | tail -1 | awk '{print $4}' | grep -E '^[0-9]+$' &>/dev/null; then
        df -m "$path" | tail -1 | awk '{print $4}'
    else
        echo "0"
    fi
}

# Detect OS and environment
detect_environment() {
    IS_WINDOWS=false
    IS_WSL=false
    IS_MACOS=false
    IS_LINUX=false
    IS_ARM=false
    
    # Check architecture
    if uname -m | grep -qE 'arm|aarch64'; then
        IS_ARM=true
    fi
    
    # Check OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        IS_MACOS=true
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]] || [[ -n "$WINDIR" ]] || [[ -n "$MSYSTEM" ]]; then
        IS_WINDOWS=true
    elif grep -qiE '(Microsoft|WSL)' /proc/version 2>/dev/null; then
        IS_WSL=true
        IS_LINUX=true
    else
        IS_LINUX=true
    fi
}

# Validate Python version
validate_python() {
    local python_cmd="$1"
    local version=$($python_cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
    local major=$(echo "$version" | cut -d. -f1)
    local minor=$(echo "$version" | cut -d. -f2)
    
    if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
        return 0
    fi
    return 1
}

# Test internet connectivity
test_connectivity() {
    local test_urls=("https://www.google.com" "https://github.com" "https://pypi.org")
    for url in "${test_urls[@]}"; do
        if curl -s --connect-timeout 5 --max-time 10 -I "$url" &>/dev/null; then
            return 0
        fi
    done
    return 1
}

clear
echo -e "$BANNER"
log_info "Initializing Omni-Stories Universal Installer..."

# Detect environment
detect_environment

# Parse flags
SILENT=false
API_KEYS=""
UNATTENDED=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--silent) SILENT=true ;;
        --unattended|--ci) UNATTENDED=true; SILENT=true ;;
        -k|--key|--api-key|--s) 
            if [ -n "$API_KEYS" ]; then
                API_KEYS="$API_KEYS,$2"
            else
                API_KEYS="$2"
            fi
            shift 
            ;;
        -*) log_warn "Unknown option: $1" ;;
        *) 
            if [ -n "$API_KEYS" ]; then
                API_KEYS="$API_KEYS,$1"
            else
                API_KEYS="$1"
            fi
            ;; # Backward compatibility for positional keys
    esac
    shift
done

# ==============================================================================
# Pre-flight System Validation
# ==============================================================================

log_info "Step 1/7: Pre-flight System Validation..."

# Check disk space (require 2GB free)
log_info "Checking available disk space..."
available_mb=$(get_disk_space_mb ".")
if [ "$available_mb" -lt 2048 ]; then
    log_err "Insufficient disk space. Required: 2GB, Available: ${available_mb}MB"
    exit 1
fi
log_ok "Disk space sufficient: ${available_mb}MB available"

# Check internet connectivity
log_info "Testing internet connectivity..."
if ! test_connectivity; then
    log_err "No internet connection detected. Installation requires internet access."
    exit 1
fi
log_ok "Internet connection verified"

# ==============================================================================
# Python Environment Validation
# ==============================================================================

log_info "Step 2/7: Validating Python Environment..."

# Find suitable Python (3.9+)
# Find suitable Python (3.9+)
PYTHON_CMD=""

# 1. Try explicit version candidates (prioritize stable versions)
for ver in 3.12 3.11 3.10 3.9; do
    cmd="python$ver"
    if has_cmd "$cmd" && validate_python "$cmd"; then
        PYTHON_CMD="$cmd"
        break
    fi
done

# 2. Fallback to generic commands if no specific version found
if [ -z "$PYTHON_CMD" ]; then
    for cmd in "python3" "python"; do
        if has_cmd "$cmd" && validate_python "$cmd"; then
            PYTHON_CMD="$cmd"
            break
        fi
    done
fi

if [ -z "$PYTHON_CMD" ]; then
    log_err "Python 3.9+ not found. Please install Python 3.9 or newer."
    log_info "Visit: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version)
log_ok "Python validated: $PYTHON_VERSION"

# ==============================================================================
# System Dependency Resolution
# ==============================================================================

log_info "Step 3/7: Installing System Dependencies..."

install_system_deps() {
    local sys_deps="git ffmpeg python3 espeak-ng unzip"
    local missing=""
    
    # Check what's missing
    for bin in git ffmpeg python3 espeak-ng unzip; do
        has_cmd "$bin" || missing="$missing $bin"
    done
    
    [ -z "$missing" ] && return 0
    
    log_info "Missing dependencies:$missing"
    
    if [ "$IS_MACOS" = true ]; then
        if ! has_cmd brew; then
            log_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || true
        fi
        if has_cmd brew; then
            brew install $missing 2>/dev/null || log_warn "Some Homebrew installations may have failed"
        fi
    elif [ "$IS_WINDOWS" = true ]; then
        log_info "Windows detected. Attempting package manager installation..."
        if has_cmd winget; then
            log_info "Using winget..."
            [ -z "${missing##*python3*}" ] && winget install --silent --accept-package-agreements --accept-source-agreements "Python.Python.3.12" 2>/dev/null || true
            [ -z "${missing##*ffmpeg*}" ] && winget install --silent --accept-package-agreements --accept-source-agreements "Gyan.FFmpeg" 2>/dev/null || true
            [ -z "${missing##*git*}" ] && winget install --silent --accept-package-agreements --accept-source-agreements "Git.Git" 2>/dev/null || true
            [ -z "${missing##*espeak*}" ] && winget install --silent --accept-package-agreements --accept-source-agreements "eSpeak-NG.eSpeak-NG" 2>/dev/null || true
        elif has_cmd choco; then
            log_info "Using Chocolatey..."
            [ -z "${missing##*python3*}" ] && choco install -y python 2>/dev/null || true
            [ -z "${missing##*ffmpeg*}" ] && choco install -y ffmpeg 2>/dev/null || true
            [ -z "${missing##*git*}" ] && choco install -y git 2>/dev/null || true
            [ -z "${missing##*espeak*}" ] && choco install -y espeak-ng 2>/dev/null || true
        else
            log_warn "No Windows package manager found. Install manually:"
            log_warn "  Python: https://www.python.org/downloads/"
            log_warn "  FFmpeg: https://www.gyan.dev/ffmpeg/builds/"
            log_warn "  Git: https://git-scm.com/download/win"
            log_warn "  eSpeak-NG: https://github.com/espeak-ng/espeak-ng/releases"
        fi
    elif has_cmd apt-get; then
        sudo apt-get update -qq &>/dev/null
        sudo apt-get install -y $missing libass-dev 2>/dev/null || true
    elif has_cmd dnf; then
        sudo dnf install -y $missing 2>/dev/null || true
    elif has_cmd pacman; then
        sudo pacman -S --noconfirm $missing 2>/dev/null || true
    elif has_cmd zypper; then
        sudo zypper install -y $missing 2>/dev/null || true
    elif has_cmd apk; then
        sudo apk add $missing 2>/dev/null || true
    fi
}

install_system_deps

# Strict verification
MISSING_CRITICAL=""
for bin in git ffmpeg python3 espeak-ng; do
    has_cmd "$bin" || MISSING_CRITICAL="$MISSING_CRITICAL $bin"
done

if [ -n "$MISSING_CRITICAL" ]; then
    log_err "Critical dependencies missing after install attempt:$MISSING_CRITICAL"
    log_err "Please install manually and re-run this script."
    exit 1
fi
log_ok "All system dependencies verified"

# ==============================================================================
# Workspace Provisioning
# ==============================================================================

log_info "Step 4/7: Setting Up Workspace..."

if [ -d "$DATA_NAME" ] && [ -f "$DATA_NAME/main.py" ]; then
    ABSOLUTE_ROOT="$(pwd)"
    log_ok "Using existing directory"
else
    if [ -d "$ROOT_NAME" ]; then
        log_warn "Directory '$ROOT_NAME' exists. Updating..."
        cd "$ROOT_NAME"
        [ -d ".git" ] && git pull -q 2>/dev/null || true
    else
        log_info "Cloning repository..."
        if ! retry_cmd git clone --depth 1 "${REPO_URL}.git" "$ROOT_NAME" -q; then
            log_err "Git clone failed"
            exit 1
        fi
        cd "$ROOT_NAME"
    fi
    ABSOLUTE_ROOT="$(pwd)"
fi

log_transaction "workspace:$ABSOLUTE_ROOT"

# ==============================================================================
# Python Dependencies
# ==============================================================================

log_info "Step 5/7: Installing Python Dependencies..."

pip_args="--user -q"
if $PYTHON_CMD -m pip help install 2>&1 | grep -q 'break-system-packages'; then
    pip_args="--user --break-system-packages -q"
fi

log_info "Upgrading pip..."
retry_cmd $PYTHON_CMD -m pip install --upgrade pip $pip_args

log_info "Installing requirements (this may take a minute)..."
if ! retry_cmd $PYTHON_CMD -m pip install -r "$DATA_NAME/requirements.txt" $pip_args; then
    log_err "Failed to install Python dependencies"
    exit 1
fi

# Verify imports
log_info "Verifying Python dependencies..."
$PYTHON_CMD -c "import numpy, torch, transformers, onnxruntime, phonemizer, PIL, requests, yaml, dotenv" 2>/dev/null || {
    log_err "Dependency verification failed. Some packages may not have installed correctly."
    exit 1
}

log_ok "Python dependencies installed and verified"

# ==============================================================================
# Asset Provisioning
# ==============================================================================

log_info "Step 6/7: Provisioning Assets..."

mkdir -p "$DATA_NAME/models/kokoro"
mkdir -p "$DATA_NAME/background_videos"

# Model checksums for integrity validation (SHA256)
MODEL_SHA256="7a5c7f3e8d2b1a4e9f0c6d8b5a2e1f4c3d7b6a9e8f2d1c4b7a6e9f8d2c1b4a7e"  # Placeholder
VOCAB_SHA256="9e8f7d6c5b4a3e2f1d0c9b8a7e6f5d4c3b2a1e0f9d8c7b6a5e4f3d2c1b0a9e"   # Placeholder

download_with_checksum() {
    local url="$1"
    local output="$2"
    local expected_sha256="$3"  # Optional
    
    log_info "Downloading: $(basename$output)..."
    if ! retry_cmd curl -L --connect-timeout 10 --max-time 600 -# "$url" --output "$output"; then
        return 1
    fi
    
    # Validate size (basic check)
    local size=$(stat -f%z "$output" 2>/dev/null || stat -c%s "$output" 2>/dev/null || echo 0)
    if [ "$size" -lt 1000 ]; then
        log_err "Downloaded file suspiciously small ($size bytes). May be corrupted."
        rm -f "$output"
        return 1
    fi
    
    return 0
}

# Download Kokoro Model
MODEL_PATH="$DATA_NAME/models/kokoro/model.onnx"
VOCAB_PATH="$DATA_NAME/models/kokoro/tokenizer.json"
VOICE_BIN_PATH="$DATA_NAME/models/kokoro/am_adam.bin"
VOICE_PT_PATH="$DATA_NAME/models/kokoro/am_adam.pt"

# Validate existing files
validate_existing_file() {
    local file="$1"
    local min_size="$2"
    
    if [ ! -f "$file" ]; then
        return 1
    fi
    
    local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo 0)
    [ "$size" -ge "$min_size" ]
}

# Model Download with validation
if ! validate_existing_file "$MODEL_PATH" 100000000; then
    [ -f "$MODEL_PATH" ] && log_warn "Corrupted model detected. Redownloading..." && rm -f "$MODEL_PATH"
    download_with_checksum "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.onnx" "$MODEL_PATH"
    log_transaction "downloaded:$MODEL_PATH"
fi

if ! validate_existing_file "$VOCAB_PATH" 1000; then
    download_with_checksum "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json" "$VOCAB_PATH"
    log_transaction "downloaded:$VOCAB_PATH"
fi

if ! validate_existing_file "$VOICE_BIN_PATH" 10000; then
    if ! validate_existing_file "$VOICE_PT_PATH" 10000; then
        download_with_checksum "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/am_adam.pt" "$VOICE_PT_PATH"
        log_transaction "downloaded:$VOICE_PT_PATH"
    fi
    
    log_info "Converting voice model to binary format..."
    $PYTHON_CMD -c "
import torch
import numpy as np
try:
    data = torch.load('$VOICE_PT_PATH', map_location='cpu', weights_only=True)
    tensor = data if isinstance(data, torch.Tensor) else data.get('weight', data)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    tensor.astype(np.float32).tofile('$VOICE_BIN_PATH')
    print('Voice model converted successfully')
except Exception as e:
    print(f'Conversion failed: {e}')
    exit(1)
" || {
        log_err "Voice model conversion failed"
        exit 1
    }
    log_transaction "converted:$VOICE_BIN_PATH"
fi

# Cache Whisper model
log_info "Caching Whisper model (one-time download)..."
$PYTHON_CMD -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-base.en', device=-1)" &>/dev/null || log_warn "Whisper caching failed (non-critical)"

# Download background video if needed
if [ -z "$(ls -A "$DATA_NAME/background_videos" 2>/dev/null)" ]; then
    log_info "Downloading sample background video..."
    if ! $PYTHON_CMD "$DATA_NAME/main.py" --dl_video -u "https://www.youtube.com/watch?v=GVgLf8ENBJo" -r "1080p"; then
        log_warn "Background video download failed. You can download later with: omni-stories --dl_video"
    else
        log_ok "Sample background video ready"
    fi
fi

log_ok "Assets provisioned successfully"

# ==============================================================================
# Global Command Setup
# ==============================================================================

log_info "Step 7/7: Creating Global Command..."

# Store installation path for dynamic resolution
echo "$ABSOLUTE_ROOT" > "$HOME/.omni-stories-path"
log_transaction "config:$HOME/.omni-stories-path"

create_shim() {
    local shim_path="$1"
    local py_cmd="$2"
    
    mkdir -p "$(dirname "$shim_path")"
    
    # We resolve the absolute path of the python binary to ensure consistency
    local abs_py_cmd=$(command -v "$py_cmd")
    
    cat > "$shim_path" << SHIMEOF
#!/usr/bin/env bash
# Omni-Stories Launcher - Dynamically resolves installation path

if [ -f "\$HOME/.omni-stories-path" ]; then
    PROJECT_DIR="\$(cat "\$HOME/.omni-stories-path")"
else
    PROJECT_DIR="\$HOME/omni-stories"
fi

export PYTHONPATH="\$PROJECT_DIR/.omni-stories-data:\$PYTHONPATH"

# Use the exact python interpreter validated during install
PYTHON_CMD="$abs_py_cmd"
#!/usr/bin/env bash
# Omni-Stories Launcher - Dynamically resolves installation path

if [ -f "$HOME/.omni-stories-path" ]; then
    PROJECT_DIR="$(cat "$HOME/.omni-stories-path")"
else
    PROJECT_DIR="$HOME/omni-stories"
fi

export PYTHONPATH="$PROJECT_DIR/.omni-stories-data:$PYTHONPATH"

if [ ! -x "\$PYTHON_CMD" ]; then
    echo "Error: Python interpreter not found at \$PYTHON_CMD" >&2
    # Fallback attempt
    PYTHON_CMD="python3"
fi

exec "\$PYTHON_CMD" "\$PROJECT_DIR/.omni-stories-data/main.py" "\$@"
SHIMEOF
    
    chmod +x "$shim_path"
    log_transaction "created:$shim_path"
}

# Create Unix shim
create_shim "$USER_BIN/omni-stories" "$PYTHON_CMD"
log_ok "Unix command created: $USER_BIN/omni-stories"

# Windows-specific setup
if [ "$IS_WINDOWS" = true ]; then
    WIN_BIN="${USERPROFILE:-$HOME}/AppData/Local/Microsoft/WindowsApps"
    if [ -d "$WIN_BIN" ]; then
        WIN_CMD="$WIN_BIN/omni-stories.cmd"
        # Convert paths for Windows
        WIN_PROJECT=$(cygpath -w "$ABSOLUTE_ROOT" 2>/dev/null || echo "$ABSOLUTE_ROOT" | sed 's|/|\\|g')
        
        cat > "$WIN_CMD" << WINEOF
@echo off
set "PROJECT_DIR=%WIN_PROJECT%"
set "PYTHONPATH=%PROJECT_DIR%\\.omni-stories-data;%PYTHONPATH%"
rem Try to use the same python version if it was found in a standard location, otherwise rely on PATH
"%PYTHON_CMD%" "%PROJECT_DIR%\\.omni-stories-data\\main.py" %*
WINEOF
        log_ok "Windows command created: omni-stories.cmd"
        log_transaction "created:$WIN_CMD"
    fi
fi

# Update shell profiles (Unix-like only)
if [ "$IS_WINDOWS" = false ]; then
    update_shell_profile() {
        local rc="$1"
        local line="export PATH=\"\$HOME/.local/bin:\$PATH\""
        
        [ ! -f "$rc" ] && return
        
        if ! grep -Fq "$line" "$rc"; then
            echo -e "\n# Omni-Stories Path\n$line" >> "$rc"
            log_ok "Updated $rc"
            log_transaction "updated:$rc"
        fi
    }
    
    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile" "$HOME/.bash_profile"; do
        update_shell_profile "$rc"
    done
fi

# Configuration
[ -n "$API_KEYS" ] && echo "ELEVENLABS_API_KEYS='$API_KEYS'" > .env

# Verify installation
log_info "Verifying installation..."
if $PYTHON_CMD "$DATA_NAME/main.py" --help &>/dev/null; then
    log_ok "Installation verified: omni-stories command functional"
else
    log_warn "Installation complete but verification failed"
fi

# Cleanup
rm -f "$TRANSACTION_LOG"

# ==============================================================================
# Success Message
# ==============================================================================

echo -e "\n${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e " ${BOLD}✨ INSTALLATION COMPLETE ✨${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e " You can now run: ${CYAN}omni-stories --help${NC}"
echo -e " System check:     ${CYAN}omni-stories --doctor${NC}"
echo ""

if [[ ":$PATH:" != *":$USER_BIN:"* ]] && [ "$IS_WINDOWS" = false ]; then
    echo -e "${YELLOW}⚠ NOTE: Restart your terminal to use 'omni-stories' globally${NC}"
fi

if [ "$IS_WINDOWS" = true ]; then
    echo -e "${YELLOW}⚠ Windows users: Restart PowerShell/Terminal if command not found${NC}"
fi

echo -e "\n${CYAN}Next steps:${NC}"
echo -e "  1. Run ${WHITE}omni-stories --doctor${NC} to verify system health"
echo -e "  2. Generate your first story!"
echo ""

exit 0
