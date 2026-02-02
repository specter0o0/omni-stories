#!/usr/bin/env bash

# Omni-Stories v1.0.0 Setup Wizard
# Universal, Zero-Failure, Fully Automated.

set -e

# Terminal Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m'
BOLD='\033[1m'

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

clear
echo -e "$BANNER"
echo -e "${CYAN}${BOLD}Omni-Stories v1.0.0 Setup Wizard${NC}\n"

# 1. Platform Detection
SILENT=false
API_KEYS=""
for arg in "$@"; do
    if [[ "$arg" == "-s" || "$arg" == "--silent" ]]; then SILENT=true; fi
    if [[ "$arg" != "-"* && -n "$arg" ]]; then API_KEYS="$arg"; fi
done

ROOT_NAME="omni-stories"
DATA_NAME=".omni-stories-data"
LOCAL_BIN="$HOME/.local/bin"
REPO_URL="https://github.com/specter0o0/omni-stories"

install_system_deps() {
    MISSING=""
    for pkg in git ffmpeg python3; do
        if ! command -v $pkg &> /dev/null; then MISSING="$MISSING $pkg"; fi
    done

    if [ -n "$MISSING" ] || [ -n "$(command -v apt-get)" ]; then
        echo -e "${CYAN}Finalizing system dependencies...${NC}"

        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            command -v brew &> /dev/null || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            brew install git ffmpeg python3 espeak-ng
        elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
            # Windows
            if command -v winget &> /dev/null; then
                winget install --id=Git.Git -e --silent && winget install --id=Gyan.FFmpeg -e --silent && winget install --id=Python.Python.3.12 -e --silent
            else
                echo -e "${RED}Winget missing. Please install Git, FFmpeg, and Python 3 manually.${NC}"
                exit 1
            fi
        elif command -v apt-get &> /dev/null; then
            # Linux (Apt) - Including libass for captions, espeak-ng for local TTS
            sudo apt-get update && sudo apt-get install -y git ffmpeg python3 python3-pip libass-dev espeak-ng
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y git ffmpeg python3 python3-pip espeak-ng
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm git ffmpeg python python-pip espeak-ng
        elif command -v zypper &> /dev/null; then
            sudo zypper install -y git ffmpeg python3 python3-pip espeak-ng
        fi
    fi
}
install_system_deps

# 2. Workspace Setup
if [ -d "$DATA_NAME" ] && [ -f "$DATA_NAME/main.py" ]; then
    ABSOLUTE_ROOT="$(pwd)"
elif [ -d "$ROOT_NAME/$DATA_NAME" ]; then
    cd "$ROOT_NAME"
    ABSOLUTE_ROOT="$(pwd)"
else
    echo -e "${CYAN}Deploying Omni-Stories...${NC}"
    if command -v git &> /dev/null; then
        git clone "$REPO_URL.git" "$ROOT_NAME" || { echo -e "${RED}Clone failed.${NC}"; exit 1; }
    else
        mkdir -p "$ROOT_NAME"
        curl -L "$REPO_URL/archive/refs/heads/main.zip" -o "os.zip"
        unzip -q "os.zip" -d "os-tmp"
        mv "os-tmp/omni-stories-main/"* "$ROOT_NAME/"
        rm -rf "os-tmp" "os.zip"
    fi
    cd "$ROOT_NAME"
    ABSOLUTE_ROOT="$(pwd)"
fi

# 3. Python Optimization
PYTHON_CMD="python3"
for cmd in python3.12 python3.11 python3 python; do
    if command -v $cmd &> /dev/null; then
        if $cmd -c "import sys; exit(0) if sys.version_info >= (3, 9) else exit(1)" 2>/dev/null; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

pip_install() {
    if $PYTHON_CMD -m pip help install | grep -q 'break-system-packages'; then
        $PYTHON_CMD -m pip install --user --break-system-packages "$@"
    else
        $PYTHON_CMD -m pip install --user "$@"
    fi
}

echo -e "${CYAN}Installing Python requirements...${NC}"
pip_install --upgrade pip > /dev/null 2>&1 || true
pip_install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio || true
pip_install -r "$DATA_NAME/requirements.txt" || { echo -e "${RED}Dependency failure.${NC}"; exit 1; }

# 4. Interactive Configuration
if [ "$SILENT" = false ]; then
    if [ -z "$API_KEYS" ]; then
        echo -en "${WHITE}${BOLD}ElevenLabs API Key(s) [Comma-separated, optional]: ${NC}"
        # Ensure 'read' works when piped
        read -r API_KEYS < /dev/tty || true
    fi
fi
echo "ELEVENLABS_API_KEYS='$API_KEYS'" > .env

# 5. Asset & Model Provisioning (Silent)
mkdir -p "$DATA_NAME/models/kokoro"
MODEL_PATH="$DATA_NAME/models/kokoro/model.onnx"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${CYAN}Downloading Kokoro-TTS models...${NC}"
    curl -L -# "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.onnx" --output "$MODEL_PATH"
fi

echo -e "${CYAN}Caching AI models...${NC}"
$PYTHON_CMD -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-base.en')" > /dev/null 2>&1 || true

# Conditional background footage download
mkdir -p "$DATA_NAME/background_videos"
if [ -z "$(ls -A "$DATA_NAME/background_videos")" ]; then
    echo -e "${CYAN}Downloading default background clips...${NC}"
    $PYTHON_CMD "$DATA_NAME/main.py" --dl_video -u "https://www.youtube.com/watch?v=n_Dv4JH_G_E" -r "1080p" > /dev/null 2>&1 || true
fi

# 6. Command Delivery
mkdir -p "$LOCAL_BIN"
SHIM="$LOCAL_BIN/omni-stories"
cat <<EOF > "$SHIM"
#!/usr/bin/env bash
PROJECT_DIR="$ABSOLUTE_ROOT"
export PYTHONPATH="\$PROJECT_DIR/\$DATA_NAME:\$PYTHONPATH"
exec "$PYTHON_CMD" "\$PROJECT_DIR/$DATA_NAME/main.py" "\$@"
EOF
chmod +x "$SHIM"

inject_path() {
    local shell_rc="$1"
    if [ -f "$shell_rc" ]; then
        if ! grep -q "$LOCAL_BIN" "$shell_rc"; then
            echo -e "\n# Omni-Stories Bin\nexport PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$shell_rc"
        fi
    fi
}
inject_path "$HOME/.bashrc"
inject_path "$HOME/.zshrc"
[[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "msys" ]] && inject_path "$HOME/.profile"

export PATH="$LOCAL_BIN:$PATH"

# 7. Handover
echo -e "\n${GREEN}${BOLD}Setup Complete. v1.0.0 is ready for use.${NC}"
echo -e "${WHITE}Command: ${CYAN}omni-stories${NC}"
echo -e "Diagnostic: omni-stories --doctor"

if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
    echo -e "\n${YELLOW}Note: Restart your terminal or run: source $([ -f "$HOME/.zshrc" ] && echo "~/.zshrc" || echo "~/.bashrc")${NC}"
fi
