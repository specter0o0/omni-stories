#!/usr/bin/env bash
# Omni-Stories Uninstaller
# Version: 1.0.1

set -e

# Terminal Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'
BOLD='\033[1m'

log_info() { echo -e "${CYAN}[I] $1${NC}"; }
log_ok()   { echo -e "${GREEN}[✔] $1${NC}"; }
log_warn() { echo -e "${YELLOW}[!] $1${NC}"; }
log_err()  { echo -e "${RED}[✖] $1${NC}"; }

echo -e "${RED}${BOLD}--- Omni-Stories Uninstaller ---${NC}"
echo -e "This will remove the application and optionally its data.\n"

# 1. Discover Directories
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$SCRIPT_DIR" == *".omni-stories-data" ]]; then
    ROOT_DIR=$(dirname "$SCRIPT_DIR")
else
    ROOT_DIR=$(pwd)
fi

# 2. confirmation
echo -en "${YELLOW}Are you sure you want to uninstall Omni-Stories? [y/N]: ${NC}"
read -r main_confirm
[[ $main_confirm != [yY] ]] && { echo "Aborted."; exit 0; }

# 3. Remove Global Shim
SHIM="$HOME/.local/bin/omni-stories"
if [ -f "$SHIM" ]; then
    rm "$SHIM"
    log_ok "Removed global command 'omni-stories'"
fi

# 4. Clean Shell Profiles
clean_path() {
    local rc="$1"
    local line="export PATH=\"\$HOME/.local/bin:\$PATH\""
    if [ -f "$rc" ]; then
        # Remove the specific line and the comment block above it
        if grep -Fq "$line" "$rc"; then
            # Use temporary file for safety
            sed -i "/# Omni-Stories Path/d" "$rc"
            sed -i "\|$line|d" "$rc"
            log_ok "Cleaned $rc"
        fi
    fi
}
log_info "Cleaning shell profiles..."
clean_path "$HOME/.bashrc"
clean_path "$HOME/.zshrc"
clean_path "$HOME/.profile"
clean_path "$HOME/.bash_profile"

# 5. Data Removal
echo -en "\n${CYAN}Remove engine data (.omni-stories-data)? [y/N]: ${NC}"
read -r data_confirm
if [[ $data_confirm == [yY] ]]; then
    rm -rf "$ROOT_DIR/.omni-stories-data"
    log_ok "Removed engine data."
fi

echo -en "${CYAN}Remove generated videos (output/)? [y/N]: ${NC}"
read -r out_confirm
if [[ $out_confirm == [yY] ]]; then
    rm -rf "$ROOT_DIR/output"
    log_ok "Removed output directory."
fi

echo -en "${CYAN}Remove configuration and secrets (config.yaml, .env)? [y/N]: ${NC}"
read -r config_confirm
if [[ $config_confirm == [yY] ]]; then
    rm -f "$ROOT_DIR/config.yaml" "$ROOT_DIR/.env"
    log_ok "Removed configuration files."
fi

echo -e "\n${GREEN}${BOLD}Omni-Stories has been successfully uninstalled.${NC}"
echo -e "${YELLOW}Note: You may need to restart your terminal for PATH changes to take effect.${NC}"

