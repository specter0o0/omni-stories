#!/usr/bin/env bash
# Omni-Stories Uninstaller
# Version: 1.1.0

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

# 3. Remove Global Shims and Config
SHIM="$HOME/.local/bin/omni-stories"
if [ -f "$SHIM" ]; then
    rm "$SHIM"
    log_ok "Removed global command 'omni-stories'"
fi

# Remove Windows shim if present
WIN_SHIM="$USERPROFILE/AppData/Local/Microsoft/WindowsApps/omni-stories.cmd"
if [ -f "$WIN_SHIM" ]; then
    rm "$WIN_SHIM"
    log_ok "Removed Windows command 'omni-stories.cmd'"
fi

# Remove path config file
if [ -f "$HOME/.omni-stories-path" ]; then
    rm "$HOME/.omni-stories-path"
    log_ok "Removed path configuration file"
fi

# 4. Clean Shell Profiles (macOS compatible)
clean_path() {
    local rc="$1"
    local line="export PATH=\"\$HOME/.local/bin:\$PATH\""
    
    [ ! -f "$rc" ] && return
    
    if grep -Fq "$line" "$rc"; then
        # Use temp file for macOS/Linux compatibility
        local temp_file="${rc}.tmp"
        grep -v "# Omni-Stories Path" "$rc" | grep -Fv "$line" > "$temp_file"
        mv "$temp_file" "$rc"
        log_ok "Cleaned $rc"
    fi
}
log_info "Cleaning shell profiles..."
clean_path "$HOME/.bashrc"
clean_path "$HOME/.zshrc"
clean_path "$HOME/.profile"
clean_path "$HOME/.bash_profile"

# 5. Data Removal with size reporting
get_dir_size() {
    local dir="$1"
    if [ -d "$dir" ]; then
        du -sh "$dir" 2>/dev/null | cut -f1 || echo "unknown"
    else
        echo "0"
    fi
}

echo -en "\n${CYAN}Remove engine data (.omni-stories-data)? [y/N]: ${NC}"
read -r data_confirm
if [[ $data_confirm == [yY] ]]; then
    data_size=$(get_dir_size "$ROOT_DIR/.omni-stories-data")
    log_info "Data size: $data_size"
    rm -rf "$ROOT_DIR/.omni-stories-data"
    log_ok "Removed engine data"
fi

echo -en "${CYAN}Remove generated videos (output/)? [y/N]: ${NC}"
read -r out_confirm
if [[ $out_confirm == [yY] ]]; then
    if [ -f "$ROOT_DIR/config.yaml" ] && [ -d "$ROOT_DIR/.omni-stories-data" ]; then
        output_size=$(get_dir_size "$ROOT_DIR/output")
        log_info "Output size: $output_size"
        rm -rf "$ROOT_DIR/output"
        log_ok "Removed output directory"
    else
        log_warn " Skipping 'output' deletion: Current directory does not appear to be the Omni-Stories root."
    fi
fi

echo -en "${CYAN}Remove configuration and secrets (config.yaml, .env)? [y/N]: ${NC}"
read -r config_confirm
if [[ $config_confirm == [yY] ]]; then
    # Create backup before deletion
    if [ -f "$ROOT_DIR/config.yaml" ] || [ -f "$ROOT_DIR/.env" ]; then
        backup_dir="$HOME/.omni-stories-backup-$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        [ -f "$ROOT_DIR/config.yaml" ] && cp "$ROOT_DIR/config.yaml" "$backup_dir/"
        [ -f "$ROOT_DIR/.env" ] && cp "$ROOT_DIR/.env" "$backup_dir/"
        log_info "Backup created: $backup_dir"
    fi
    
    rm -f "$ROOT_DIR/config.yaml" "$ROOT_DIR/.env"
    log_ok "Removed configuration files"
fi

echo -e "\n${GREEN}${BOLD}Omni-Stories has been successfully uninstalled.${NC}"
echo -e "${YELLOW}Note: You may need to restart your terminal for PATH changes to take effect.${NC}"


