#!/usr/bin/env bash
# Omni-Stories Uninstaller
# Version: 1.0.0

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${RED}--- Omni-Stories Uninstaller ---${NC}"
echo -en "${CYAN}Purge engine data and output? [y/N]: ${NC}"
read -r confirm

if [[ $confirm == [yY] ]]; then
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    ROOT_DIR=$(dirname "$SCRIPT_DIR")
    
    # Remove engine data and output
    rm -rf "$ROOT_DIR/.omni-stories-data"
    rm -rf "$ROOT_DIR/output"
    
    # Remove global shim
    SHIM="$HOME/.local/bin/omni-stories"
    if [ -f "$SHIM" ]; then
        rm "$SHIM"
        echo -e "${GREEN}✔ Global command 'omni-stories' removed.${NC}"
    fi

    echo -en "${CYAN}Remove configuration files (config.yaml, .env)? [y/N]: ${NC}"
    read -r config_confirm
    if [[ $config_confirm == [yY] ]]; then
        rm -f "$ROOT_DIR/config.yaml" "$ROOT_DIR/.env"
        echo -e "${GREEN}✔ Configuration removed.${NC}"
    fi
    
    echo -e "${GREEN}✔ Cleanup complete.${NC}"
else
    echo -e "Uninstall aborted."
fi
