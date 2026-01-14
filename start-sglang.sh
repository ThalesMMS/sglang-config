#!/bin/bash
#
# Start script for SGLang Server
# Supports multiple models: Llama 3.2 3B and Qwen 2.5 7B
#
# Usage:
#   ./start-sglang.sh              # Interactive menu
#   ./start-sglang.sh llama        # Direct Llama 3.2 3B
#   ./start-sglang.sh qwen         # Direct Qwen 2.5 7B
#

set -e

# Check if running as root; otherwise re-exec with sudo
if [ "$EUID" -ne 0 ]; then
    echo "Elevating privileges to root..."
    exec sudo "$0" "$@"
fi

# Capture the original user's home directory
ORIGINAL_USER="${SUDO_USER:-$USER}"
ORIGINAL_HOME=$(eval echo ~$ORIGINAL_USER)

# Output colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Model configuration
declare -A MODELS
MODELS[llama]="unsloth/Llama-3.2-3B-Instruct"
MODELS[qwen]="Qwen/Qwen2.5-7B-Instruct-AWQ"

declare -A MODEL_CONFIGS
# Format: "context_length:max_running_requests:mem_fraction_static"
# OPTIMIZED FOR RTX 5080 (16GB VRAM)
MODEL_CONFIGS[llama]="65536:2:0.90"     # 3B model - 64K context, minimal concurrency
MODEL_CONFIGS[qwen]="32768:1:0.85"      # 7B AWQ model - 16K context, 1 request (faster)

# Extra flags for memory optimization (model-specific)
declare -A EXTRA_FLAGS
EXTRA_FLAGS[llama]=""
EXTRA_FLAGS[qwen]=""  # AWQ model is small enough, no extra optimizations needed

# Tool call parsers for function calling support
# See: https://docs.sglang.ai/advanced_features/tool_parser.html
declare -A TOOL_PARSERS
TOOL_PARSERS[llama]="llama3"            # For Llama 3.1/3.2/3.3 models
TOOL_PARSERS[qwen]="qwen"               # For Qwen 2.5 models - excellent tool calling

declare -A MODEL_NAMES
MODEL_NAMES[llama]="Llama 3.2 3B Instruct (Meta)"
MODEL_NAMES[qwen]="Qwen 2.5 7B Instruct AWQ (Alibaba)"

show_menu() {
    echo -e "${CYAN}=== SGLang Server - Model Selection ===${NC}"
    echo ""
    echo "Available models:"
    echo ""
    echo -e "  ${GREEN}1)${NC} Llama 3.2 3B Instruct (Meta)"
    echo "     - 3B params, ${GREEN}native tool calling${NC}"
    echo "     - VRAM: ~6GB | Context: 64K tokens"
    echo ""
    echo -e "  ${GREEN}2)${NC} Qwen 2.5 7B Instruct AWQ (Alibaba)"
    echo "     - 7B params quantized 4-bit, ${GREEN}excellent tool calling${NC}"
    echo "     - VRAM: ~4GB | Context: 32K tokens"
    echo ""
    echo -e "  ${YELLOW}0)${NC} Exit"
    echo ""
}

select_model() {
    while true; do
        show_menu
        read -p "Choose a model [1/2/0]: " choice
        case $choice in
            1) MODEL_KEY="llama"; break ;;
            2) MODEL_KEY="qwen"; break ;;
            0) echo "Exiting."; exit 0 ;;
            *) echo -e "${YELLOW}Invalid option. Try again.${NC}"; echo "" ;;
        esac
    done
}

stop_existing_sglang() {
    # Check processes using the GPU
    local gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')

    # Check "sglang.launch_server" process
    local serve_pids=$(pgrep -f "sglang.launch_server" 2>/dev/null || true)

    if [ -n "$gpu_pids" ] || [ -n "$serve_pids" ]; then
        echo -e "${YELLOW}SGLang instance detected. Shutting down...${NC}"

        # Kill "sglang.launch_server" specifically
        if [ -n "$serve_pids" ]; then
            echo "$serve_pids" | xargs kill -9 2>/dev/null || true
        fi
        sleep 2

        # Kill processes still using the GPU
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
        for pid in $gpu_pids; do
            echo "Killing GPU process: $pid"
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 2

        echo "Previous instance shut down."
    fi

    # Check if memory was freed
    local free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$free_mem" ] && [ "$free_mem" -lt 10000 ]; then
        echo -e "${YELLOW}Warning: GPU still low on free memory (${free_mem} MiB). Waiting...${NC}"
        sleep 3
    fi
}

start_server() {
    local model_key=$1
    local model_id="${MODELS[$model_key]}"
    local model_name="${MODEL_NAMES[$model_key]}"
    local config="${MODEL_CONFIGS[$model_key]}"
    local tool_parser="${TOOL_PARSERS[$model_key]}"
    local extra_flags="${EXTRA_FLAGS[$model_key]}"

    # Parse config
    IFS=':' read -r context_length max_running_requests mem_fraction_static <<< "$config"

    echo ""
    echo -e "${GREEN}=== Starting $model_name ===${NC}"
    echo ""

    # Stop any existing SGLang instance
    stop_existing_sglang

    # Stop display manager to free VRAM
    echo "Stopping display manager..."
    sudo systemctl stop gdm3 cosmic-comp display-manager 2>/dev/null || true

    # Check available GPU memory
    echo "Free GPU memory:"
    nvidia-smi --query-gpu=memory.free --format=csv,noheader
    echo ""

    # Activate virtual environment
    echo "Activating virtual environment..."
    source $ORIGINAL_HOME/sglang-env/bin/activate

    # Build command
    # Note: FlashInfer removed - incompatible with PyTorch 2.9
    cmd="python -m sglang.launch_server --model-path $model_id --host 0.0.0.0 --port 8001 --context-length $context_length --max-running-requests $max_running_requests --mem-fraction-static $mem_fraction_static --dtype half"

    # Add tool call parser if available for this model
    if [ -n "$tool_parser" ]; then
        cmd="$cmd --tool-call-parser $tool_parser"
        echo -e "${CYAN}Tool calling enabled with parser: $tool_parser${NC}"
    else
        echo -e "${YELLOW}Note: Tool calling not available for this model${NC}"
    fi

    # Add extra flags for memory optimization
    if [ -n "$extra_flags" ]; then
        cmd="$cmd $extra_flags"
        echo -e "${CYAN}Memory optimizations: $extra_flags${NC}"
    fi

    echo ""
    echo "Command:"
    echo "$cmd"
    echo ""
    echo -e "${GREEN}Starting server on port 8001...${NC}"
    echo ""

    # Run
    eval $cmd
}

# Main
if [ -n "$1" ]; then
    # Argument passed directly
    case "$1" in
        llama|llama32|1) MODEL_KEY="llama" ;;
        qwen|qwen25|2) MODEL_KEY="qwen" ;;
        *) echo "Usage: $0 [llama|qwen]"; exit 1 ;;
    esac
else
    # Interactive menu
    select_model
fi

start_server "$MODEL_KEY"
