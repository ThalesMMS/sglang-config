#!/bin/bash
#
# Setup script for SGLang Server
# Installs SGLang and dependencies for running local LLMs
# Tested on: Pop!_OS / Ubuntu 24.04 with NVIDIA RTX 5080
#
# Usage: sudo bash setup-sglang.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Check if running as root
[[ $EUID -ne 0 ]] && error "This script must be run as root (sudo)"

# Configuration
SGLANG_USER="${SGLANG_USER:-$SUDO_USER}"
[[ -z "$SGLANG_USER" ]] && error "Could not determine user. Set SGLANG_USER environment variable."
SGLANG_HOME="/home/$SGLANG_USER"
VENV_PATH="$SGLANG_HOME/sglang-env"
CUDA_VERSION="12-8"

log "Starting SGLang setup for user: $SGLANG_USER"

# ============================================================
# 1. Disable System Sleep (for SSH access)
# ============================================================
log "Configuring system to prevent sleep..."

systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target 2>/dev/null || true

mkdir -p /etc/systemd/logind.conf.d
cat > /etc/systemd/logind.conf.d/nosleep.conf << 'EOF'
[Login]
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
IdleAction=ignore
EOF

systemctl restart systemd-logind

# ============================================================
# 2. Install System Dependencies
# ============================================================
log "Installing system dependencies..."

apt update
apt install -y python3-pip python3-venv wget curl

# ============================================================
# 3. Install CUDA 12.8
# ============================================================
log "Checking CUDA installation..."

if ! command -v /usr/local/cuda-12.8/bin/nvcc &> /dev/null; then
    log "Installing CUDA $CUDA_VERSION..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt update
    apt install -y cuda-toolkit-$CUDA_VERSION
    rm -f cuda-keyring_1.1-1_all.deb
else
    log "CUDA 12.8 already installed"
fi

# ============================================================
# 4. Create Python Virtual Environment
# ============================================================
log "Creating Python virtual environment..."

if [[ -d "$VENV_PATH" ]]; then
    warn "Virtual environment already exists at $VENV_PATH"
else
    sudo -u "$SGLANG_USER" python3 -m venv "$VENV_PATH"
fi
sudo -u "$SGLANG_USER" "$VENV_PATH/bin/pip" install --upgrade pip

# ============================================================
# 5. Install SGLang
# ============================================================
log "Installing SGLang (this may take a few minutes)..."

# Install PyTorch with CUDA support first
sudo -u "$SGLANG_USER" "$VENV_PATH/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install SGLang with FlashInfer support
sudo -u "$SGLANG_USER" "$VENV_PATH/bin/pip" install "sglang[all]"
sudo -u "$SGLANG_USER" "$VENV_PATH/bin/pip" install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4

# ============================================================
# 6. Create systemd Service
# ============================================================
log "Creating systemd service..."

cat > /etc/systemd/system/sglang.service << EOF
[Unit]
Description=SGLang LLM Server
After=network.target

[Service]
Type=simple
User=root
Environment="PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin"
ExecStartPre=/bin/bash -c 'systemctl stop gdm3 cosmic-comp display-manager 2>/dev/null || true'
ExecStart=$VENV_PATH/bin/python -m sglang.launch_server --model-path unsloth/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8000 --context-length 65536 --max-running-requests 32 --mem-fraction-static 0.92 --dtype half --attention-backend flashinfer
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

# ============================================================
# 7. Summary
# ============================================================
log "Setup complete!"

echo ""
echo "============================================================"
echo "  SGLang Installation Complete"
echo "============================================================"
echo ""
echo "Available models:"
echo "  - Llama 3.2 3B Instruct (unsloth/Llama-3.2-3B-Instruct)"
echo "  - Phi-4-mini-instruct (microsoft/Phi-4-mini-instruct)"
echo ""
echo "To start the server manually:"
echo "  ~/sglang-config/start-sglang.sh"
echo ""
echo "To start as a service:"
echo "  sudo systemctl start sglang"
echo "  sudo systemctl enable sglang  # Auto-start on boot"
echo ""
echo "To test the server:"
echo "  ~/sglang-config/test-sglang.sh"
echo ""
echo "NOTE: First run will download model files (~6-8GB each)."
echo "============================================================"
