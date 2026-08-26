#!/usr/bin/env bash
set -e

echo ""
echo "  ============================================="
echo "     Watermark Slayer Setup (Linux/macOS)"
echo "  ============================================="
echo ""

# China mirror configuration
CHINA_MODE=0
PIP_MIRROR=""
HF_ENDPOINT=""

# Check if user is in China (for mirror selection)
echo "  [?] Are you in China? (y/n)"
echo "      This will use faster mirrors for downloads"
read -p "      " -n 1 -r china_choice
echo
if [[ $china_choice =~ ^[Yy]$ ]]; then
    CHINA_MODE=1
    PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
    HF_ENDPOINT="https://hf-mirror.com"
    echo "  [OK] Using China mirrors (Tsinghua PyPI + HF-Mirror)"
else
    echo "  [OK] Using default mirrors"
fi
echo ""

# Detect OS
OS_TYPE="linux"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
    echo "  [*] Detected macOS"
else
    echo "  [*] Detected Linux"
fi

# Download LaMA model directly from GitHub (avoids iopaint CLI dependency on fastapi)
echo "  [*] Downloading LaMA model (~196MB)..."
LAMA_DIR="$HOME/.cache/torch/hub/checkpoints"
LAMA_FILE="$LAMA_DIR/big-lama.pt"
if [ ! -f "$LAMA_FILE" ]; then
    mkdir -p "$LAMA_DIR"
    curl -L -o "$LAMA_FILE" "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt" || echo "  [!] LaMA download failed, will retry on first use"
    echo "  [OK] LaMA model downloaded"
else
    echo "  [OK] LaMA model already exists"
fi

# Download Florence-2 model
echo "  [*] Downloading Florence-2 model (~1.5GB)..."
if [ "$CHINA_MODE" == "1" ]; then
    echo "      Using HF-Mirror for faster download in China"
    HF_ENDPOINT="$HF_ENDPOINT" python -c "import os; os.environ['HF_ENDPOINT']='$HF_ENDPOINT'; from huggingface_hub import snapshot_download; snapshot_download('florence-community/Florence-2-large', local_dir_use_symlinks=False)" || echo "  [!] Florence-2 download failed, will retry on first use"
else
    python -c "from huggingface_hub import snapshot_download; snapshot_download('florence-community/Florence-2-large', local_dir_use_symlinks=False)" || echo "  [!] Florence-2 download failed, will retry on first use"
fi
