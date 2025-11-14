#!/bin/bash

echo "============================"
echo "  Starting Runpod Setup      "
echo "============================"

# ------------------------------
# 1. System Updates + Tools
# ------------------------------
apt-get update -y
apt-get install -y git wget unzip poppler-utils libgl1

# ------------------------------
# 2. Python Dependencies
# ------------------------------
pip install --upgrade pip

# Required for DeepSeek-OCR
pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==4.46.3" "tokenizers==0.20.3"
pip install einops addict easydict accelerate safetensors huggingface_hub

# PDF/Image processing
pip install pillow pdf2image python-docx tqdm

# ------------------------------
# 3. Flash-Attention (A100/H100 Optimized)
# ------------------------------
pip install flash-attn==2.7.3 --no-build-isolation

# ------------------------------
# 4. HuggingFace Settings
# ------------------------------
unset HF_HUB_OFFLINE
export HF_HUB_ENABLE_HF_TRANSFER=1

# Clean old caches
rm -rf ~/.cache/huggingface
rm -rf /root/.cache/huggingface

# ------------------------------
# 5. Clone Your Repo (Optional)
# ------------------------------
cd /workspace
if [ ! -d "deepseek-ocr" ]; then
    git clone https://github.com/SpectaclesGuy/deepseek-ocr.git
fi

cd deepseek-ocr

# ------------------------------
# 6. Final Output
# ------------------------------
echo "========================================"
echo " Setup complete! Run your app with: "
echo "   python3 app.py"
echo "========================================"
