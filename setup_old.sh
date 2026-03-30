#!/bin/bash
# Setup script for clean environment with vLLM >= 0.12.0
# Requires: CUDA 12.4+, Python 3.12

set -e

ENV_NAME="${1:-sdpo2}"

echo "Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.12 -y
conda activate "$ENV_NAME"

# Install PyTorch with CUDA 12.4
echo "Installing PyTorch..."
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM 0.12.0
echo "Installing vLLM 0.12.0..."
pip install vllm==0.12.0

# Install Flash Attention
echo "Installing Flash Attention..."
pip install flash-attn --no-build-isolation

# Install SDPO dependencies
echo "Installing SDPO dependencies..."
pip install -r requirements.txt

# Install SDPO in editable mode
echo "Installing SDPO..."
pip install -e .

echo ""
echo "Done! Activate with: conda activate $ENV_NAME"
echo ""
echo "Verify installation:"
echo "  python -c \"import vllm; print('vLLM:', vllm.__version__)\""
echo "  python -c \"import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())\""
