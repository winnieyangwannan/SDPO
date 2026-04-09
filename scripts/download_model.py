#!/usr/bin/env python3
"""Download Qwen3-8B model from HuggingFace."""

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    model_id = "Qwen/Qwen3-8B"
    local_dir = "/checkpoint/agentic-models/winnieyangwn/models/Qwen3-8B"
    
    print(f"Downloading {model_id} to {local_dir}...")
    
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    
    print(f"Done! Model saved to {local_dir}")
