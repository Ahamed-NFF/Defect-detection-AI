# Windows / PowerShell setup notes

The team works on Windows. A few gotchas:

## Python env
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt --legacy-peer-deps   # note: legacy-peer-deps is npm; for pip just omit

For pip just run:
    pip install -r requirements.txt

## PyTorch + CUDA
Install the CUDA build that matches the lab GPUs (check `nvidia-smi`):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

## Ollama (for LLaVA) -- Member 4
1. Install Ollama for Windows: https://ollama.com/download
2. Pull the model:  ollama pull llava:7b
3. Ollama serves on http://localhost:11434 by default.

## Frontend
    cd frontend
    npm install --legacy-peer-deps
    npm run dev
