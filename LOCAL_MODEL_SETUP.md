# Local Model Setup for Financial RAG

This document explains how to set up local models to avoid HuggingFace Hub dependencies.

## Overview

The system now includes a `local_embedder.py` module that can load models directly from local directories, eliminating the need for online downloads and resolving version conflicts.

## Local Model Structure

Each model directory should contain:
```
model_directory/
├── config.json          # Model configuration
├── tokenizer.json       # Tokenizer configuration  
├── vocab.txt           # Vocabulary (alternative to tokenizer.json)
├── pytorch_model.bin    # Model weights (PyTorch format)
└── tokenizer_config.json # Additional tokenizer config (optional)
```

## Setting Up Local Models

### 1. Obtain Model Files

You can get model files by:

**Option A: Download from HuggingFace (one-time)**
```bash
# Install huggingface_hub temporarily
pip install huggingface_hub

# Download models to local directories
python -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-small-en-v1.5', local_dir='C:/models/bge-small-en-v1.5')
snapshot_download('thenlper/gte-small', local_dir='C:/models/gte-small') 
snapshot_download('cross-encoder/ms-marco-MiniLM-L-12-v2', local_dir='C:/models/msmarco-MiniLM-L-12-v2')
"

# Uninstall to avoid conflicts
pip uninstall huggingface_hub -y
```

**Option B: Copy from another machine**
- Download models on a machine with internet access
- Copy the model directories to your local machine

### 2. Environment Variables

Set these environment variables to use local models:

```powershell
# PowerShell
$env:EMBED_BAAI_PATH="C:\models\bge-small-en-v1.5"
$env:EMBED_GTE_PATH="C:\models\gte-small"
$env:CROSS_ENCODER_PATH="C:\models\msmarco-MiniLM-L-12-v2"

# For ensemble mode (optional)
$env:ENSEMBLE_ENCODERS="BAAI/bge-small-en-v1.5, thenlper/gte-small"
$env:EMBEDDER_PATHS="C:\models\bge-small-en-v1.5, C:\models\gte-small"
```

```bash
# Bash
export EMBED_BAAI_PATH="/path/to/models/bge-small-en-v1.5"
export EMBED_GTE_PATH="/path/to/models/gte-small"
export CROSS_ENCODER_PATH="/path/to/models/msmarco-MiniLM-L-12-v2"
```

### 3. Dependencies

With local models, you only need these core dependencies:
```
torch>=1.9.0
numpy>=1.20.0
```

The system will automatically fall back to the local embedder if sentence-transformers is not available.

## Testing

Test your local model setup:

```powershell
python test_local_embedder.py
```

This will verify that models can be loaded and generate embeddings.

## Fallback Behavior

The system includes graceful fallbacks:

1. **If sentence-transformers is available**: Uses it normally
2. **If sentence-transformers fails to import**: Falls back to local embedder
3. **If local model paths don't exist**: Shows helpful error messages
4. **If model loading fails**: Degrades gracefully with warnings

## Benefits

- ✅ No internet required after setup
- ✅ No version conflicts with transformers/huggingface_hub
- ✅ Faster loading (no network overhead)
- ✅ More predictable behavior
- ✅ Works in air-gapped environments

## Limitations

- Cross-encoder reranking requires sentence-transformers (for now)
- Local embedder is simplified - may have slight accuracy differences
- Initial setup requires downloading models once
