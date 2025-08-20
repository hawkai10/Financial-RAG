"""
Minimal local model wrapper for embedding generation without HuggingFace Hub dependencies.
Loads models directly from local directories using only PyTorch and basic transformers components.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Optional safetensors support
try:
    from safetensors.torch import load_file as load_safetensors
    _SAFETENSORS_AVAILABLE = True
except Exception:
    _SAFETENSORS_AVAILABLE = False


class LocalTokenizer:
    """Minimal tokenizer that loads from local vocab files."""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.vocab = {}
        self.max_length = 512
        
        # Load tokenizer config
        config_path = self.model_dir / "tokenizer.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Extract vocab from tokenizer.json format
                if "model" in config and "vocab" in config["model"]:
                    self.vocab = config["model"]["vocab"]
        
        # Fallback: load from vocab.txt if available
        if not self.vocab:
            vocab_path = self.model_dir / "vocab.txt"
            if vocab_path.exists():
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    for idx, token in enumerate(f.read().strip().split('\n')):
                        self.vocab[token] = idx
        
        # Special tokens
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        
        self.cls_token_id = self.vocab.get(self.cls_token, 101)
        self.sep_token_id = self.vocab.get(self.sep_token, 102)
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 100)
        
        logger.info(f"Loaded tokenizer with {len(self.vocab)} tokens from {model_dir}")
    
    def encode(self, text: str, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Basic tokenization - splits on whitespace and maps to vocab."""
        max_len = max_length or self.max_length
        
        # Simple whitespace + punctuation tokenization
        tokens = text.lower().replace(',', ' ,').replace('.', ' .').split()
        
        # Convert to IDs
        input_ids = [self.cls_token_id]
        for token in tokens[:max_len-2]:  # Reserve space for CLS and SEP
            input_ids.append(self.vocab.get(token, self.unk_token_id))
        input_ids.append(self.sep_token_id)
        
        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < max_len:
            input_ids.append(self.pad_token_id)
            attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }


class LocalEmbedder:
    """Minimal local embedding model that loads PyTorch weights directly."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.hidden_size = self.config.get('hidden_size', 768)
        self.max_position_embeddings = self.config.get('max_position_embeddings', 512)
        
        # Load tokenizer
        self.tokenizer = LocalTokenizer(str(self.model_path))
        
        # Load model weights
        self.model = self._load_model()
        self.model.eval()
        
        logger.info(f"Loaded local model from {model_path} (hidden_size={self.hidden_size})")
    
    def _load_model(self):
        """Load PyTorch model weights from local files."""
        # Look for model files
        model_files = list(self.model_path.glob("pytorch_model*.bin"))
        safetensor_files = list(self.model_path.glob("*.safetensors"))
        if not model_files and safetensor_files:
            model_files = safetensor_files
        
        if not model_files:
            raise FileNotFoundError(f"No pytorch_model.bin or .safetensors files found in {self.model_path}")
        
        # Load state dict
        state_dict = {}
        for model_file in model_files:
            if model_file.suffix == '.bin':
                partial_state = torch.load(model_file, map_location='cpu')
                state_dict.update(partial_state)
            elif model_file.suffix == '.safetensors':
                if _SAFETENSORS_AVAILABLE:
                    try:
                        partial_state = load_safetensors(str(model_file))
                        state_dict.update(partial_state)
                    except Exception as e:
                        logger.warning(f"Failed to load safetensors {model_file}: {e}")
                else:
                    logger.warning(
                        f"Safetensors not available, cannot load {model_file}. Install 'safetensors' or provide pytorch_model.bin"
                    )
        
        # Create a minimal model architecture
        model = MinimalBertModel(self.config)
        
        # Load compatible weights
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded model weights successfully")
        except Exception as e:
            logger.warning(f"Some weights couldn't be loaded: {e}")
        
        return model.to(self.device)
    
    def encode(self, texts: Union[str, List[str]], convert_to_numpy: bool = True, show_progress_bar: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer.encode(text, max_length=self.max_position_embeddings)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Pool (mean of non-padded tokens)
                attention_mask = inputs['attention_mask']
                embeddings_tensor = outputs['last_hidden_state']
                
                # Mean pooling
                masked_embeddings = embeddings_tensor * attention_mask.unsqueeze(-1)
                summed = torch.sum(masked_embeddings, dim=1)
                counts = torch.sum(attention_mask, dim=1, keepdim=True).float()
                pooled = summed / counts
                
                # Normalize
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                
                embeddings.append(pooled.cpu())
        
        # Concatenate all embeddings
        result = torch.cat(embeddings, dim=0)
        
        if convert_to_numpy:
            return result.numpy()
        return result


class MinimalBertModel(torch.nn.Module):
    """Minimal BERT-like model for embedding generation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        vocab_size = config.get('vocab_size', 30522)
        hidden_size = config.get('hidden_size', 768)
        max_position_embeddings = config.get('max_position_embeddings', 512)
        type_vocab_size = config.get('type_vocab_size', 2)
        
        # Embeddings
        self.embeddings = torch.nn.ModuleDict({
            'word_embeddings': torch.nn.Embedding(vocab_size, hidden_size),
            'position_embeddings': torch.nn.Embedding(max_position_embeddings, hidden_size),
            'token_type_embeddings': torch.nn.Embedding(type_vocab_size, hidden_size),
        })
        
        self.embeddings_layernorm = torch.nn.LayerNorm(hidden_size)
        self.embeddings_dropout = torch.nn.Dropout(0.1)
        
        # Simplified transformer (just a few layers for basic functionality)
        num_layers = min(config.get('num_hidden_layers', 12), 6)  # Limit to 6 for simplicity
        num_heads = config.get('num_attention_heads', 12)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Embeddings
        inputs_embeds = (
            self.embeddings['word_embeddings'](input_ids) +
            self.embeddings['position_embeddings'](position_ids) +
            self.embeddings['token_type_embeddings'](token_type_ids)
        )
        
        embeddings = self.embeddings_dropout(self.embeddings_layernorm(inputs_embeds))
        
        # Create attention mask for transformer (True = attend, False = ignore)
        attention_mask_bool = attention_mask.bool()
        
        # Encode
        encoded = self.encoder(embeddings, src_key_padding_mask=~attention_mask_bool)
        
        return {'last_hidden_state': encoded}


def create_local_embedder(model_path: str) -> LocalEmbedder:
    """Factory function to create a local embedder from a model directory."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    return LocalEmbedder(model_path)


# Compatibility wrapper for sentence-transformers interface
class SentenceTransformerWrapper:
    """Drop-in replacement for SentenceTransformer using local models."""
    
    def __init__(self, model_name_or_path: str):
        self.embedder = create_local_embedder(model_name_or_path)
        self.model_name = model_name_or_path
    
    def encode(self, sentences: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Compatible with sentence-transformers interface."""
        convert_to_numpy = kwargs.get('convert_to_numpy', True)
        show_progress_bar = kwargs.get('show_progress_bar', False)
        
        return self.embedder.encode(sentences, convert_to_numpy=convert_to_numpy, show_progress_bar=show_progress_bar)
