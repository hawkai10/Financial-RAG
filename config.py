import os
from dataclasses import dataclass
from dotenv import load_dotenv
import numpy as np


def sanitize_for_json(obj):
    """Sanitize data for JSON serialization by converting numpy types to Python types."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return 0.0
    else:
        return obj


load_dotenv()


@dataclass
class Config:
    """Minimal configuration used by the current model."""
    # API (optional; only used if present)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_API_URL: str = (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent"
    )

    # Files used by chunk cache and lookups
    CHUNKS_FILE: str = "contextualized_chunks.json"
    CONTEXTUALIZED_CHUNKS_JSON_PATH: str = "contextualized_chunks.json"

    def validate_config(self) -> None:
        """Lightweight, non-fatal validation (keeps defaults flexible)."""
        # Keep method for backward compatibility; no hard failures.
        return


# Create config (validation is intentionally non-fatal)
config = Config()
