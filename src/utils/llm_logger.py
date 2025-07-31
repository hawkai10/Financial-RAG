#!/usr/bin/env python3
"""
LLM Interaction Logger
Creates a separate log file specifically for LLM prompts and responses
"""

import logging
import os
from datetime import datetime

# Create LLM-specific logger
llm_logger = logging.getLogger('llm_interactions')
llm_logger.setLevel(logging.INFO)

# Create file handler for LLM logs
llm_log_file = f"llm_interactions_{datetime.now().strftime('%Y%m%d')}.log"
llm_handler = logging.FileHandler(llm_log_file, encoding='utf-8')
llm_handler.setLevel(logging.INFO)

# Create formatter for LLM logs
llm_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
llm_handler.setFormatter(llm_formatter)

# Add handler to logger
llm_logger.addHandler(llm_handler)

def log_llm_interaction(phase: str, content: str, **kwargs):
    """Log LLM interactions with structured format - COMPLETELY DISABLED"""
    # LLM LOGGING COMPLETELY DISABLED: No file or console logging per user request
    # Function has been completely disabled - all calls are ignored
    return

# Export the logger for use in other modules
__all__ = ['log_llm_interaction', 'llm_logger']
