#!/usr/bin/env python3
"""
Quick script to run marked.py processing on all documents.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from marked_pipeline_orchestrator import MarkedPipelineOrchestrator
from utils import logger

def main():
    """Run marked.py processing pipeline."""
    try:
        logger.info("🚀 Starting marked.py processing pipeline...")
        
        orchestrator = MarkedPipelineOrchestrator()
        orchestrator.run_full_pipeline(force_rebuild=False)
        
        logger.info("✅ marked.py processing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ marked.py processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
