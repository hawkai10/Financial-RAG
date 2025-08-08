#!/usr/bin/env python3
"""
Script to force rebuild the entire RAG system using marked.py.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from marked_pipeline_orchestrator import MarkedPipelineOrchestrator
from utils import logger

def main():
    """Force rebuild entire RAG system with marked.py."""
    try:
        logger.info("🔄 Force rebuilding entire RAG system with marked.py...")
        
        orchestrator = MarkedPipelineOrchestrator()
        orchestrator.run_full_pipeline(force_rebuild=True)
        
        logger.info("✅ Force rebuild completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Force rebuild failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
