#!/usr/bin/env python3
"""
Enhanced Pipeline Orchestrator using marked.py as the main chunking engine.
This replaces the existing auto_parse_folder.py pipeline with the superior marked.py system.
"""

import os
import sys
import json
import sqlite3
import hashlib
import threading
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import shutil
from datetime import datetime

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import logger
from marked import main as process_documents_marked
from config import Config

class MarkedPipelineOrchestrator:
    """Enhanced Pipeline Orchestrator using marked.py as the main chunking engine."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.source_dir = "Source_Documents"
        self.manifest_file = "processing_manifest.json"
        self.contextualized_json = self.config.CONTEXTUALIZED_CHUNKS_JSON
        self.chunks_db = "chunks.db"
        self.embeddings_index = self.config.INDEX_PATH
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.check_interval = 30  # Check every 30 seconds
        
        # Ensure directories exist
        os.makedirs(self.source_dir, exist_ok=True)
        os.makedirs(self.embeddings_index, exist_ok=True)
        
        logger.info("MarkedPipelineOrchestrator initialized with marked.py as main chunking engine")
    
    def load_manifest(self) -> Dict:
        """Load processing manifest to track processed files."""
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load manifest: {e}")
                return {"files": {}, "last_updated": None}
        return {"files": {}, "last_updated": None}
    
    def save_manifest(self, manifest: Dict):
        """Save processing manifest."""
        try:
            manifest["last_updated"] = datetime.now().isoformat()
            with open(self.manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Could not save manifest: {e}")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Could not calculate hash for {file_path}: {e}")
            return ""
    
    def scan_source_directory(self) -> Dict[str, Dict]:
        """Scan source directory for PDF files and their metadata."""
        files_info = {}
        
        if not os.path.exists(self.source_dir):
            logger.warning(f"Source directory {self.source_dir} does not exist")
            return files_info
        
        for file_path in Path(self.source_dir).rglob("*.pdf"):
            try:
                stat = file_path.stat()
                relative_path = str(file_path.relative_to(self.source_dir))
                
                files_info[relative_path] = {
                    "full_path": str(file_path),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "hash": self.calculate_file_hash(str(file_path))
                }
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        return files_info
    
    def detect_changes(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Detect changes in source directory.
        Returns: (new_files, modified_files, deleted_files)
        """
        manifest = self.load_manifest()
        old_files = manifest.get("files", {})
        current_files = self.scan_source_directory()
        
        new_files = set()
        modified_files = set()
        deleted_files = set()
        
        # Check for new and modified files
        for file_path, file_info in current_files.items():
            if file_path not in old_files:
                new_files.add(file_path)
            elif (old_files[file_path].get("hash") != file_info["hash"] or 
                  old_files[file_path].get("modified") != file_info["modified"]):
                modified_files.add(file_path)
        
        # Check for deleted files
        for file_path in old_files:
            if file_path not in current_files:
                deleted_files.add(file_path)
        
        return new_files, modified_files, deleted_files
    
    def remove_chunks_from_db(self, file_paths: Set[str]):
        """Remove chunks for deleted files from the database."""
        if not os.path.exists(self.chunks_db):
            return
        
        try:
            conn = sqlite3.connect(self.chunks_db)
            cursor = conn.cursor()
            
            for file_path in file_paths:
                # Remove chunks for this document
                cursor.execute("DELETE FROM chunks WHERE document_name = ?", (file_path,))
                logger.info(f"Removed chunks for deleted file: {file_path}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error removing chunks from database: {e}")
    
    def remove_chunks_from_json(self, file_paths: Set[str]):
        """Remove chunks for deleted/modified files from JSON."""
        if not os.path.exists(self.contextualized_json):
            return
        
        try:
            with open(self.contextualized_json, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Remove chunks for specified files
            original_count = len(chunks)
            chunks = [chunk for chunk in chunks 
                     if chunk.get("document_name", "") not in file_paths]
            
            # Save updated chunks
            with open(self.contextualized_json, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            removed_count = original_count - len(chunks)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} chunks from JSON for modified/deleted files")
                
        except Exception as e:
            logger.error(f"Error removing chunks from JSON: {e}")
    
    def process_files_with_marked(self, file_paths: Set[str]) -> List[Dict]:
        """Process files using marked.py chunking engine."""
        if not file_paths:
            return []
        
        logger.info(f"Processing {len(file_paths)} files with marked.py engine")
        
        # Create a temporary list of full file paths for marked.py
        full_paths = []
        current_files = self.scan_source_directory()
        
        for relative_path in file_paths:
            if relative_path in current_files:
                full_paths.append(current_files[relative_path]["full_path"])
        
        if not full_paths:
            logger.warning("No valid file paths found for processing")
            return []
        
        try:
            # Backup existing chunks file if it exists
            if os.path.exists(self.contextualized_json):
                backup_path = f"{self.contextualized_json}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(self.contextualized_json, backup_path)
                logger.info(f"Backed up existing chunks to: {backup_path}")
            
            # Use marked.py to process the documents
            # The marked.py main function will handle all the processing and save to contextualized_chunks.json
            logger.info("Running marked.py processing pipeline...")
            
            # Import and run marked.py processing
            from marked import main as marked_main
            
            # Temporarily modify sys.argv to pass file paths to marked.py
            original_argv = sys.argv.copy()
            try:
                # Set up arguments for marked.py
                sys.argv = ["marked.py"] + full_paths
                
                # Run marked.py processing
                marked_main()
                
                logger.info("✅ marked.py processing completed successfully")
                
            finally:
                # Restore original sys.argv
                sys.argv = original_argv
            
            # Load the processed chunks
            processed_chunks = []
            if os.path.exists(self.contextualized_json):
                with open(self.contextualized_json, 'r', encoding='utf-8') as f:
                    processed_chunks = json.load(f)
                
                logger.info(f"✅ Loaded {len(processed_chunks)} processed chunks from marked.py")
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"❌ Error processing files with marked.py: {e}")
            
            # Restore backup if processing failed
            backup_files = [f for f in os.listdir('.') if f.startswith(f"{self.contextualized_json}.backup")]
            if backup_files:
                latest_backup = max(backup_files)
                shutil.copy2(latest_backup, self.contextualized_json)
                logger.info(f"Restored backup from: {latest_backup}")
            
            return []
    
    def rebuild_database(self):
        """Rebuild the chunks database from the JSON file."""
        try:
            from init_chunks_db import create_chunks_database
            
            logger.info("Rebuilding chunks database from JSON...")
            success = create_chunks_database(self.chunks_db, self.contextualized_json)
            
            if success:
                logger.info("✅ Chunks database rebuilt successfully")
            else:
                logger.error("❌ Failed to rebuild chunks database")
                
        except Exception as e:
            logger.error(f"❌ Error rebuilding database: {e}")
    
    def rebuild_embeddings_index(self):
        """Rebuild the embeddings index."""
        try:
            logger.info("Rebuilding embeddings index...")
            
            # Import and run the embeddings script
            import subprocess
            result = subprocess.run([
                sys.executable, "embed_chunks_txtai.py"
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                logger.info("✅ Embeddings index rebuilt successfully")
            else:
                logger.error(f"❌ Failed to rebuild embeddings index: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ Error rebuilding embeddings index: {e}")
    
    def run_full_pipeline(self, force_rebuild: bool = False):
        """Run the complete pipeline with marked.py as the chunking engine."""
        logger.info("🚀 Starting Enhanced Pipeline with marked.py chunking engine")
        
        try:
            # Detect changes
            new_files, modified_files, deleted_files = self.detect_changes()
            
            # Log detected changes
            if new_files:
                logger.info(f"📄 New files detected: {len(new_files)}")
                for file in new_files:
                    logger.info(f"  + {file}")
            
            if modified_files:
                logger.info(f"📝 Modified files detected: {len(modified_files)}")
                for file in modified_files:
                    logger.info(f"  ~ {file}")
            
            if deleted_files:
                logger.info(f"🗑️ Deleted files detected: {len(deleted_files)}")
                for file in deleted_files:
                    logger.info(f"  - {file}")
            
            # Handle deletions
            if deleted_files:
                self.remove_chunks_from_json(deleted_files)
                self.remove_chunks_from_db(deleted_files)
            
            # Process new and modified files
            files_to_process = new_files | modified_files
            
            if files_to_process or force_rebuild:
                if force_rebuild:
                    logger.info("🔄 Force rebuild requested - processing all files")
                    current_files = self.scan_source_directory()
                    files_to_process = set(current_files.keys())
                    
                    # Clear existing chunks for force rebuild
                    if os.path.exists(self.contextualized_json):
                        os.remove(self.contextualized_json)
                        logger.info("🗑️ Cleared existing chunks for force rebuild")
                
                if files_to_process:
                    # Remove chunks for modified files from JSON
                    if modified_files and not force_rebuild:
                        self.remove_chunks_from_json(modified_files)
                    
                    # Process files with marked.py
                    processed_chunks = self.process_files_with_marked(files_to_process)
                    
                    if processed_chunks:
                        # Rebuild database
                        self.rebuild_database()
                        
                        # Rebuild embeddings index
                        self.rebuild_embeddings_index()
                        
                        # Update manifest
                        manifest = self.load_manifest()
                        current_files = self.scan_source_directory()
                        
                        for file_path in files_to_process:
                            if file_path in current_files:
                                manifest["files"][file_path] = current_files[file_path]
                        
                        # Remove deleted files from manifest
                        for file_path in deleted_files:
                            manifest["files"].pop(file_path, None)
                        
                        self.save_manifest(manifest)
                        
                        logger.info("✅ Pipeline completed successfully with marked.py chunking engine")
                    else:
                        logger.error("❌ No chunks were processed - pipeline failed")
                
            else:
                logger.info("✅ No changes detected - pipeline up to date")
        
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def start_background_monitoring(self):
        """Start background monitoring of source directory for changes."""
        if self.monitoring_active:
            logger.info("📡 Background monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"📡 Started background monitoring (checking every {self.check_interval} seconds)")
    
    def stop_background_monitoring(self):
        """Stop background monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            logger.info("🛑 Stopped background monitoring")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        logger.info("🔍 Background monitoring started")
        
        while self.monitoring_active:
            try:
                # Check for changes
                new_files, modified_files, deleted_files = self.detect_changes()
                
                if new_files or modified_files or deleted_files:
                    logger.info(f"🔄 Changes detected: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_files)} deleted")
                    
                    # Run pipeline to process changes
                    self.run_full_pipeline(force_rebuild=False)
                    
                    logger.info("✅ Background processing completed")
                
                # Wait before next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"❌ Background monitoring error: {e}")
                time.sleep(self.check_interval)  # Continue monitoring despite errors
    
    def is_monitoring_active(self) -> bool:
        """Check if background monitoring is active."""
        return self.monitoring_active

def main():
    """Main function to run the enhanced pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Pipeline Orchestrator with marked.py")
    parser.add_argument("--force-rebuild", action="store_true", 
                       help="Force rebuild of entire pipeline")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Create and run orchestrator
    orchestrator = MarkedPipelineOrchestrator(config)
    orchestrator.run_full_pipeline(force_rebuild=args.force_rebuild)

if __name__ == "__main__":
    main()

# Global orchestrator instance for API server integration
_global_orchestrator = None

def get_global_orchestrator() -> MarkedPipelineOrchestrator:
    """Get or create the global orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = MarkedPipelineOrchestrator()
    return _global_orchestrator

def start_background_monitoring():
    """Start background monitoring (for API server integration)."""
    orchestrator = get_global_orchestrator()
    orchestrator.start_background_monitoring()

def stop_background_monitoring():
    """Stop background monitoring (for API server integration)."""
    global _global_orchestrator
    if _global_orchestrator:
        _global_orchestrator.stop_background_monitoring()

def is_monitoring_active() -> bool:
    """Check if background monitoring is active."""
    global _global_orchestrator
    if _global_orchestrator:
        return _global_orchestrator.is_monitoring_active()
    return False
