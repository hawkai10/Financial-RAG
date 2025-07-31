"""
Pipeline Orchestrator for DocuChat AI
Manages the entire data pipeline: chunking, contextualization, and embedding
Ensures all data is up-to-date before serving the API
"""

import os
import sys
import json
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

# Project imports
from ..utils.paths import (
    SOURCE_MANIFEST_JSON, CHUNKS_DB, CONTEXTUALIZED_CHUNKS_JSON,
    SOURCE_DOCUMENTS_DIR, BUSINESS_DOCS_INDEX_DIR, EMBEDDINGS_DIR
)

# Configuration constants
MANIFEST_FILE = str(SOURCE_MANIFEST_JSON)
CHUNKS_DB_FILE = str(CHUNKS_DB)
CONTEXTUALIZED_JSON = str(CONTEXTUALIZED_CHUNKS_JSON)
from tqdm import tqdm

# Configuration
SOURCE_DIR = Path(r"C:\Users\arvin\OneDrive\Desktop\trial\docling\Source_Documents")
MANIFEST_FILE = "source_manifest.json"
CHUNKS_DB = "chunks.db"
CONTEXTUALIZED_JSON = "contextualized_chunks.json"
EMBEDDINGS_INDEX = "business-docs-index"

class PipelineOrchestrator:
    """Manages the complete data pipeline from documents to embeddings."""
    
    def __init__(self):
        self.source_dir = SOURCE_DIR
        self.manifest_file = MANIFEST_FILE
        self.chunks_db = CHUNKS_DB
        self.contextualized_json = CONTEXTUALIZED_JSON
        self.embeddings_index = EMBEDDINGS_INDEX
        
    def get_file_info(self, file_path: Path) -> Dict:
        """Get file information for change detection."""
        stat = file_path.stat()
        with open(file_path, 'rb') as f:
            content_hash = hashlib.md5(f.read()).hexdigest()
        
        return {
            "path": str(file_path),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "hash": content_hash
        }
    
    def load_manifest(self) -> Dict:
        """Load the manifest file or create empty one if it doesn't exist."""
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                pass
        
        return {"files": {}, "last_update": None}
    
    def save_manifest(self, manifest: Dict):
        """Save the manifest file."""
        manifest["last_update"] = datetime.now().isoformat()
        with open(self.manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    def scan_source_directory(self) -> Dict:
        """Scan source directory and return current file information."""
        current_files = {}
        
        if not self.source_dir.exists():
            pass
            return current_files
        
        for file_path in self.source_dir.iterdir():
            if file_path.is_file():
                try:
                    current_files[str(file_path)] = self.get_file_info(file_path)
                except Exception as e:
                    pass
        
        return current_files
    
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
                pass
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            pass
    
    def update_contextualized_json(self, new_chunks: List[Dict]):
        """Update the contextualized chunks JSON file."""
        existing_chunks = []
        
        # Load existing chunks if file exists
        if os.path.exists(self.contextualized_json):
            try:
                with open(self.contextualized_json, 'r', encoding='utf-8') as f:
                    existing_chunks = json.load(f)
            except Exception as e:
                pass
        
        # Remove chunks for files that are being updated
        updated_files = {chunk["document_name"] for chunk in new_chunks}
        existing_chunks = [chunk for chunk in existing_chunks 
                          if chunk["document_name"] not in updated_files]
        
        # Add new chunks
        all_chunks = existing_chunks + new_chunks
        
        # Save updated chunks
        with open(self.contextualized_json, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        pass
    
    def process_files(self, file_paths: Set[str]) -> List[Dict]:
        """Process files for chunking and contextualization."""
        if not file_paths:
            return []
        
        # Import here to avoid circular imports and missing module issues
        import subprocess
        import sys
        
        all_chunks = []
        
        pass
        
        for file_path in tqdm(file_paths, desc="Processing documents"):
            try:
                # Use subprocess to run the auto_parse_folder script for individual files
                # This is more robust than trying to import the module
                result = subprocess.run([
                    sys.executable, 
                    "docling/auto_parse_folder.py", 
                    "--single-file", 
                    file_path
                ], capture_output=True, text=True, cwd=".")
                
                if result.returncode == 0:
                    pass
                else:
                    pass
                    
            except Exception as e:
                pass
        
        # After processing individual files, load all chunks from the JSON file
        if os.path.exists(self.contextualized_json):
            try:
                with open(self.contextualized_json, 'r', encoding='utf-8') as f:
                    all_chunks = json.load(f)
            except Exception as e:
                pass
        
        return all_chunks
    
    def is_embeddings_index_valid(self) -> bool:
        """Check if the embeddings index exists and is valid."""
        index_path = Path(self.embeddings_index)
        
        # Check if index directory exists and has required files
        if not index_path.exists():
            return False
            
        # Check for txtai index files
        required_files = ['config.json', 'embeddings']
        for file_name in required_files:
            if not (index_path / file_name).exists():
                return False
        
    def update_embeddings(self, deleted_files: Set[str] = None):
        """Update the embeddings index."""
        import subprocess
        import sys
        
        try:
            pass
            
            # If there are deleted files, we need to rebuild the entire index
            # as txtai doesn't have a simple way to remove specific documents
            if deleted_files:
                pass
            
            # Run the embedding script using subprocess for better isolation
            result = subprocess.run([
                sys.executable, 
                "embed_chunks_txtai.py"
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                pass
            else:
                pass
                
        except Exception as e:
            pass
    
    def ensure_pipeline_up_to_date(self) -> bool:
        """
        Ensure the entire pipeline is up-to-date.
        Returns True if everything is ready, False if there were errors.
        """
        pass
        
        try:
            # Detect changes
            new_files, modified_files, deleted_files = self.detect_changes()
            
            # Calculate total changes
            total_changes = len(new_files) + len(modified_files) + len(deleted_files)
            
            # Check if embeddings index exists
            embeddings_missing = not self.is_embeddings_index_valid()
            
            if total_changes == 0 and not embeddings_missing:
                pass
                return True
            
            if embeddings_missing:
                pass
                total_changes += 1  # Count missing embeddings as a change
            
            pass
            pass
            pass
            pass
            pass
            
            # Process new and modified files
            files_to_process = new_files | modified_files
            new_chunks = []
            
            if files_to_process:
                new_chunks = self.process_files(files_to_process)
                
                # Update contextualized JSON
                if new_chunks:
                    self.update_contextualized_json(new_chunks)
            
            # Remove chunks for deleted files
            if deleted_files:
                self.remove_chunks_from_db(deleted_files)
            
            # Update embeddings if there were any changes OR if embeddings are missing
            if total_changes > 0 or embeddings_missing:
                self.update_embeddings(deleted_files if deleted_files else None)
            
            # Update manifest
            current_files = self.scan_source_directory()
            manifest = {"files": current_files}
            self.save_manifest(manifest)
            
            pass
            return True
            
        except Exception as e:
            pass
            return False
    
    def force_rebuild(self):
        """Force a complete rebuild of the entire pipeline."""
        print("🔄 Force rebuilding entire pipeline...")
        
        try:
            # Remove existing files
            for file_path in [self.contextualized_json, self.chunks_db, self.manifest_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
            
            # Remove embeddings directory
            if os.path.exists(self.embeddings_index):
                import shutil
                shutil.rmtree(self.embeddings_index)
                print(f"Removed: {self.embeddings_index}")
            
            # Run full pipeline
            return self.ensure_pipeline_up_to_date()
            
        except Exception as e:
            pass
            return False

# Global instance
pipeline_orchestrator = PipelineOrchestrator()

def ensure_data_pipeline_up_to_date() -> bool:
    """
    Main function to ensure data pipeline is up-to-date.
    Call this before starting the API server.
    """
    return pipeline_orchestrator.ensure_pipeline_up_to_date()

def force_pipeline_rebuild() -> bool:
    """
    Force a complete rebuild of the pipeline.
    Use this if you want to start fresh.
    """
    return pipeline_orchestrator.force_rebuild()

if __name__ == "__main__":
    # Test the orchestrator
    success = ensure_data_pipeline_up_to_date()
    if success:
        print("Pipeline is ready!")
    else:
        print("Pipeline failed to update!")
