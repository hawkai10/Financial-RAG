#!/usr/bin/env python3
"""
Test script to compare marked.py vs auto_parse_folder.py chunking within the normal RAG pipeline.
Both systems will chunk the same document, then we'll use the normal RAG backend for retrieval and answering.
"""

import os
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import logging
import gc

# Import your existing RAG components
from rag_backend import execute_standard_rag
from chunk_manager import ChunkManager
from init_chunks_db import create_chunks_database
from txtai import Embeddings
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkingComparisonRAG:
    """Compare chunking methods using the normal RAG pipeline"""
    
    def __init__(self):
        self.test_file_path = "C:\\Users\\arvin\\OneDrive\\Desktop\\trial\\Final FInancial RAG\\Source_Documents\\cn 19-20.pdf"
        self.test_questions = [
            "What is the GSTIN of Krishna Prabhash Agro Oil?",
            "What is the total amount chargeable by Bhartiya Enterprise to Krishna Prabhash Agro Oil Pvt?",
            "What is the GST in the invoice to Krishna Prabhash Agro Oil Pvt?",
            "What is the GST Rate?"
        ]
        
        # Backup original files
        self.backup_chunks_json = "contextualized_chunks_backup.json"
        self.backup_chunks_db = "chunks_backup.db"
        
        self.results = {}
    
    def safe_remove_file(self, filepath: str, max_attempts: int = 5):
        """Safely remove a file with retry logic"""
        for attempt in range(max_attempts):
            try:
                if os.path.exists(filepath):
                    gc.collect()  # Force garbage collection
                    time.sleep(0.5)  # Wait a bit
                    os.remove(filepath)
                    logger.info(f"✅ Removed {filepath}")
                return True
            except PermissionError:
                if attempt < max_attempts - 1:
                    logger.warning(f"⚠️ File {filepath} in use, retrying in {attempt + 1}s...")
                    time.sleep(attempt + 1)
                else:
                    logger.error(f"❌ Could not remove {filepath} after {max_attempts} attempts")
                    return False
        return True
    
    def backup_existing_data(self):
        """Backup existing chunks data"""
        logger.info("Backing up existing chunks data...")
        
        if os.path.exists("contextualized_chunks.json"):
            shutil.copy2("contextualized_chunks.json", self.backup_chunks_json)
            logger.info("✅ Backed up contextualized_chunks.json")
        
        if os.path.exists("chunks.db"):
            shutil.copy2("chunks.db", self.backup_chunks_db)
            logger.info("✅ Backed up chunks.db")
    
    def restore_original_data(self):
        """Restore original chunks data"""
        logger.info("Restoring original chunks data...")
        
        if os.path.exists(self.backup_chunks_json):
            shutil.copy2(self.backup_chunks_json, "contextualized_chunks.json")
            os.remove(self.backup_chunks_json)
            logger.info("✅ Restored contextualized_chunks.json")
        
        if os.path.exists(self.backup_chunks_db):
            shutil.copy2(self.backup_chunks_db, "chunks.db")
            os.remove(self.backup_chunks_db)
            logger.info("✅ Restored chunks.db")
    
    def run_marked_chunking(self) -> Dict[str, Any]:
        """Run marked.py chunking and setup database"""
        logger.info("=== TESTING MARKED.PY CHUNKING ===")
        start_time = time.time()
        
        try:
            # Clear existing chunks safely
            self.safe_remove_file("contextualized_chunks.json")
            self.safe_remove_file("chunks.db")
            
            # Wait a moment for processes to release files
            time.sleep(2)
            
            # Run marked.py on single file
            logger.info(f"Running marked.py on: {self.test_file_path}")
            result = subprocess.run(
                ["C:/Users/arvin/OneDrive/Desktop/trial/Final FInancial RAG/venv/Scripts/python.exe", "marked.py", "--single-file", self.test_file_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"❌ Marked.py failed: {result.stderr}")
                return {"success": False, "error": result.stderr, "processing_time": time.time() - start_time}
            
            # Check if chunks were generated
            if not os.path.exists("contextualized_chunks.json"):
                logger.error("❌ Marked.py did not generate contextualized_chunks.json")
                return {"success": False, "error": "No chunks file generated", "processing_time": time.time() - start_time}
            
            # Load and analyze chunks
            with open("contextualized_chunks.json", 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Initialize chunks database using existing function
            logger.info("Initializing chunks database...")
            create_chunks_database()
            
            processing_time = time.time() - start_time
            
            # Analyze chunks
            analysis = self.analyze_chunks(chunks, "marked")
            
            logger.info(f"✅ Marked.py completed successfully")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            logger.info(f"   Total chunks: {len(chunks)}")
            
            return {
                "success": True,
                "processing_time": processing_time,
                "total_chunks": len(chunks),
                "chunks": chunks,
                "analysis": analysis,
                "stdout": result.stdout
            }
            
        except Exception as e:
            logger.error(f"❌ Error in marked.py processing: {str(e)}")
            return {"success": False, "error": str(e), "processing_time": time.time() - start_time}
    
    def run_auto_parse_chunking(self) -> Dict[str, Any]:
        """Run auto_parse_folder.py chunking and setup database"""
        logger.info("=== TESTING AUTO_PARSE_FOLDER.PY CHUNKING ===")
        start_time = time.time()
        
        try:
            # Clear existing chunks safely
            self.safe_remove_file("contextualized_chunks.json")
            self.safe_remove_file("chunks.db")
            
            # Wait a moment for processes to release files
            time.sleep(2)
            
            # Copy the test file to a temporary location in Source_Documents for auto_parse_folder.py
            temp_source_dir = Path("temp_source_for_autoparse")
            if temp_source_dir.exists():
                shutil.rmtree(temp_source_dir)
            temp_source_dir.mkdir()
            
            # Copy only our test file
            shutil.copy2(self.test_file_path, temp_source_dir / "cn 19-20.pdf")
            
            # Temporarily move original Source_Documents and replace with our temp
            original_source = Path("Source_Documents")
            source_backup = Path("Source_Documents_backup")
            
            if original_source.exists():
                shutil.move(str(original_source), str(source_backup))
            shutil.move(str(temp_source_dir), str(original_source))
            
            # Run auto_parse_folder.py
            logger.info(f"Running auto_parse_folder.py...")
            result = subprocess.run(
                ["C:/Users/arvin/OneDrive/Desktop/trial/Final FInancial RAG/venv/Scripts/python.exe", "docling/auto_parse_folder.py"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Restore original Source_Documents
            shutil.rmtree(original_source)
            if source_backup.exists():
                shutil.move(str(source_backup), str(original_source))
            
            if result.returncode != 0:
                logger.error(f"❌ Auto_parse_folder.py failed: {result.stderr}")
                return {"success": False, "error": result.stderr, "processing_time": time.time() - start_time}
            
            # Check if chunks were generated
            if not os.path.exists("contextualized_chunks.json"):
                logger.error("❌ Auto_parse_folder.py did not generate contextualized_chunks.json")
                return {"success": False, "error": "No chunks file generated", "processing_time": time.time() - start_time}
            
            # Load and analyze chunks
            with open("contextualized_chunks.json", 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Initialize chunks database using existing function
            logger.info("Initializing chunks database...")
            create_chunks_database()
            
            processing_time = time.time() - start_time
            
            # Analyze chunks
            analysis = self.analyze_chunks(chunks, "auto_parse")
            
            logger.info(f"✅ Auto_parse_folder.py completed successfully")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            logger.info(f"   Total chunks: {len(chunks)}")
            
            return {
                "success": True,
                "processing_time": processing_time,
                "total_chunks": len(chunks),
                "chunks": chunks,
                "analysis": analysis,
                "stdout": result.stdout
            }
            
        except Exception as e:
            logger.error(f"❌ Error in auto_parse_folder.py processing: {str(e)}")
            return {"success": False, "error": str(e), "processing_time": time.time() - start_time}
    
    def analyze_chunks(self, chunks: List[Dict], system_name: str) -> Dict[str, Any]:
        """Analyze chunk characteristics"""
        if not chunks:
            return {"error": "No chunks to analyze"}
        
        # Basic statistics
        total_chunks = len(chunks)
        table_chunks = [c for c in chunks if c.get('is_table', False)]
        text_chunks = [c for c in chunks if not c.get('is_table', False)]
        
        # Token analysis for text chunks
        token_counts = [c.get('num_tokens', 0) for c in text_chunks if c.get('num_tokens')]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        
        # Content analysis for financial keywords
        financial_keywords = ['GSTIN', 'GST', 'Krishna', 'Prabhash', 'Bhartiya', 'invoice', 'amount', 'total']
        keyword_coverage = {}
        
        for keyword in financial_keywords:
            keyword_coverage[keyword] = sum(
                1 for chunk in chunks 
                if keyword.lower() in chunk.get('chunk_text', '').lower()
            )
        
        return {
            "total_chunks": total_chunks,
            "table_chunks": len(table_chunks),
            "text_chunks": len(text_chunks),
            "avg_tokens_per_chunk": round(avg_tokens, 2),
            "keyword_coverage": keyword_coverage
        }
    
    def test_rag_pipeline(self, system_name: str) -> Dict[str, Any]:
        """Test the RAG pipeline with current chunks"""
        logger.info(f"=== TESTING RAG PIPELINE FOR {system_name.upper()} ===")
        
        try:
            # Initialize embeddings (required for RAG functions)
            embeddings = Embeddings()
            embeddings.load("business-docs-index")
            
            answers = {}
            for question in self.test_questions:
                logger.info(f"🔍 Processing question: {question}")
                
                # Use the async RAG function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Get answer using the standard RAG function with all required parameters
                    result = loop.run_until_complete(
                        execute_standard_rag(
                            corrected_query=question,
                            intent="Standard",
                            confidence=0.8,
                            alternative_queries=[question],  # Required parameter
                            embeddings=embeddings,
                            topn=5,  # Required parameter
                            filters=None,  # Required parameter
                            enable_reranking=True,  # Required parameter
                            session_id=f"test_session_{system_name}",  # Required parameter
                            enable_optimization=True,  # Required parameter
                            start_time=time.time(),
                            processed={"classification": {"intent": "Standard", "confidence": 0.8}}  # Required parameter
                        )
                    )
                    
                    answer = result.get('answer', 'No answer generated')
                    
                    answers[question] = {
                        "answer": answer,
                        "question": question,
                        "metadata": result
                    }
                    
                    logger.info(f"✅ Got answer for: {question[:50]}...")
                    
                finally:
                    loop.close()
            
            return {
                "success": True,
                "answers": answers
            }
            
        except Exception as e:
            logger.error(f"❌ Error in RAG pipeline testing: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def save_results(self):
        """Save comparison results"""
        results_file = f"chunking_comparison_results_{int(time.time())}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Results saved to: {results_file}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print comparison summary"""
        print("\n" + "=" * 80)
        print("CHUNKING COMPARISON SUMMARY - USING NORMAL RAG PIPELINE")
        print("=" * 80)
        
        for system in ['marked', 'auto_parse']:
            if system in self.results:
                result = self.results[system]
                
                print(f"\n🔧 {system.upper()} CHUNKING:")
                print("-" * 40)
                
                if result['chunking']['success']:
                    chunking = result['chunking']
                    print(f"  ✅ Processing Time: {chunking['processing_time']:.2f}s")
                    print(f"  📊 Total Chunks: {chunking['total_chunks']}")
                    print(f"  📋 Table Chunks: {chunking['analysis']['table_chunks']}")
                    print(f"  📝 Text Chunks: {chunking['analysis']['text_chunks']}")
                    print(f"  🏷️  Avg Tokens/Chunk: {chunking['analysis']['avg_tokens_per_chunk']}")
                    
                    # Show keyword coverage
                    print(f"  🔍 Keyword Coverage:")
                    for keyword, count in chunking['analysis']['keyword_coverage'].items():
                        print(f"     {keyword}: {count} chunks")
                else:
                    print(f"  ❌ Chunking failed: {result['chunking']['error']}")
                
                # RAG pipeline results
                if 'rag_pipeline' in result and result['rag_pipeline']['success']:
                    print(f"\n  🤖 RAG PIPELINE ANSWERS:")
                    answers = result['rag_pipeline']['answers']
                    for i, (question, data) in enumerate(answers.items(), 1):
                        print(f"     Q{i}: {question}")
                        answer_preview = data['answer'][:200].replace('\n', ' ')
                        print(f"     A{i}: {answer_preview}...")
                        print()
                else:
                    print(f"  ❌ RAG Pipeline failed")
        
        # Comparison
        if 'marked' in self.results and 'auto_parse' in self.results:
            marked_success = self.results['marked']['chunking']['success']
            auto_parse_success = self.results['auto_parse']['chunking']['success']
            
            if marked_success and auto_parse_success:
                marked_time = self.results['marked']['chunking']['processing_time']
                auto_parse_time = self.results['auto_parse']['chunking']['processing_time']
                marked_chunks = self.results['marked']['chunking']['total_chunks']
                auto_parse_chunks = self.results['auto_parse']['chunking']['total_chunks']
                
                print(f"\n🏆 PERFORMANCE COMPARISON:")
                print(f"   ⚡ Faster: {'Marked' if marked_time < auto_parse_time else 'Auto Parse'}")
                print(f"   📊 More Chunks: {'Marked' if marked_chunks > auto_parse_chunks else 'Auto Parse'}")
                print(f"   ⏱️  Time Difference: {abs(marked_time - auto_parse_time):.2f}s")
    
    def run_complete_comparison(self):
        """Run the complete comparison workflow"""
        logger.info("🚀 Starting chunking comparison using normal RAG pipeline...")
        
        # Backup existing data
        self.backup_existing_data()
        
        # Wait for any file handles to be released
        logger.info("⏳ Waiting for file handles to be released...")
        time.sleep(3)
        
        try:
            # Test marked.py
            logger.info("\n" + "="*60)
            logger.info("PHASE 1: Testing marked.py chunking")
            logger.info("="*60)
            
            marked_chunking = self.run_marked_chunking()
            marked_rag = {"success": False}
            
            if marked_chunking['success']:
                marked_rag = self.test_rag_pipeline('marked')
            
            self.results['marked'] = {
                'chunking': marked_chunking,
                'rag_pipeline': marked_rag
            }
            
            # Test auto_parse_folder.py
            logger.info("\n" + "="*60)
            logger.info("PHASE 2: Testing auto_parse_folder.py chunking")
            logger.info("="*60)
            
            auto_parse_chunking = self.run_auto_parse_chunking()
            auto_parse_rag = {"success": False}
            
            if auto_parse_chunking['success']:
                auto_parse_rag = self.test_rag_pipeline('auto_parse')
            
            self.results['auto_parse'] = {
                'chunking': auto_parse_chunking,
                'rag_pipeline': auto_parse_rag
            }
            
            # Save and display results
            self.save_results()
            
        finally:
            # Always restore original data
            self.restore_original_data()
            logger.info("🔄 Original chunks data restored")

def main():
    """Main function"""
    comparison = ChunkingComparisonRAG()
    comparison.run_complete_comparison()

if __name__ == "__main__":
    main()
