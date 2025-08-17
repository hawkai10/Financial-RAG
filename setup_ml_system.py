"""
ML-based Setup System
Sets up the complete keyword-free, ML-based RAG system with Dgraph + Qdrant
"""

import os
import sys
import subprocess
import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import requests
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLSystemSetup:
    """Sets up the complete ML-based RAG system."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.setup_status = {
            'dependencies': False,
            'spacy_model': False,
            'dgraph': False,
            'qdrant': False,
            'databases': False,
            'configuration': False
        }
    
    def run_complete_setup(self):
        """Run the complete setup process."""
        logger.info("Starting ML-based RAG system setup...")
        
        try:
            # Step 1: Install Python dependencies
            self.install_dependencies()
            
            # Step 2: Download spaCy model
            self.setup_spacy_model()
            
            # Step 3: Check and setup Dgraph
            self.setup_dgraph()
            
            # Step 4: Check and setup Qdrant
            self.setup_qdrant()
            
            # Step 5: Initialize databases
            self.initialize_databases()
            
            # Step 6: Create configuration files
            self.setup_configuration()
            
            # Step 7: Run system tests
            self.run_system_tests()
            
            logger.info("✅ ML-based RAG system setup completed successfully!")
            self.print_setup_summary()
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            sys.exit(1)
    
    def install_dependencies(self):
        """Install Python dependencies."""
        logger.info("Installing Python dependencies...")
        
        try:
            # Install from the ML requirements file
            requirements_file = self.project_root / "requirements_ml.txt"
            
            if not requirements_file.exists():
                logger.warning("requirements_ml.txt not found, using basic requirements")
                requirements_file = self.project_root / "requirements.txt"
            
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            
            self.setup_status['dependencies'] = True
            logger.info("✅ Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            raise
    
    def setup_spacy_model(self):
        """Download and setup spaCy language model."""
        logger.info("Setting up spaCy language model...")
        
        try:
            # Download the English language model
            subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ], check=True)
            
            self.setup_status['spacy_model'] = True
            logger.info("✅ spaCy model installed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install spaCy model: {e}")
            # Try alternative installation method
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
                ], check=True)
                self.setup_status['spacy_model'] = True
                logger.info("✅ spaCy model installed via direct download")
            except Exception as e2:
                logger.error(f"❌ Both spaCy installation methods failed: {e2}")
                raise
    
    def install_qdrant(self):
        """Install and start Qdrant (polling, no input)."""
        logger.info("Installing Qdrant...")
        qdrant_instructions = """
        Qdrant needs to be installed separately. Please follow these steps:
        1. Docker installation (Recommended):
           docker run -p 6333:6333 qdrant/qdrant
        2. Or download binary from: https://qdrant.tech/documentation/quick-start/
        3. After installation, verify it's running at http://localhost:6333
        """
        logger.info(qdrant_instructions)
        # Poll for Qdrant health with timeout
        max_wait = 120  # seconds
        poll_interval = 3
        waited = 0
        logger.info("Waiting for Qdrant to become available at http://localhost:6333/collections ...")
        while waited < max_wait:
            try:
                response = requests.get("http://localhost:6333/collections", timeout=5)
                if response.status_code == 200:
                    self.setup_status['qdrant'] = True
                    logger.info("✅ Qdrant is now running")
                    break
            except Exception:
                pass
            time.sleep(poll_interval)
            waited += poll_interval
        else:
            logger.error(f"❌ Qdrant did not become available after {max_wait} seconds. Please check your Qdrant setup and try again.")
            raise RuntimeError("Qdrant startup timed out.")
    
    def setup_qdrant(self):
        """Setup Qdrant vector database."""
        logger.info("Checking Qdrant setup...")
        
        try:
            # Check if Qdrant is running
            response = requests.get("http://localhost:6333/collections", timeout=5)
            
            if response.status_code == 200:
                logger.info("✅ Qdrant is running")
                self.setup_status['qdrant'] = True
            else:
                self.install_qdrant()
                
        except requests.RequestException:
            self.install_qdrant()
    
    def install_qdrant(self):
        """Install and start Qdrant."""
        logger.info("Installing Qdrant...")
        
        # Provide installation instructions
        qdrant_instructions = """
        Qdrant needs to be installed separately. Please follow these steps:
        
        1. Docker installation (Recommended):
           docker run -p 6333:6333 qdrant/qdrant
        
        2. Or download binary from: https://qdrant.tech/documentation/quick-start/
        
        3. After installation, verify it's running at http://localhost:6333
        """
        
        logger.info(qdrant_instructions)
        
        # Wait for user to setup Qdrant
        input("Press Enter after Qdrant is running at http://localhost:6333...")
        
        # Verify again
        try:
            response = requests.get("http://localhost:6333/collections", timeout=5)
            if response.status_code == 200:
                self.setup_status['qdrant'] = True
                logger.info("✅ Qdrant is now running")
            else:
                raise Exception("Qdrant collections endpoint failed")
        except Exception as e:
            logger.error(f"❌ Qdrant is not responding: {e}")
            raise
    
    def initialize_databases(self):
        """Initialize database schemas and collections."""
        logger.info("Initializing databases...")
        
        try:
            # Import our ML components
            from adaptive_dgraph_manager import AdaptiveDgraphManager
            from adaptive_qdrant_manager import AdaptiveQdrantManager
            from dynamic_schema_manager import DynamicSchemaManager
            
            # Initialize Dgraph schema
            dgraph_manager = AdaptiveDgraphManager()
            logger.info("✅ Dgraph schema initialized")
            
            # Initialize Qdrant collections
            qdrant_manager = AdaptiveQdrantManager()
            logger.info("✅ Qdrant collections initialized")
            
            self.setup_status['databases'] = True
            logger.info("✅ Databases initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def setup_configuration(self):
        """Create configuration files."""
        logger.info("Creating configuration files...")
        
        try:
            # Create .env file if it doesn't exist
            env_file = self.project_root / ".env"
            
            if not env_file.exists():
                env_content = """# ML-based RAG System Configuration
MARKER_API_KEY=your_marker_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Database URLs
DGRAPH_URL=http://localhost:8080
QDRANT_URL=http://localhost:6333

# System Settings
LOG_LEVEL=INFO
MAX_CHUNK_SIZE=500
MIN_CHUNK_SIZE=100
OVERLAP_SIZE=50

# ML Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
SPACY_MODEL=en_core_web_sm

# Performance Settings
BATCH_SIZE=100
MAX_WORKERS=4
"""
                with open(env_file, 'w') as f:
                    f.write(env_content)
                
                logger.info("✅ .env file created")
            
            # Create ML-specific config
            ml_config_file = self.project_root / "ml_config.json"
            
            ml_config = {
                "content_analyzer": {
                    "entity_types": ["PERSON", "ORG", "GPE", "MONEY", "DATE", "PERCENT"],
                    "confidence_threshold": 0.7,
                    "max_entities_per_type": 10
                },
                "chunking": {
                    "min_chunk_size": 100,
                    "max_chunk_size": 500,
                    "overlap_size": 50,
                    "semantic_similarity_threshold": 0.8
                },
                "retrieval": {
                    "default_top_k": 10,
                    "max_top_k": 50,
                    "strategy_multipliers": {
                        "Standard": 1.0,
                        "Analyse": 1.5,
                        "Aggregation": 2.0
                    }
                },
                "databases": {
                    "dgraph_url": "http://localhost:8080",
                    "qdrant_url": "http://localhost:6333",
                    "collection_name": "adaptive_chunks"
                }
            }
            
            with open(ml_config_file, 'w') as f:
                json.dump(ml_config, f, indent=2)
            
            self.setup_status['configuration'] = True
            logger.info("✅ Configuration files created")
            
        except Exception as e:
            logger.error(f"❌ Configuration setup failed: {e}")
            raise
    
    def run_system_tests(self):
        """Run basic system tests."""
        logger.info("Running system tests...")
        
        try:
            # Test 1: Content Analyzer
            from content_analyzer import DynamicContentAnalyzer
            analyzer = DynamicContentAnalyzer()
            
            test_text = "Apple Inc. reported revenue of $365.8 billion in 2021."
            insight = analyzer.analyze_content(test_text)
            
            if insight.entities:
                logger.info("✅ Content Analyzer test passed")
            else:
                raise Exception("Content Analyzer test failed")
            
            # Test 2: Enhanced Chunker
            from enhanced_json_chunker import EnhancedJSONChunker
            chunker = EnhancedJSONChunker()
            
            # Create test data
            test_data = [{
                'document_id': 'test_setup',
                'pages': [{
                    'blocks': [{
                        'type': 'text',
                        'content': test_text
                    }]
                }]
            }]
            
            test_file = self.project_root / "test_setup.json"
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
            
            chunks = chunker.process_extracted_json(str(test_file))
            
            if chunks:
                logger.info("✅ Enhanced Chunker test passed")
            else:
                raise Exception("Enhanced Chunker test failed")
            
            # Clean up test file
            test_file.unlink()
            
            # Test 3: Database Connections
            from adaptive_dgraph_manager import AdaptiveDgraphManager
            from adaptive_qdrant_manager import AdaptiveQdrantManager
            
            dgraph_manager = AdaptiveDgraphManager()
            qdrant_manager = AdaptiveQdrantManager()
            
            # Test Dgraph
            dgraph_stats = dgraph_manager.get_statistics()
            logger.info("✅ Dgraph connection test passed")
            
            # Test Qdrant
            qdrant_stats = qdrant_manager.get_collection_statistics()
            logger.info("✅ Qdrant connection test passed")
            
            logger.info("✅ All system tests passed")
            
        except Exception as e:
            logger.error(f"❌ System tests failed: {e}")
            raise
    
    def print_setup_summary(self):
        """Print setup summary."""
        
        summary = """
🎉 ML-based RAG System Setup Complete!

✅ Components Status:
"""
        
        for component, status in self.setup_status.items():
            emoji = "✅" if status else "❌"
            summary += f"   {emoji} {component.replace('_', ' ').title()}: {'Ready' if status else 'Failed'}\n"
        
        summary += """
🚀 Next Steps:
1. Update your .env file with actual API keys
2. Test the system with: python ml_rag_backend.py
3. Start processing documents with the new ML-based pipeline

📚 Key Files Created:
- content_analyzer.py: ML-based content analysis
- enhanced_json_chunker.py: Advanced chunking with ML insights
- adaptive_dgraph_manager.py: Graph database operations
- adaptive_qdrant_manager.py: Vector database operations
- ml_rag_backend.py: Main orchestrator
- ml_config.json: ML system configuration

🔧 System URLs:
- Dgraph Admin: http://localhost:8080
- Qdrant Dashboard: http://localhost:6333/dashboard

Happy analyzing! 🎯
"""
        
        print(summary)

def main():
    """Main setup function."""
    setup = MLSystemSetup()
    setup.run_complete_setup()

if __name__ == "__main__":
    main()
