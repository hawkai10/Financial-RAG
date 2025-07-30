#!/usr/bin/env python3
"""
DocuChat AI Enhanced Setup Script
Automates the complete setup process with Phase 2 optimizations
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any
import shutil

class DocuChatSetup:
    """Enhanced setup manager for DocuChat AI with Phase 2 optimizations."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.required_files = [
            'config.py', 'utils.py', 'feedback_database.py',
            'rag_backend.py', 'rag_app.py', 'progressive_retrieval.py',
            'aggregation_optimizer.py', 'auto_parse_folder.py',
            'embed_chunks_txtai.py', 'requirements.txt'
        ]
        self.optional_dirs = [
            'api-server', 'extraction_logs', 'business-docs-index'
        ]
    
    def run_command(self, command: List[str], description: str) -> bool:
        """Run a command with error handling."""
        print(f"🔄 {description}...")
        try:
            result = subprocess.run(
                command, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            print(f"✅ {description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ {description} failed: {e}")
            return False
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        print("🐍 Checking Python version...")
        version = sys.version_info
        
        if version.major == 3 and version.minor >= 8:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
            return True
        else:
            print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
            print("   Required: Python 3.8 or higher")
            return False
    
    def check_required_files(self) -> bool:
        """Check if all required files exist."""
        print("📁 Checking required files...")
        missing_files = []
        
        for file_name in self.required_files:
            file_path = self.project_root / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"❌ Missing required files: {', '.join(missing_files)}")
            return False
        else:
            print("✅ All required files present")
            return True
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        print("📂 Creating directories...")
        try:
            for dir_name in self.optional_dirs:
                dir_path = self.project_root / dir_name
                dir_path.mkdir(exist_ok=True)
                print(f"   Created: {dir_name}")
            
            print("✅ Directories created successfully")
            return True
        except Exception as e:
            print(f"❌ Directory creation failed: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        print("📦 Installing dependencies...")
        
        # Upgrade pip first
        if not self.run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                               "Upgrading pip"):
            return False
        
        # Install requirements
        if not self.run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                               "Installing requirements"):
            return False
        
        return True
    
    def check_environment_variables(self) -> bool:
        """Check and setup environment variables."""
        print("🔑 Checking environment variables...")
        
        env_file = self.project_root / ".env"
        
        if not env_file.exists():
            print("📝 Creating .env file...")
            with open(env_file, 'w') as f:
                f.write("# DocuChat AI Environment Variables\n")
                f.write("GEMINI_API_KEY=your_gemini_api_key_here\n")
                f.write("\n# Optional: Optimization settings\n")
                f.write("PROGRESSIVE_RETRIEVAL_ENABLED=true\n")
                f.write("SAMPLING_AGGREGATION_ENABLED=true\n")
                f.write("HYBRID_SEARCH_ENABLED=true\n")
            
            print("⚠️  Please edit .env file and add your Gemini API key")
            return False
        
        # Check if API key is set
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            print("⚠️  Please set your GEMINI_API_KEY in the .env file")
            return False
        
        print("✅ Environment variables configured")
        return True
    
    def setup_sample_data(self) -> bool:
        """Setup sample data if no documents exist."""
        print("📄 Checking for document data...")
        
        chunks_file = self.project_root / "contextualized_chunks.json"
        index_dir = self.project_root / "business-docs-index"
        
        if not chunks_file.exists():
            print("⚠️  No document chunks found.")
            print("   To process documents:")
            print("   1. Place PDF files in Source_Documents/ folder")
            print("   2. Update SOURCE_DIR path in auto_parse_folder.py")
            print("   3. Run: python auto_parse_folder.py")
            print("   4. Run: python embed_chunks_txtai.py")
            return False
        
        if not index_dir.exists() or not any(index_dir.iterdir()):
            print("⚠️  No embeddings index found.")
            print("   Run: python embed_chunks_txtai.py")
            return False
        
        print("✅ Document data ready")
        return True
    
    def run_tests(self) -> bool:
        """Run basic system tests."""
        print("🧪 Running system tests...")
        
        try:
            # Test imports
            print("   Testing imports...")
            import config
            import utils
            import rag_backend
            import progressive_retrieval
            import aggregation_optimizer
            from feedback_database import EnhancedFeedbackDatabase
            
            # Test configuration
            print("   Testing configuration...")
            if not hasattr(config.config, 'OPTIMAL_CHUNK_LIMITS'):
                print("❌ Configuration missing Phase 2 enhancements")
                return False
            
            # Test database
            print("   Testing database...")
            db = EnhancedFeedbackDatabase(":memory:")
                
            print("✅ System tests passed")
            return True
            
        except Exception as e:
            print(f"❌ System tests failed: {e}")
            return False
    
    def display_next_steps(self):
        """Display next steps for the user."""
        print("\n" + "="*60)
        print("🎉 DocuChat AI Enhanced Setup Complete!")
        print("="*60)
        
        print("\n📋 Next Steps:")
        print("1. Ensure your documents are processed:")
        print("   python auto_parse_folder.py")
        print("   python embed_chunks_txtai.py")
        
        print("\n2. Start the application:")
        print("   streamlit run rag_app.py")
        
        print("\n3. Or use Docker:")
        print("   docker-compose up -d")
        
        print("\n🚀 New Phase 2 Features:")
        print("• Cost optimization (40-65% reduction)")
        print("• Progressive retrieval")
        print("• Aggregation sampling")
        print("• Enhanced caching")
        print("• Performance dashboard")
        
        print("\n📊 Access Points:")
        print("• Streamlit UI: http://localhost:8501")
        print("• FastAPI docs: http://localhost:8000/docs")
        
        print("\n💡 Optimization Settings:")
        print("• Progressive retrieval: Enabled by default")
        print("• Aggregation sampling: Enabled for 8+ chunks")
        print("• Cache TTL: 1 hour")
        print("• Cost tracking: Automatic")

def main():
    """Main setup function."""
    print("🚀 DocuChat AI Enhanced Setup")
    print("Phase 2: Cost Optimization & Performance Enhancement")
    print("-" * 60)
    
    setup = DocuChatSetup()
    
    # Run setup steps
    steps = [
        ("Python Version", setup.check_python_version),
        ("Required Files", setup.check_required_files),
        ("Create Directories", setup.create_directories),
        ("Install Dependencies", setup.install_dependencies),
        ("Environment Variables", setup.check_environment_variables),
        ("Document Data", setup.setup_sample_data),
        ("System Tests", setup.run_tests)
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        if not step_function():
            failed_steps.append(step_name)
    
    if failed_steps:
        print(f"\n❌ Setup incomplete. Failed steps: {', '.join(failed_steps)}")
        print("Please resolve the issues above and run setup again.")
        sys.exit(1)
    
    setup.display_next_steps()

if __name__ == "__main__":
    main()
