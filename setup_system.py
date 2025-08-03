#!/usr/bin/env python3
"""
Setup script for Financial RAG System
Initializes the system and checks all dependencies
"""

import os
import sys
from pathlib import Path
import subprocess

def check_python_version():
    """Check if Python version is 3.10+"""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_environment_file():
    """Check if .env file exists"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("⚠️  .env file not found. Please copy .env.example to .env and fill in your API keys")
            return False
        else:
            print("❌ No .env.example file found")
            return False
    
    print("✅ .env file found")
    return True

def check_directories():
    """Check required directories exist"""
    required_dirs = [
        "Source_Documents",
        "amber-ai-search"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name} directory exists")
        else:
            print(f"❌ {dir_name} directory missing")
            all_exist = False
    
    return all_exist

def install_requirements():
    """Install Python requirements"""
    try:
        print("📦 Installing Python requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Python requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_node_frontend():
    """Setup React frontend"""
    frontend_dir = Path("amber-ai-search")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    try:
        print("📦 Installing Node.js dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True, capture_output=True)
        print("✅ Node.js dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Node.js dependencies: {e}")
        print("💡 Make sure Node.js is installed: https://nodejs.org/")
        return False
    except FileNotFoundError:
        print("❌ npm not found. Please install Node.js: https://nodejs.org/")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Financial RAG System...")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_environment_file),
        ("Required Directories", check_directories),
        ("Python Requirements", install_requirements),
        ("Node.js Frontend", setup_node_frontend)
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        if not check_func():
            failed_checks.append(check_name)
    
    print("\n" + "=" * 50)
    
    if failed_checks:
        print("❌ Setup incomplete. Failed checks:")
        for check in failed_checks:
            print(f"   - {check}")
        print("\n💡 Please fix the issues above and run this script again.")
        return False
    else:
        print("✅ Setup completed successfully!")
        print("\n🎉 Next steps:")
        print("   1. Copy .env.example to .env and add your Gemini API key")
        print("   2. Add your documents to the Source_Documents folder")
        print("   3. Run: python api_server.py (for backend)")
        print("   4. Run: cd amber-ai-search && npm run dev (for frontend)")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
