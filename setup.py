#!/usr/bin/env python3
"""
Opinion Market - Setup Script
Automated setup for development environment
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def create_env_file():
    """Create .env file from template"""
    if not os.path.exists('.env'):
        if os.path.exists('env.example'):
            shutil.copy('env.example', '.env')
            print("✅ Created .env file from template")
            print("⚠️  Please update .env with your actual configuration")
        else:
            print("⚠️  No env.example found, please create .env manually")

def setup_database():
    """Setup database and run migrations"""
    print("🔄 Setting up database...")
    # This would typically involve creating the database
    # For now, we'll just create the tables
    try:
        from app.core.database import engine
        from app.models import user, market, trade, vote
        from app.core.database import Base
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created")
    except Exception as e:
        print(f"⚠️  Database setup failed: {e}")
        print("   You may need to set up PostgreSQL and update DATABASE_URL in .env")

def main():
    """Main setup function"""
    print("🚀 Setting up Opinion Market...")
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists('venv'):
        print("🔄 Creating virtual environment...")
        run_command("python -m venv venv", "Creating virtual environment")
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Install dependencies
    run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies")
    
    # Create .env file
    create_env_file()
    
    # Setup database
    setup_database()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Update .env with your configuration")
    print("2. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("3. Run the application:")
    print("   python run.py")
    print("\nAPI documentation will be available at:")
    print("   http://localhost:8000/docs")

if __name__ == "__main__":
    main()
