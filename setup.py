#!/usr/bin/env python3
"""Setup script for the molecular graph prediction project."""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def install_pytorch_geometric():
    """Install PyTorch Geometric."""
    print("Installing PyTorch Geometric...")
    try:
        # Install PyTorch Geometric and its dependencies
        packages = [
            "torch-geometric",
            "torch-scatter",
            "torch-sparse", 
            "torch-cluster"
        ]
        
        for package in packages:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        
        print("✓ PyTorch Geometric installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing PyTorch Geometric: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    directories = [
        "data/raw",
        "data/processed", 
        "checkpoints",
        "logs",
        "assets"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories created successfully")
    return True

def test_installation():
    """Test the installation."""
    print("Testing installation...")
    try:
        result = subprocess.run([sys.executable, "test_installation.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Installation test passed")
            return True
        else:
            print(f"✗ Installation test failed: {result.stdout}")
            return False
    except Exception as e:
        print(f"✗ Error running installation test: {e}")
        return False

def main():
    """Main setup function."""
    print("Setting up Molecular Graph Prediction Project")
    print("=" * 50)
    
    steps = [
        ("Installing requirements", install_requirements),
        ("Installing PyTorch Geometric", install_pytorch_geometric),
        ("Creating directories", create_directories),
        ("Testing installation", test_installation)
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if step_func():
            success_count += 1
        else:
            print(f"Failed at step: {step_name}")
            break
    
    print("\n" + "=" * 50)
    if success_count == len(steps):
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the demo: python run_demo.py")
        print("2. Train a model: python train.py --mode single")
        print("3. Run tests: python test_installation.py")
        return 0
    else:
        print("✗ Setup failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
