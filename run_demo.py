#!/usr/bin/env python3
"""Script to run the Streamlit demo."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit demo."""
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        return 1
    
    try:
        # Run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(demo_path)]
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running demo: {e}")
        return 1
    except FileNotFoundError:
        print("Streamlit not found. Please install it with: pip install streamlit")
        return 1

if __name__ == "__main__":
    sys.exit(main())
