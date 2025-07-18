"""
Entry point for Streamlit Cloud deployment
"""

import os
import sys

# Add src to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import and run the cloud app
from src.web.app_cloud import main

if __name__ == "__main__":
    main()
