"""
Script to run the Streamlit web interface
"""

import streamlit.web.cli as stcli
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    sys.argv = [
        "streamlit", 
        "run", 
        "src/web/app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    sys.exit(stcli.main())