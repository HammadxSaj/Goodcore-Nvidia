"""
Script to run the FastAPI server
"""

import uvicorn
import sys
import os

if __name__ == "__main__":
    # Add src to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )