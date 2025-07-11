"""
FastAPI main application for Speaker Preference AI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
from contextlib import asynccontextmanager

from .routes import router, initialize_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Speaker Preference AI API...")
    success = await initialize_system()
    if not success:
        logger.error("❌ Failed to initialize system")
        raise Exception("System initialization failed")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Speaker Preference AI API...")

# Create FastAPI app
app = FastAPI(
    title="Speaker Preference AI API",
    description="AI-powered speaker search and recommendation system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Speaker Preference AI API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)