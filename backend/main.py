"""FastAPI main application."""

import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from backend.api.routes import router
from backend.api.simulation_routes import router as simulation_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ConversaVoice API",
    description="Backend API for ConversaVoice - AI-Powered Emotional Intelligence Assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# API Key Security Function
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    """Validate API key."""
    expected_api_key = os.getenv("API_SECRET_KEY")
    # If no key is set in environment, allow all (development mode)
    if not expected_api_key:
        return ""
    
    if api_key_header == expected_api_key:
        return api_key_header
        
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

from backend.limiter import limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", 
        "https://your-frontend.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, dependencies=[Depends(get_api_key)])
app.include_router(simulation_router, dependencies=[Depends(get_api_key)])


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Starting ConversaVoice API...")
    logger.info(f"API Documentation available at: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down ConversaVoice API...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "ConversaVoice API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "simulation": "/api/simulation/scenarios",
        "modes": ["assistant", "simulation"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
#
