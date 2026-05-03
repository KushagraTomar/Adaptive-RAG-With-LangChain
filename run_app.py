#!/usr/bin/env python3
"""
Application launcher for Adaptive RAG.
Starts the FastAPI server on http://localhost:8000
"""

import os
import sys
import subprocess
from pathlib import Path


def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = [
        "MISTRAL_API_KEY",
        "TAVILY_API_KEY",
        "COHERE_API_KEY",
        "PINECONE_API_KEY",
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("⚠️  Warning: The following environment variables are not set:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set them in your .env file or as environment variables.")
        return False
    
    return True


def main():
    """Start the Adaptive RAG API server"""
    # Check environment
    if not check_environment_variables():
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting.")
            sys.exit(1)
    
    print("\n Starting Adaptive RAG API Server...\n")
    print(" API available at: http://localhost:8000")
    print(" Documentation at: http://localhost:8000/docs")
    print(" Interactive API at: http://localhost:8000/openapi.json")
    print("\nPress Ctrl+C to stop the server.\n")
    
    project_dir = Path(__file__).parent.absolute()
    
    try:
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "app.api.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ],
            cwd=project_dir
        )
    except KeyboardInterrupt:
        print("\n\n✋ Server stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()