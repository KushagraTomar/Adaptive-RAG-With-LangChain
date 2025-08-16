#!/usr/bin/env python3
"""
Unified script to run the Adaptive RAG application.
This script starts the FastAPI server and opens the web interface in a browser.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_environment_variables():
    """Check if required environment variables are set."""
    missing_vars = []
    
    # Check for MistralAI API key (required for embeddings and LLM)
    if not os.environ.get("MISTRAL_API_KEY"):
        missing_vars.append("MISTRAL_API_KEY")
    
    # Check for Tavily API key (required for web search)
    if not os.environ.get("TAVILY_API_KEY"):
        missing_vars.append("TAVILY_API_KEY")

    if not os.environ.get("USER_AGENT"):
        missing_vars.append("USER_AGENT")
    
    if missing_vars:
        print("Warning: The following environment variables are not set:")
        for var in missing_vars:
            print(f"  - {var}")
        
        print("\nPlease set these environment variables before running the application.")
        print("You can set them in your terminal like this:")
        print("  export MISTRAL_API_KEY='your_mistral_api_key'")
        print("  export TAVILY_API_KEY='your_tavily_api_key'")
        return False
    
    return True

def main():
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Define paths
    backend_script = script_dir / "app" / "main.py"
    frontend_html = script_dir / "frontend" / "index.html"
    
    # Check if required files exist
    if not backend_script.exists():
        print(f"Error: Backend script not found at {backend_script}")
        sys.exit(1)
        
    if not frontend_html.exists():
        print(f"Error: Frontend HTML not found at {frontend_html}")
        sys.exit(1)
    
    # Check environment variables
    if not check_environment_variables():
        response = input("\nDo you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting application.")
            sys.exit(1)
    
    print("Starting Adaptive RAG application...")
    print("Starting backend server...")
    
    # Start the backend server as a subprocess
    try:
        # Use the same Python interpreter that's running this script
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], cwd=script_dir)
        
        print("Backend server started with PID:", backend_process.pid)
        print("Waiting for server to initialize...")
        
        # Wait a few seconds for the server to start
        time.sleep(3)
        
        # Open the frontend in the default browser
        frontend_url = frontend_html.as_uri()
        print(f"Opening frontend: {frontend_url}")
        webbrowser.open(frontend_url)
        
        print("\nApplication is now running!")
        print("Backend API: http://localhost:8000")
        print("Frontend: Opened in your default browser")
        print("\nPress Ctrl+C to stop the application")
        
        # Wait for the backend process
        backend_process.wait()
        
    except KeyboardInterrupt:
        print("\n\nShutting down application...")
        if 'backend_process' in locals():
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
        print("Application stopped.")
        
    except Exception as e:
        print(f"Error running application: {e}")
        if 'backend_process' in locals():
            backend_process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()