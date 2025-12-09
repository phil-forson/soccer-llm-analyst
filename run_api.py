#!/usr/bin/env python3
"""
Simple script to run the Soccer LLM Analyst API server.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api import run_server

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Soccer LLM Analyst API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Soccer LLM Analyst API Server...")
    print(f"📍 Server will be available at: http://{args.host}:{args.port}")
    print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔍 ReDoc: http://{args.host}:{args.port}/redoc")
    print()
    
    run_server(host=args.host, port=args.port, reload=args.reload)


