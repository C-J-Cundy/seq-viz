#!/usr/bin/env python3
"""
Run the token prediction dashboard with the Llama Shakespeare trainer.

Usage:
    python run_token_dashboard.py
    
Then open http://localhost:8000/token_dashboard.html in your browser.
"""

import asyncio
import subprocess
import webbrowser
import time
import os
import signal
import sys

def signal_handler(sig, frame):
    print('\n🛑 Shutting down servers...')
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting Token Prediction Dashboard")
    print("=" * 50)
    
    # Start the training server
    print("📚 Starting Llama Shakespeare training server...")
    train_process = subprocess.Popen(
        [sys.executable, "train_llama_shakespeare.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give the training server time to start
    time.sleep(3)
    
    # Start a simple HTTP server for the dashboard
    print("🌐 Starting web server...")
    web_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give the web server time to start
    time.sleep(2)
    
    # Open the dashboard in the browser
    dashboard_url = "http://localhost:8000/token_dashboard.html"
    print(f"\n✨ Opening dashboard at {dashboard_url}")
    webbrowser.open(dashboard_url)
    
    print("\n" + "=" * 50)
    print("📊 Dashboard is running!")
    print("🔑 Press Ctrl+C to stop all servers")
    print("=" * 50)
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        train_process.terminate()
        web_process.terminate()
        train_process.wait()
        web_process.wait()
        print("✅ All servers stopped")

if __name__ == "__main__":
    main()