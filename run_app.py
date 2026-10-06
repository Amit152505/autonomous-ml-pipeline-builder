"""
run_app.py
Dev server launcher for local Streamlit execution.
"""
import sys
import subprocess

if __name__ == "__main__":
    # Launch Streamlit bound to port 3000
    cmd = [
        "streamlit",
        "run",
        "app.py",
        "--server.port=3000",
        "--server.address=0.0.0.0",
        "--server.headless=true",
    ]
    subprocess.run(cmd)
