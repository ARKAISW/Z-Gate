"""render_entrypoint.py — Dual-process entrypoint for Render.com Free Web Service.

Spawns the 24/7 live paper trading daemon in the background and serves
the Streamlit quantitative dashboard on the Render-assigned $PORT.
"""
import os
import signal
import subprocess
import sys
import time

def main():
    print("[Render Entrypoint] Starting Z-Gate 24/7 Live Paper Trading Daemon...")
    
    # 1. Start the live trading runner as a background sub-process
    trader_proc = subprocess.Popen(
        [sys.executable, "scripts/run_live_paper.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    
    # 2. Get the port assigned by Render (default 10000 or 8501)
    port = os.environ.get("PORT", "10000")
    print(f"[Render Entrypoint] Starting Streamlit Dashboard on port {port}...")

    # Handle graceful termination on Render shutdown
    def handle_sigterm(signum, frame):
        print("[Render Entrypoint] Received termination signal, shutting down trader...")
        trader_proc.terminate()
        try:
            trader_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            trader_proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    # 3. Launch Streamlit (blocking process that keeps the web port active)
    try:
        streamlit_cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "scripts/dashboard.py",
            f"--server.port={port}",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
        ]
        subprocess.run(streamlit_cmd, check=True)
    except KeyboardInterrupt:
        handle_sigterm(None, None)
    finally:
        if trader_proc.poll() is None:
            trader_proc.terminate()

if __name__ == "__main__":
    main()
