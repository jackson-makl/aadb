#!/usr/bin/env python3
"""
AADB Risk Prediction App Launcher
SIMPLE VERSION - No fancy features, just works
"""

import sys
import os
import webbrowser
import time
import socket
from pathlib import Path

def is_port_in_use(port):
    """Check if port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except:
            return True

def main():
    """Main function"""
    # Get app directory
    if getattr(sys, 'frozen', False):
        app_dir = Path(sys._MEIPASS)
    else:
        app_dir = Path(__file__).parent.absolute()
    
    os.chdir(app_dir)
    
    port = 8501
    url = f"http://localhost:{port}"
    
    # If port is already in use, just exit silently (app already running)
    if is_port_in_use(port):
        sys.exit(0)
    
    # Open browser after delay
    def open_browser():
        time.sleep(3)
        webbrowser.open(url)
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Configure and run Streamlit
    sys.argv = [
        "streamlit", "run",
        str(app_dir / "app.py"),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--global.developmentMode", "false",
        "--server.runOnSave", "false",
        "--server.fileWatcherType", "none"
    ]
    
    # Run Streamlit directly - blocks until quit
    from streamlit.web import cli as stcli
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
