#!/usr/bin/env python3
"""
AADB Risk Prediction App Launcher - Browser Version
Simple and bulletproof
"""

import sys
import os
import webbrowser
import time
import socket
from pathlib import Path

# Lock file - check IMMEDIATELY
LOCK_FILE = Path.home() / ".aadb_running.lock"

def check_already_running():
    """Check if app is already running"""
    if LOCK_FILE.exists():
        try:
            age = time.time() - LOCK_FILE.stat().st_mtime
            if age < 30:  # Less than 30 seconds old = still running
                sys.exit(0)  # Exit silently
            else:
                LOCK_FILE.unlink()  # Remove stale lock
        except:
            pass
    
    # Create lock
    try:
        LOCK_FILE.write_text(str(os.getpid()))
    except:
        sys.exit(1)

# Check immediately before importing anything else
check_already_running()

def cleanup():
    """Remove lock file on exit"""
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except:
        pass

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
    try:
        # Get app directory
        if getattr(sys, 'frozen', False):
            app_dir = Path(sys._MEIPASS)
        else:
            app_dir = Path(__file__).parent.absolute()
        
        os.chdir(app_dir)
        
        port = 8501
        url = f"http://localhost:{port}"
        
        # Configure Streamlit to NOT open browser automatically
        sys.argv = [
            "streamlit", "run",
            str(app_dir / "app.py"),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--global.developmentMode", "false",
            "--server.runOnSave", "false"
        ]
        
        # Open browser after short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open(url)
        
        import threading
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Start Streamlit (blocks)
        from streamlit.web import cli as stcli
        stcli.main()
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if getattr(sys, 'frozen', False):
            with open(Path.home() / "aadb_error.log", 'w') as f:
                f.write(f"ERROR: {e}\n")
                import traceback
                traceback.print_exc(file=f)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
