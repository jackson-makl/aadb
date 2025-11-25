#!/usr/bin/env python3
"""
AADB Risk Prediction App Launcher - Native Window Version
Bulletproof single-instance native macOS app
"""

import sys
import os
import subprocess
import socket
import time
import signal
from pathlib import Path

# CRITICAL: Lock file check FIRST before any other imports
LOCK_FILE = Path.home() / ".aadb_running.lock"

def check_single_instance():
    """Check if another instance is running - exit immediately if so"""
    if LOCK_FILE.exists():
        # Check if it's a stale lock (>30 seconds old)
        try:
            age = time.time() - LOCK_FILE.stat().st_mtime
            if age < 30:
                # Fresh lock, another instance is running
                sys.exit(0)
        except:
            pass
        # Remove stale lock
        try:
            LOCK_FILE.unlink()
        except:
            pass
    
    # Create our lock
    try:
        LOCK_FILE.write_text(str(os.getpid()))
    except:
        sys.exit(1)

# Check immediately
check_single_instance()

# Global variables
streamlit_process = None

def cleanup_and_exit(signum=None, frame=None):
    """Cleanup function - called on exit"""
    global streamlit_process
    
    # Kill streamlit
    if streamlit_process:
        try:
            streamlit_process.terminate()
            time.sleep(1)
            streamlit_process.kill()
        except:
            pass
    
    # Remove lock
    try:
        LOCK_FILE.unlink()
    except:
        pass
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, cleanup_and_exit)
signal.signal(signal.SIGINT, cleanup_and_exit)

def is_port_in_use(port):
    """Check if port is in use"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', port))
        sock.close()
        return False
    except:
        return True

def wait_for_server(port, timeout=30):
    """Wait for server to start"""
    for _ in range(timeout * 2):
        if is_port_in_use(port):
            return True
        time.sleep(0.5)
    return False

def log_error(msg):
    """Log error to file"""
    try:
        with open(Path.home() / "aadb_error.log", 'w') as f:
            f.write(msg + "\n")
            import traceback
            traceback.print_exc(file=f)
    except:
        pass

def main():
    """Main function"""
    global streamlit_process
    
    try:
        # Get app directory
        if getattr(sys, 'frozen', False):
            app_dir = Path(sys._MEIPASS)
        else:
            app_dir = Path(__file__).parent.absolute()
        
        os.chdir(app_dir)
        
        port = 8501
        app_path = app_dir / "app.py"
        
        # Check app.py exists
        if not app_path.exists():
            log_error(f"app.py not found at {app_path}")
            cleanup_and_exit()
        
        # Start Streamlit
        streamlit_process = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run",
                str(app_path),
                "--server.port", str(port),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
                "--global.developmentMode", "false",
                "--server.runOnSave", "false",
                "--server.fileWatcherType", "none",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )
        
        # Wait for server
        if not wait_for_server(port, 30):
            log_error("Streamlit failed to start")
            cleanup_and_exit()
        
        # Import webview (after server is ready)
        try:
            import webview
        except Exception as e:
            log_error(f"Failed to import webview: {e}")
            cleanup_and_exit()
        
        # Create window
        try:
            window = webview.create_window(
                'AADB Risk Prediction System',
                f'http://localhost:{port}',
                width=1400,
                height=900,
                resizable=True,
                min_size=(1000, 700)
            )
        except Exception as e:
            log_error(f"Failed to create window: {e}")
            cleanup_and_exit()
        
        # Start webview - this blocks until window closes
        try:
            webview.start()
        except Exception as e:
            log_error(f"Failed to start webview: {e}")
        
    except Exception as e:
        log_error(f"Fatal error: {e}")
    finally:
        cleanup_and_exit()

if __name__ == "__main__":
    main()
