#!/usr/bin/env python3
"""
Check if environment is ready for building native applications
"""

import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    
    print("="*70)
    print("Checking Build Requirements")
    print("="*70)
    print()
    
    all_good = True
    
    # Check Python version
    print("1. Python Version:")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        all_good = False
    print()
    
    # Check required files
    print("2. Required Files:")
    required_files = [
        'app.py',
        'app_launcher.py',
        'data_loader.py',
        'risk_predictor.py',
        'config.py',
        'models/risk_predictor.pkl',
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} (MISSING)")
            all_good = False
    print()
    
    # Check required packages
    print("3. Required Packages:")
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'pickle',
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} (NOT INSTALLED)")
            all_good = False
    print()
    
    # Check PyInstaller
    print("4. Build Tool (PyInstaller):")
    try:
        import PyInstaller
        print(f"   ✓ PyInstaller installed")
    except ImportError:
        print(f"   ⚠ PyInstaller not installed (will be installed during build)")
    print()
    
    # Check data directory
    print("5. Data Directory:")
    if Path('data').exists():
        data_files = list(Path('data').glob('*.csv'))
        print(f"   ✓ data/ directory exists ({len(data_files)} CSV files)")
    else:
        print(f"   ✗ data/ directory not found")
        all_good = False
    print()
    
    # Check OS
    print("6. Operating System:")
    if sys.platform == 'darwin':
        print(f"   ✓ macOS detected - use build_mac.sh")
    elif sys.platform == 'win32':
        print(f"   ✓ Windows detected - use build_windows.bat")
    elif sys.platform == 'linux':
        print(f"   ⚠ Linux detected - native apps work but not optimized")
    else:
        print(f"   ? Unknown OS: {sys.platform}")
    print()
    
    # Final status
    print("="*70)
    if all_good:
        print("✓ ALL CHECKS PASSED - Ready to build!")
        print("="*70)
        print()
        if sys.platform == 'darwin':
            print("Run: ./build_mac.sh")
        elif sys.platform == 'win32':
            print("Run: build_windows.bat")
        print()
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Fix issues above before building")
        print("="*70)
        print()
        print("Common fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Train model: python main_analysis.py")
        print("  - Check data files are in data/ directory")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(check_requirements())

