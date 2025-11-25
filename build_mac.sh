#!/bin/bash
################################################################################
# Build script for macOS standalone application
# Creates AADB_Risk_Predictor.app for macOS
################################################################################

set -e  # Exit on error

echo "======================================================================"
echo "Building AADB Risk Predictor - macOS Application"
echo "======================================================================"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "✗ Error: This script must be run on macOS"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "env" ]; then
    echo "Activating virtual environment..."
    source env/bin/activate
fi

# Install PyInstaller if not already installed
echo "Installing build dependencies..."
pip install pyinstaller pillow -q

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist AADB_Risk_Predictor.app

# Create the macOS app bundle
echo ""
echo "Building macOS application..."
echo "This may take a few minutes..."
echo ""

# Check if icon exists, use it if available
ICON_FLAG=""
if [ -f "app_icon.icns" ]; then
    ICON_FLAG="--icon=app_icon.icns"
    echo "Using custom icon: app_icon.icns"
fi

pyinstaller --noconfirm --clean \
    --name "AADB_Risk_Predictor" \
    --windowed \
    $ICON_FLAG \
    --osx-bundle-identifier "org.aimahead.aadb" \
    --add-data "app.py:." \
    --add-data "data_loader.py:." \
    --add-data "risk_predictor.py:." \
    --add-data "config.py:." \
    --add-data "models:models" \
    --add-data "data:data" \
    --add-data ".streamlit:.streamlit" \
    --hidden-import "sklearn.utils._weight_vector" \
    --hidden-import "sklearn.neighbors._partition_nodes" \
    --hidden-import "sklearn.ensemble._forest" \
    --hidden-import "sklearn.ensemble" \
    --hidden-import "sklearn.tree._tree" \
    --hidden-import "sklearn.preprocessing._label" \
    --hidden-import "streamlit" \
    --hidden-import "altair" \
    --hidden-import "pandas" \
    --hidden-import "numpy" \
    --hidden-import "matplotlib" \
    --hidden-import "seaborn" \
    --collect-all streamlit \
    --collect-all altair \
    app_launcher.py

echo ""
echo "======================================================================"
echo "✓ Build complete!"
echo "======================================================================"
echo ""
echo "Application location: dist/AADB_Risk_Predictor.app"
echo ""
echo "To run the application:"
echo "  1. Open dist/AADB_Risk_Predictor.app"
echo "  2. Or double-click it in Finder"
echo ""
echo "To distribute:"
echo "  1. Copy dist/AADB_Risk_Predictor.app to Applications folder"
echo "  2. Or create a DMG: hdiutil create -volname AADB -srcfolder dist/AADB_Risk_Predictor.app -ov -format UDZO AADB_Risk_Predictor.dmg"
echo ""
echo "======================================================================"

