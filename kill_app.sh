#!/bin/bash
################################################################################
# Kill any running instances of the AADB Risk Predictor
# Use this if the app gets stuck or is running multiple times
################################################################################

echo "Stopping any running AADB Risk Predictor instances..."
echo ""

# Kill any Python processes running Streamlit on port 8501
STREAMLIT_PIDS=$(lsof -ti:8501)
if [ -n "$STREAMLIT_PIDS" ]; then
    echo "Found Streamlit processes on port 8501:"
    echo "$STREAMLIT_PIDS"
    echo ""
    echo "Killing processes..."
    kill $STREAMLIT_PIDS 2>/dev/null || kill -9 $STREAMLIT_PIDS 2>/dev/null
    echo "✓ Processes terminated"
else
    echo "No Streamlit processes found on port 8501"
fi

echo ""
echo "Cleaning up any remaining AADB processes..."
pkill -f "AADB_Risk_Predictor" 2>/dev/null

echo ""
echo "Removing lock files..."
rm -f ~/.aadb_browser_opened 2>/dev/null
rm -f ~/.aadb_app.lock 2>/dev/null
rm -f ~/.aadb_running.lock 2>/dev/null

echo ""
echo "✓ Cleanup complete!"
echo ""
echo "You can now safely launch the app again."

