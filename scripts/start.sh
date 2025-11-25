#!/bin/bash

################################################################################
# AADB Analysis - Complete Setup and Run Script
# This script handles installation, setup, and execution of the AADB analysis
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory (scripts folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Project root (parent of scripts folder)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "${BLUE}"
echo "════════════════════════════════════════════════════════════════"
echo "  AADB Risk Prediction System - Setup & Launch"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"

################################################################################
# Step 1: Check/Install Python
################################################################################

echo -e "${YELLOW}[1/5] Checking Python installation...${NC}"

# Function to check if Python version is adequate
check_python_version() {
    local python_cmd=$1
    if command -v "$python_cmd" &> /dev/null; then
        version=$($python_cmd --version 2>&1 | awk '{print $2}')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        
        if [ "$major" -eq 3 ] && [ "$minor" -ge 8 ]; then
            echo -e "${GREEN}✓ Found $python_cmd (version $version)${NC}"
            PYTHON_CMD=$python_cmd
            return 0
        fi
    fi
    return 1
}

# Check for Python 3
PYTHON_CMD=""
if check_python_version "python3"; then
    :
elif check_python_version "python"; then
    :
else
    echo -e "${RED}✗ Python 3.8+ is not installed${NC}"
    echo -e "${YELLOW}Installing Python...${NC}"
    
    # Detect OS and install Python
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo "Installing Python via Homebrew..."
            brew install python3
            PYTHON_CMD="python3"
        else
            echo -e "${RED}Error: Homebrew not found. Please install Python 3.8+ manually:${NC}"
            echo "Visit: https://www.python.org/downloads/"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            echo "Installing Python via apt-get..."
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv
            PYTHON_CMD="python3"
        elif command -v yum &> /dev/null; then
            echo "Installing Python via yum..."
            sudo yum install -y python3 python3-pip
            PYTHON_CMD="python3"
        else
            echo -e "${RED}Error: Could not determine package manager. Please install Python 3.8+ manually.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: Unsupported OS. Please install Python 3.8+ manually.${NC}"
        exit 1
    fi
    
    # Verify installation
    if check_python_version "$PYTHON_CMD"; then
        echo -e "${GREEN}✓ Python installed successfully${NC}"
    else
        echo -e "${RED}✗ Python installation failed${NC}"
        exit 1
    fi
fi

################################################################################
# Step 2: Create Virtual Environment
################################################################################

echo ""
echo -e "${YELLOW}[2/5] Setting up virtual environment...${NC}"

if [ -d "env" ]; then
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
else
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv env
    
    if [ -d "env" ]; then
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${RED}✗ Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}✗ Failed to activate virtual environment${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Virtual environment activated${NC}"

################################################################################
# Step 3: Install/Update Requirements
################################################################################

echo ""
echo -e "${YELLOW}[3/5] Installing Python packages...${NC}"

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Check if requirements are already installed
if [ -f "requirements.txt" ]; then
    echo "Checking installed packages..."
    
    # Simple check: if streamlit is installed, assume requirements are met
    if python -c "import streamlit" 2>/dev/null; then
        echo -e "${GREEN}✓ Required packages already installed${NC}"
        
        # Ask if user wants to reinstall
        read -p "Reinstall packages? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Installing requirements..."
            pip install -r requirements.txt
            echo -e "${GREEN}✓ Packages reinstalled${NC}"
        fi
    else
        echo "Installing requirements..."
        pip install -r requirements.txt
        echo -e "${GREEN}✓ All packages installed successfully${NC}"
    fi
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi

################################################################################
# Step 4: Check Data Files and Run Analysis
################################################################################

echo ""
echo -e "${YELLOW}[4/5] Preparing data and training model...${NC}"

# Check if data files exist
DATA_DIR="data"
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}✗ Data directory not found${NC}"
    echo "Please ensure the 'data' directory exists with CSV files."
    exit 1
fi

# Count CSV files
csv_count=$(find "$DATA_DIR" -name "*.csv" 2>/dev/null | wc -l)
if [ "$csv_count" -lt 1 ]; then
    echo -e "${YELLOW}⚠ Warning: No CSV files found in data directory${NC}"
    echo "The analysis may not work without data files."
fi

# Check if model already exists
MODEL_FILE="models/risk_predictor.pkl"
if [ -f "$MODEL_FILE" ]; then
    echo -e "${GREEN}✓ Trained model found at $MODEL_FILE${NC}"
    
    read -p "Retrain model? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running analysis and training model..."
        echo "═══════════════════════════════════════════════════════════"
        python src/main_analysis.py
        echo "═══════════════════════════════════════════════════════════"
        echo -e "${GREEN}✓ Analysis complete and model trained${NC}"
    fi
else
    echo "No trained model found. Running analysis..."
    echo "═══════════════════════════════════════════════════════════"
    
    # Check if main_analysis.py exists
    if [ -f "main_analysis.py" ]; then
        python src/main_analysis.py
    else
        echo -e "${YELLOW}⚠ main_analysis.py not found. Using alternative training...${NC}"
        python -c "
from data_loader import AADBDataLoader
from risk_predictor import AADBRiskPredictor

# Load and preprocess data
loader = AADBDataLoader()
loader.load_all_data().preprocess_all()

# Train model
predictor = AADBRiskPredictor(loader)
predictor.train_model()
predictor.save_model()

print('Model training complete!')
"
    fi
    
    echo "═══════════════════════════════════════════════════════════"
    
    if [ -f "$MODEL_FILE" ]; then
        echo -e "${GREEN}✓ Model trained and saved successfully${NC}"
    else
        echo -e "${RED}✗ Model training failed${NC}"
        echo "You may need to check the data and run the analysis manually."
        exit 1
    fi
fi

################################################################################
# Step 5: Launch Streamlit App
################################################################################

echo ""
echo -e "${YELLOW}[5/5] Launching Streamlit application...${NC}"

if [ ! -f "src/app.py" ]; then
    echo -e "${RED}✗ src/app.py not found${NC}"
    exit 1
fi

echo -e "${GREEN}"
echo "════════════════════════════════════════════════════════════════"
echo "  🚀 Starting AADB Risk Prediction App"
echo "════════════════════════════════════════════════════════════════"
echo -e "${NC}"
echo ""
echo -e "${BLUE}The app will open in your browser automatically.${NC}"
echo -e "${BLUE}If it doesn't, visit: ${NC}${GREEN}http://localhost:8501${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Run Streamlit
streamlit run src/app.py

# This line will only execute if Streamlit exits
echo ""
echo -e "${GREEN}✓ Application closed${NC}"
echo -e "${BLUE}Thank you for using AADB Risk Prediction System!${NC}"

