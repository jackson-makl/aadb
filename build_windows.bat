@echo off
REM ==============================================================================
REM Build script for Windows standalone application
REM Creates AADB_Risk_Predictor.exe for Windows
REM ==============================================================================

echo ======================================================================
echo Building AADB Risk Predictor - Windows Application
echo ======================================================================
echo.

REM Activate virtual environment if it exists
if exist env\Scripts\activate.bat (
    echo Activating virtual environment...
    call env\Scripts\activate.bat
)

REM Install PyInstaller if not already installed
echo Installing build dependencies...
pip install pyinstaller pillow --quiet

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist AADB_Risk_Predictor.exe del AADB_Risk_Predictor.exe

REM Check if icon exists
set ICON_FLAG=
if exist app_icon.ico (
    set ICON_FLAG=--icon=app_icon.ico
    echo Using custom icon: app_icon.ico
)

REM Create the Windows executable
echo.
echo Building Windows application...
echo This may take a few minutes...
echo.

pyinstaller --noconfirm --clean ^
    --name "AADB_Risk_Predictor" ^
    --onefile ^
    --windowed ^
    %ICON_FLAG% ^
    --add-data "app.py;." ^
    --add-data "data_loader.py;." ^
    --add-data "risk_predictor.py;." ^
    --add-data "config.py;." ^
    --add-data "models;models" ^
    --add-data "data;data" ^
    --add-data ".streamlit;.streamlit" ^
    --hidden-import "sklearn.utils._weight_vector" ^
    --hidden-import "sklearn.neighbors._partition_nodes" ^
    --hidden-import "sklearn.ensemble._forest" ^
    --hidden-import "sklearn.ensemble" ^
    --hidden-import "sklearn.tree._tree" ^
    --hidden-import "sklearn.preprocessing._label" ^
    --hidden-import "streamlit" ^
    --hidden-import "altair" ^
    --hidden-import "pandas" ^
    --hidden-import "numpy" ^
    --hidden-import "matplotlib" ^
    --hidden-import "seaborn" ^
    --collect-all streamlit ^
    --collect-all altair ^
    app_launcher.py

echo.
echo ======================================================================
echo Build complete!
echo ======================================================================
echo.
echo Application location: dist\AADB_Risk_Predictor.exe
echo.
echo To run the application:
echo   1. Navigate to dist folder
echo   2. Double-click AADB_Risk_Predictor.exe
echo.
echo To distribute:
echo   1. Copy dist\AADB_Risk_Predictor.exe to desired location
echo   2. Or create installer using Inno Setup
echo.
echo ======================================================================

pause

