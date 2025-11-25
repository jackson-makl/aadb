# Building Native Desktop Applications

This guide explains how to build standalone native applications for macOS and Windows.

## 🎯 Overview

The AADB Risk Prediction System can be packaged as:
- **macOS**: `.app` bundle (double-click to run)
- **Windows**: `.exe` executable (double-click to run)

Both versions include all dependencies and don't require Python installation.

---

## 📋 Prerequisites

### Common Requirements
- Python 3.8 or higher
- All dependencies installed (`pip install -r requirements.txt`)
- Trained model file (`models/risk_predictor.pkl`)

### Platform-Specific

**macOS:**
- macOS 10.13 or higher
- Xcode Command Line Tools (install with: `xcode-select --install`)

**Windows:**
- Windows 10 or higher
- Microsoft Visual C++ Redistributable

---

## 🍎 Building for macOS

### Step 1: Prepare Environment

```bash
# Navigate to project directory
cd /Users/jmakl/capstone/aadb_analysis

# Activate virtual environment
source env/bin/activate

# Install build dependencies
pip install pyinstaller pillow
```

### Step 2: Run Build Script

```bash
# Make script executable
chmod +x build_mac.sh

# Build the app
./build_mac.sh
```

### Step 3: Test the Application

```bash
# Run from command line
open dist/AADB_Risk_Predictor.app

# Or double-click in Finder
```

### Step 4: Distribute (Optional)

**Option 1: Copy to Applications**
```bash
cp -r dist/AADB_Risk_Predictor.app /Applications/
```

**Option 2: Create DMG Installer**
```bash
hdiutil create -volname "AADB Risk Predictor" \
    -srcfolder dist/AADB_Risk_Predictor.app \
    -ov -format UDZO \
    AADB_Risk_Predictor.dmg
```

**Output:**
- **Application**: `dist/AADB_Risk_Predictor.app`
- **Size**: ~150-200 MB (includes all dependencies)
- **Runs on**: macOS 10.13+ (High Sierra and later)

---

## 🪟 Building for Windows

### Step 1: Prepare Environment

```cmd
REM Navigate to project directory
cd C:\path\to\aadb_analysis

REM Activate virtual environment
call env\Scripts\activate.bat

REM Install build dependencies
pip install pyinstaller pillow
```

### Step 2: Run Build Script

```cmd
REM Simply double-click build_windows.bat
REM Or run from command line:
build_windows.bat
```

### Step 3: Test the Application

```cmd
REM Run the executable
dist\AADB_Risk_Predictor.exe

REM Or double-click in File Explorer
```

### Step 4: Distribute (Optional)

**Option 1: Simple Distribution**
- Copy `dist\AADB_Risk_Predictor.exe` to any location
- Share the single .exe file

**Option 2: Create Installer with Inno Setup**
1. Download Inno Setup: https://jrsoftware.org/isdl.php
2. Create installer script (provided below)
3. Compile to create `AADB_Risk_Predictor_Setup.exe`

**Output:**
- **Executable**: `dist\AADB_Risk_Predictor.exe`
- **Size**: ~150-200 MB (includes all dependencies)
- **Runs on**: Windows 10+ (64-bit)

---

## 🎨 Customizing the Icon (Optional)

### macOS Icon (.icns)

```bash
# Convert PNG to ICNS
# 1. Create 1024x1024 PNG image: icon.png
# 2. Convert:
mkdir icon.iconset
sips -z 16 16     icon.png --out icon.iconset/icon_16x16.png
sips -z 32 32     icon.png --out icon.iconset/icon_16x16@2x.png
sips -z 32 32     icon.png --out icon.iconset/icon_32x32.png
sips -z 64 64     icon.png --out icon.iconset/icon_32x32@2x.png
sips -z 128 128   icon.png --out icon.iconset/icon_128x128.png
sips -z 256 256   icon.png --out icon.iconset/icon_128x128@2x.png
sips -z 256 256   icon.png --out icon.iconset/icon_256x256.png
sips -z 512 512   icon.png --out icon.iconset/icon_256x256@2x.png
sips -z 512 512   icon.png --out icon.iconset/icon_512x512.png
sips -z 1024 1024 icon.png --out icon.iconset/icon_512x512@2x.png
iconutil -c icns icon.iconset
mv icon.icns app_icon.icns
```

### Windows Icon (.ico)

Use online converter or:
```bash
pip install pillow
python -c "from PIL import Image; img = Image.open('icon.png'); img.save('app_icon.ico', sizes=[(256,256)])"
```

Or use: https://convertico.com/

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: "Module not found" error when running app

**Solution**: Add missing module to build script:
```bash
# In build_mac.sh or build_windows.bat, add:
--hidden-import "module_name"
```

---

**Issue**: App crashes on startup

**Solution**: Run from terminal to see error:
```bash
# macOS
./dist/AADB_Risk_Predictor.app/Contents/MacOS/AADB_Risk_Predictor

# Windows (in CMD)
dist\AADB_Risk_Predictor.exe
```

---

**Issue**: "App is damaged" on macOS (Catalina+)

**Solution**: Remove quarantine attribute:
```bash
xattr -cr dist/AADB_Risk_Predictor.app
```

Or properly code sign the app:
```bash
codesign --force --deep --sign - dist/AADB_Risk_Predictor.app
```

---

**Issue**: Windows Defender blocks the .exe

**Solution**: This is normal for PyInstaller apps. Either:
1. Add exception in Windows Defender
2. Submit to Microsoft for analysis
3. Code sign the executable (requires certificate)

---

**Issue**: App opens multiple browser windows non-stop (macOS)

**Solution**: The app has built-in protection against this, but if it still happens:

1. **Kill all stuck instances:**
   ```bash
   ./kill_app.sh
   ```

2. **If the script doesn't exist, manually kill processes:**
   ```bash
   # Find and kill processes on port 8501
   lsof -ti:8501 | xargs kill
   
   # Kill any AADB processes
   pkill -f "AADB_Risk_Predictor"
   ```

3. **Wait 5 seconds, then relaunch the app**

**Why this happens:**
- The app checks if port 8501 is already in use
- If already running, it opens the existing instance instead of starting a new one
- Previous versions didn't have this protection (now fixed)

**If you built the app before November 25, 2025, rebuild it:**
```bash
./build_mac.sh
```

---

## 📦 File Size Optimization

To reduce app size:

**Option 1: Exclude unnecessary packages**
```bash
# Add to build script:
--exclude-module matplotlib
--exclude-module seaborn
# (Only if not used in visualizations)
```

**Option 2: Use UPX compression**
```bash
brew install upx  # macOS
# Windows: Download from https://upx.github.io/

# Add to build script:
--upx-dir /path/to/upx
```

**Option 3: One-folder instead of one-file (Windows)**
```bash
# Change in build_windows.bat:
--onedir  # Instead of --onefile
# Results in folder with .exe + dependencies (faster startup)
```

---

## 🚀 Quick Reference

### Build Commands

```bash
# macOS
./build_mac.sh

# Windows
build_windows.bat
```

### Output Locations

```
macOS:    dist/AADB_Risk_Predictor.app
Windows:  dist/AADB_Risk_Predictor.exe
```

### Testing

```bash
# macOS
open dist/AADB_Risk_Predictor.app

# Windows
start dist\AADB_Risk_Predictor.exe
```

---

## 📝 Notes

1. **First-time build** takes 5-10 minutes (downloads dependencies)
2. **Subsequent builds** take 2-3 minutes
3. **App size** is large (~150-200MB) because it includes Python + all packages
4. **No Python required** for end users - apps are fully standalone
5. **Model file** is bundled - any updates require rebuild

---

## ✅ Distribution Checklist

Before distributing:

- [ ] Test app on clean machine (without Python installed)
- [ ] Verify model predictions work correctly
- [ ] Check all visualizations display properly
- [ ] Test with different input values
- [ ] Verify CSV export works
- [ ] Test on target OS version
- [ ] Include README or user guide
- [ ] Consider code signing (macOS/Windows)

---

## 🆘 Support

If you encounter issues:

1. Check terminal output for specific errors
2. Ensure all source files are present
3. Verify model file exists: `models/risk_predictor.pkl`
4. Try cleaning and rebuilding: `rm -rf build dist`
5. Check PyInstaller logs in `build/` directory

---

**Ready to build!** 🚀

Run the appropriate build script for your platform and your standalone app will be ready in the `dist/` folder.

