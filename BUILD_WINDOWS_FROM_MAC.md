# Building Windows Version from macOS

You have several options to build the Windows `.exe` version while working on a Mac.

---

## ✅ Option 1: GitHub Actions (Recommended - FREE)

**Build both Mac and Windows versions automatically in the cloud**

### Setup (One-time):

1. **Create a GitHub repository** (if you don't have one):
   ```bash
   cd /Users/jmakl/capstone/aadb_analysis
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Push to GitHub**:
   ```bash
   gh repo create aadb_analysis --private --source=. --remote=origin --push
   # Or use GitHub Desktop / web interface
   ```

### Build Apps:

1. Go to your repository on GitHub.com
2. Click **"Actions"** tab
3. Click **"Build Native Apps"** workflow
4. Click **"Run workflow"** button
5. Wait 5-10 minutes

### Download:

- Go to the workflow run
- Download artifacts:
  - `AADB-Risk-Predictor-Windows.zip` (Windows .exe)
  - `AADB-Risk-Predictor-macOS.zip` (macOS .dmg)

**Pros:**
- ✅ Completely free
- ✅ Builds both platforms simultaneously
- ✅ No Windows machine needed
- ✅ Reproducible builds

**Cons:**
- ⚠️ Requires GitHub account
- ⚠️ 5-10 minute build time

---

## Option 2: Windows Virtual Machine

**Run Windows on your Mac**

### Using Parallels Desktop (Paid - $99/year):

1. Install [Parallels Desktop](https://www.parallels.com/)
2. Create Windows 11 VM
3. Share your project folder with Windows
4. In Windows VM:
   ```cmd
   cd Z:\Users\jmakl\capstone\aadb_analysis
   python -m venv env
   env\Scripts\activate
   pip install -r requirements.txt
   build_windows.bat
   ```

### Using UTM (Free):

1. Install [UTM](https://mac.getutm.app/) from App Store
2. Create Windows 11 VM
3. Transfer files via shared folder
4. Build in Windows

**Pros:**
- ✅ Full Windows environment
- ✅ Can test the app before distribution

**Cons:**
- ⚠️ Requires ~40GB disk space
- ⚠️ Parallels costs money (UTM is free but slower)
- ⚠️ Resource intensive

---

## Option 3: Cloud Windows Machine

**Rent a Windows machine temporarily**

### AWS EC2 Windows:

1. Launch Windows Server instance
2. Copy files via RDP
3. Build and download

### Paperspace / Azure:

Similar process, usually $0.10-0.50/hour

**Pros:**
- ✅ No local resource usage
- ✅ Fast Windows machine

**Cons:**
- ⚠️ Costs money (small amount)
- ⚠️ Requires setup

---

## Option 4: Wine + PyInstaller (Experimental)

**Try to build Windows exe on Mac using Wine**

⚠️ **This is experimental and may not work properly!**

```bash
# Install Wine
brew install wine-stable

# Create Windows Python environment
pip install pyinstaller

# Try cross-compile (often fails)
pyinstaller --target-os=windows AADB_Risk_Predictor.spec
```

**Pros:**
- ✅ No VM needed
- ✅ Fast

**Cons:**
- ❌ Usually doesn't work
- ❌ Executables may be broken
- ❌ Not recommended

---

## Option 5: Build on an Actual Windows Machine

**Give the project to someone with Windows**

### Send them:
1. The entire `aadb_analysis` folder
2. `BUILD_NATIVE_APPS.md` instructions

### They run:
```cmd
cd aadb_analysis
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
build_windows.bat
```

**Pros:**
- ✅ Most reliable
- ✅ Can test on real hardware

**Cons:**
- ⚠️ Requires access to Windows machine

---

## 🎯 Recommended Approach

**For quick/easy builds:** Use **GitHub Actions** (Option 1)

**For frequent development:** Set up **Parallels VM** (Option 2)

**For one-time builds:** Ask someone with Windows or use AWS (Options 3 or 5)

---

## 📦 What You'll Get (Windows)

After building on Windows, you'll have:
- `dist/AADB_Risk_Predictor.exe` (~200MB)
- Single executable file
- Runs on Windows 10/11
- Opens browser window with app

---

## 🚀 Quick Start: GitHub Actions

**Fastest way to get started:**

```bash
# From your Mac
cd /Users/jmakl/capstone/aadb_analysis

# Initialize git if needed
git init
git add .
git commit -m "AADB Risk Predictor app"

# Create GitHub repo
gh repo create aadb-risk-predictor --private --source=. --push

# Or use GitHub Desktop
```

Then:
1. Go to github.com/YOUR_USERNAME/aadb-risk-predictor
2. Actions → Build Native Apps → Run workflow
3. Wait 10 minutes
4. Download both Mac and Windows builds

Done! ✨

---

## Need Help?

- GitHub Actions not working? Check the logs in the Actions tab
- VM issues? Make sure virtualization is enabled in macOS
- Build errors? Check `build/AADB_Risk_Predictor/warn*.txt` for details

