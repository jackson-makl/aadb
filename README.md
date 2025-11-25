# AADB Data Access Workflow Analysis System

**Comprehensive analysis and predictive modeling for the AIM-AHEAD Data Bridge (AADB) data access workflow**

> 🚀 **Quick Start**: Run `./start.sh` to automatically install, train, and launch the web app!

---

## Executive Summary

This system provides end-to-end analysis of the AADB data access workflow to identify bottlenecks, predict at-risk projects, and deliver actionable recommendations for process improvement. It addresses all requirements for NIH reporting and operational optimization.

### Key Capabilities

- **Process Analysis**: Historical data analysis with bottleneck identification and severity scoring
- **Risk Prediction**: Machine learning model to identify at-risk awardees before delays occur
- **Operational Insights**: Data-driven recommendations for reducing IRB and DUA delays
- **NIH Reporting**: Automated generation of monthly reports and executive dashboards

### Performance Targets

- **Overall Target**: 100 days from NOA to data access (≥80% compliance)
- **IRB Process**: ≤60 days (NIH requirement)
- **DUA Process**: ≤90 days (NIH requirement)
- **Critical Path**: 78 days (optimal scenario from network diagram)

---

## Quick Start

### 🎯 Automated Setup (Recommended)

**Run everything with one command:**

```bash
./start.sh
```

This single command will:
1. ✅ Check if Python 3.8+ is installed (installs if missing)
2. ✅ Create a virtual environment (if it doesn't exist)
3. ✅ Install all required packages from `requirements.txt`
4. ✅ Train the machine learning model (if not already trained)
5. ✅ Launch the Streamlit web application

**Expected runtime**: 
- First run: 3-5 minutes (installs packages + trains model)
- Subsequent runs: 30 seconds (just launches app)

The app will automatically open in your browser at `http://localhost:8501`

**Prerequisites:**
- **macOS**: Homebrew installed (for automatic Python installation)
- **Linux**: sudo access (for package installation)
- **All OS**: Internet connection for downloading packages

**Troubleshooting:**
- Permission denied: `chmod +x start.sh && ./start.sh`
- Port already in use: `streamlit run app.py --server.port 8502`

### 📋 Manual Setup

If you prefer step-by-step control:

```bash
# 1. Activate virtual environment
source env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run complete analysis and train model
python main_analysis.py

# 4. Launch web app
streamlit run app.py
```

**What the analysis does:**
1. Load and preprocess all data files
2. Perform comprehensive process analysis
3. Train risk prediction model with cross-validation
4. Generate visualizations and dashboards
5. Create stakeholder reports

**Expected runtime**: 1-3 minutes

### 🌐 Using the Web Application

Once the Streamlit app is running:

**1. Enter Project Details** (in the sidebar):
- Select the program type
- Days from NOA to initial consultation
- Days from intake to IRB/DUA submission
- Intake month and quarter

**2. Click "Predict Risk"** to generate prediction

**3. Review Results:**
- Risk level: Low, Medium, High, or Critical
- Risk probability percentage
- Model confidence score
- Actionable recommendations
- Visual risk indicator gauge
- Feature importance chart

**4. Download Report:**
- Export prediction results as CSV
- Includes all inputs and predictions
- Timestamped for record-keeping

**App Features:**
- ✅ Interactive risk prediction form
- 📊 Visual risk indicators and charts
- 📋 Actionable recommendations by risk level
- 💾 Export results to CSV
- 📈 Feature importance visualization
- ℹ️ Model performance metrics

**Stopping the App:**
Press `Ctrl+C` in the terminal to stop the server.

### 📁 View Generated Reports

```bash
# Check analysis reports
open reports/

# View visualization dashboards
open visualizations/
```

---

## Project Structure

```
aadb_analysis/
│
├── data_loader.py              # Data loading and preprocessing
├── process_analyzer.py         # Process flow and bottleneck analysis
├── risk_predictor.py          # Machine learning risk prediction
├── visualizations.py          # Dashboard and chart generation
├── report_generator.py        # Stakeholder report creation
├── main_analysis.py           # Main orchestration script
├── config.py                  # Configuration and benchmarks
│
├── app.py                     # Streamlit web application
├── start.sh                   # Automated setup & launch script
│
├── requirements.txt           # Python dependencies
├── README.md                  # This file (complete documentation)
├── AADB_Complete_Analysis.ipynb  # Jupyter notebook interface
│
├── data/                      # Input CSV files
│   ├── AADB_DUA_TRACKER.csv
│   ├── AADB_IRB_TRACKER.csv
│   └── YEAR_3_AADB_DATA_ACCESS_MASTER_TRACKER.csv
│
└── [Generated Outputs]
    ├── processed_data/        # Cleaned datasets
    ├── models/                # Trained ML model (risk_predictor.pkl)
    ├── reports/               # Generated stakeholder reports
    └── visualizations/        # Charts and dashboards
```

---

## System Components

### 1. Data Loader (`data_loader.py`)

**Purpose**: Load and preprocess all AADB tracker data

**Key Features**:
- Loads three CSV files (Master Tracker, DUA Tracker, IRB Tracker)
- Standardizes dates and numeric fields
- Creates performance flags (on_target, delayed)
- Extracts temporal features for trend analysis
- Saves processed data for future use

**Benchmarks Used** (from network diagram):
- Total process: 78 days (optimal), 100 days (target)
- IRB: 60 days
- DUA: 90 days
- Intake: 7 days
- PE turnaround: 30 days

### 2. Process Analyzer (`process_analyzer.py`)

**Purpose**: Comprehensive analysis of process workflow

**Key Features**:
- Stage-by-stage performance metrics
- Bottleneck identification with severity scoring (1x, 1.5x, 2x+ benchmark)
- Program-specific performance comparison
- Temporal trend analysis (monthly, quarterly)
- Risk factor identification
- Best practices from successful projects

**Outputs**:
- Stage performance tables
- Bottleneck severity rankings
- Trend analysis
- Risk distribution

### 3. Risk Predictor (`risk_predictor.py`)

**Purpose**: Machine learning model for early warning system

**Key Features**:
- Trains multiple ML models (Random Forest, Gradient Boosting, Logistic Regression)
- Automatically selects best performer
- Predicts risk probability for new projects
- Generates early warning dashboard
- Provides confidence intervals
- Saves model for future predictions

**Risk Levels**:
- **Low**: <25% probability of delay
- **Medium**: 25-50% probability
- **High**: 50-75% probability
- **Critical**: >75% probability

### 4. Visualizer (`visualizations.py`)

**Purpose**: Create comprehensive visual analytics

**Key Features**:
- Executive dashboard (8-panel multi-chart)
- Process flow diagram with bottleneck highlighting
- Risk distribution charts
- Temporal trend plots
- Program performance comparisons
- NIH compliance gauges

**Color Coding**:
- 🟢 Green: On target (≤benchmark)
- 🟠 Orange: Moderate delay (1-1.5x benchmark)
- 🔴 Red: Critical bottleneck (>1.5x benchmark)

### 5. Report Generator (`report_generator.py`)

**Purpose**: Generate stakeholder reports

**Key Features**:
- Executive summaries (text format)
- NIH monthly reports (formatted for meetings)
- Comprehensive Excel workbooks (multiple sheets)
- Critical issues identification
- Actionable recommendations prioritized by impact

### 6. Main Analysis Script (`main_analysis.py`)

**Purpose**: Orchestrate complete workflow

**Usage**: Single command to run everything
```bash
python main_analysis.py
```

### 7. Configuration (`config.py`)

**Purpose**: Central configuration and settings

**Contains**:
- All benchmarks from network diagram
- NIH performance targets
- Risk assessment thresholds
- Dataset information
- Program types
- Helper functions

---

## Data Requirements

The system expects three CSV files. Place them in the `data/` subdirectory:

```
aadb_analysis/
└── data/
    ├── YEAR_3_AADB_DATA_ACCESS_MASTER_TRACKER.csv
    ├── AADB_DUA_TRACKER.csv
    └── AADB_IRB_TRACKER.csv
```

**File Descriptions**:

1. **YEAR_3_AADB_DATA_ACCESS_MASTER_TRACKER.csv**
   - Main project tracking data
   - Contains: Name, Program, dates, duration metrics

2. **AADB_DUA_TRACKER.csv**
   - DUA process details
   - Contains: DUA status, turnaround times, contact info

3. **AADB_IRB_TRACKER.csv**
   - IRB process details
   - Contains: IRB status, determination, submission dates

*Note: The system will auto-detect files in `data/` subdirectory or parent directory.*

---

## Output Files

### Reports (`reports/`)

- **Executive_Summary_[timestamp].txt**
  - High-level overview for leadership
  - Key metrics and findings
  - Top 3-5 recommendations

- **NIH_Monthly_Report_[timestamp].txt**
  - Formatted for NIH monthly meetings
  - KPI performance tables
  - Progress updates and challenges

- **Comprehensive_Analysis_[timestamp].xlsx**
  - Multi-sheet Excel workbook
  - Executive metrics
  - Stage analysis
  - Program performance
  - Risk assessment
  - Project details
  - Recommendations

- **early_warning_dashboard.csv**
  - Current projects with risk scores
  - Intervention recommendations

### Visualizations (`visualizations/`)

- **executive_dashboard.png**
  - 8-panel comprehensive dashboard
  - KPIs, compliance rates, trends

- **process_flow_diagram.png**
  - Visual process map
  - Color-coded bottlenecks

- **risk_predictions.png** (if applicable)
  - Risk level distribution
  - Probability histogram

### Models (`models/`)

- **risk_predictor.pkl**
  - Trained machine learning model
  - Reusable for future predictions

### Processed Data (`processed_data/`)

- **master_tracker_processed.csv**
- **dua_tracker_processed.csv**
- **irb_tracker_processed.csv**

---

## Usage Examples

### Basic Usage

```bash
# Run complete analysis
python main_analysis.py
```

### Jupyter Notebook (Interactive)

```bash
jupyter notebook AADB_Complete_Analysis.ipynb
```

### Custom Analysis (Python)

```python
from data_loader import AADBDataLoader
from process_analyzer import AADBProcessAnalyzer
from risk_predictor import AADBRiskPredictor

# Load data (auto-detects data directory)
loader = AADBDataLoader()
loader.load_all_data()
loader.preprocess_all()

# Analyze bottlenecks
analyzer = AADBProcessAnalyzer(loader)
analyzer.analyze_process_stages()
analyzer.analyze_risk_factors()

# Train risk model
predictor = AADBRiskPredictor(loader)
predictor.train_model()
predictor.save_model()
```

### Predict Risk for New Project

```python
from risk_predictor import AADBRiskPredictor
from data_loader import AADBDataLoader

# Load saved model
loader = AADBDataLoader()
predictor = AADBRiskPredictor(loader)
predictor.load_model('models/risk_predictor.pkl')

# New project details
new_project = {
    'Program': 'Research Fellowship',
    'days_noa_to_consult': 30,
    'days_intake_to_submit': 15,
    'intake_month': 10,
    'intake_quarter': 4
}

# Predict risk
risk = predictor.predict_risk(new_project)

print(f"Risk Level: {risk['risk_level']}")
print(f"Risk Probability: {risk['risk_probability']:.1%}")
print(f"Confidence: {risk['confidence']:.1%}")
```

**Example Output**:
```
Risk Level: Medium
Risk Probability: 62.3%
Confidence: 72.5%
```

---

## Key Metrics Explained

### NIH Compliance Rates

- **IRB Compliance**: Percentage of projects with IRB process ≤60 days
- **DUA Compliance**: Percentage of projects with DUA process ≤90 days
- **Target**: ≥80% for both

### Bottleneck Severity

- **Severity Ratio** = Actual Duration / Benchmark Duration
- **1.0x**: On target
- **1.0-1.5x**: Moderate delay (⚠️)
- **>1.5x**: Critical bottleneck (🔴)

### Risk Scores

Projects are scored 0-6+ based on delay indicators:
- **Early warnings** (1 point each): Late consult, late submission
- **Major delays** (2 points each): IRB delay, DUA delay
- **Critical delays** (3 points): Extreme total delay (>180 days)

**Risk Levels**:
- 0-2: Low Risk
- 3-4: Medium Risk
- 5+: High Risk

---

## Recommendations Output

The system provides prioritized recommendations based on:

1. **Bottleneck Severity**: Stages with highest severity ratios
2. **Compliance Gaps**: Areas below 80% target
3. **Risk Patterns**: Common factors in delayed projects
4. **Best Practices**: Characteristics of successful projects

Example recommendations:
- "Streamline IRB process: Reduce from 89 to 60 days through expedited review procedures"
- "Implement early warning system: Flag projects at risk during initial consult"
- "Standardize onboarding: Adopt procedures from fast-track programs"

---

## Customization

### Modify Benchmarks

Edit `config.py`:

```python
STAGE_BENCHMARKS = {
    'total_process_target': 100,  # Change target
    'irb_days': 60,
    'dua_days': 90,
    # ...
}
```

### Extend Analysis

Create custom analyzer:

```python
from process_analyzer import AADBProcessAnalyzer

class CustomAnalyzer(AADBProcessAnalyzer):
    def custom_analysis(self):
        # Your custom code here
        pass
```

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'pandas'`

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

---

**Issue**: `FileNotFoundError: YEAR_3_AADB_DATA_ACCESS_MASTER_TRACKER.csv`

**Solution**: Ensure CSV files are in the `data/` subdirectory
```
aadb_analysis/
├── data/
│   ├── YEAR_3_AADB_DATA_ACCESS_MASTER_TRACKER.csv  ← Here
│   ├── AADB_DUA_TRACKER.csv                        ← Here
│   └── AADB_IRB_TRACKER.csv                        ← Here
└── main_analysis.py  ← Run from here
```

---

**Issue**: Visualizations not displaying

**Solution**: Figures are auto-saved to `visualizations/` directory even if windows don't display

---

**Issue**: "Not enough data to train model"

**Solution**: System requires minimum 10 data points. If you have fewer, predictions will be limited but analysis will still run.

---

## Monthly Workflow Recommendation

### Week 1: Data Update
- Export latest data from Monday.com
- Replace CSV files in parent directory

### Week 2: Run Analysis
```bash
python main_analysis.py
```

### Week 3: Review Findings
- Check Executive Summary in `reports/`
- Review visualizations in `visualizations/`
- Identify high-risk projects from early warning dashboard

### Week 4: NIH Reporting
- Present NIH Monthly Report in stakeholder meeting
- Implement top recommendations
- Monitor high-risk projects

---

## Technical Specifications

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Data Processing**: pandas, numpy
- **Reports**: Excel (openpyxl), text files

### Dependencies

```
pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.2.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
openpyxl >= 3.0.0
```

---

## Project Background

### Based On

- **Network Diagram**: Critical path analysis (78-day optimal path)
- **NIH Requirements**: IRB ≤60 days, DUA ≤90 days
- **AI CoLab Goals**: 100-day overall target
- **Historical Data**: AADB tracker exports from Monday.com

### Objectives Addressed

1. **Process Analysis**: Historical data review, trend identification, bottleneck visualization
2. **Predictive Model**: Risk assessment for at-risk awardees, early warning system
3. **Operational Optimization**: Data-driven recommendations, enhanced NIH reporting

---

## Success Criteria

Your AADB process is performing well when:

✅ IRB compliance ≥80% (≤60 days)  
✅ DUA compliance ≥80% (≤90 days)  
✅ Average process time ≤100 days  
✅ High-risk projects <10%  
✅ No critical bottlenecks (>2x benchmark)

---

## Contact Information

**Primary Contact**: Prabhjeet Singh, Project Manager  
**Email**: Prabhjeet.Singh@medstar.net

**Secondary Contact**: Sara Stienecker, Associate Director  
**Email**: Sara.L.Stienecker@medstar.net

**Leadership**: Dr. Nawar Shara, AIM-AHEAD MPI  
**Email**: Nawar.Shara@medstar.net

---

## License and Usage

Developed for AI CoLab AADB initiative and NIH reporting purposes.

---

## Version Information

**Version**: 1.0  
**Last Updated**: November 2025  
**Author**: MSBA Capstone Project  
**For**: AI CoLab - AIM-AHEAD Data Bridge

---

**Ready to get started?**

```bash
pip install -r requirements.txt
python main_analysis.py
```

Your comprehensive analysis, risk predictions, and NIH-ready reports will be generated automatically.
