"""
AADB Analysis Configuration
Contains benchmarks, thresholds, and configuration settings based on:
- Network diagram Critical Path analysis
- NIH performance targets
- AI CoLab kickoff presentation requirements
"""

# ============================================================================
# PERFORMANCE BENCHMARKS (from Network Diagram & NIH Targets)
# ============================================================================

# Critical Path from Network Diagram: 78 days (optimal)
CRITICAL_PATH_DAYS = 78

# Overall Target: 100 days from NOA to data access
OVERALL_TARGET_DAYS = 100

# NIH Performance Targets
NIH_TARGETS = {
    'IRB_DAYS': 60,        # NIH target for IRB process
    'DUA_DAYS': 90,        # NIH target for DUA process
    'OVERALL_DAYS': 100,   # Overall target from NOA to data access
    'COMPLIANCE_RATE': 80  # Target compliance rate (%)
}

# Stage-Specific Benchmarks (from Network Diagram)
STAGE_BENCHMARKS = {
    # Pre-access stages
    'orientation': 1,
    'intake': 7,
    'noa_to_consult': 30,
    'finalize_data_request': 10,
    
    # Regulatory clearance
    'local_irb': 30,
    'medstar_irb': 30,
    'dua_process': 42,
    
    # Data preparation
    'dataset_curation': 3,
    'database_build': 20,
    'is_request_setup': 1,
    
    # Access provision
    'user_access_assignment': 1,
    'awardee_access': 1,
    'access_confirmation': 1,
    
    # Combined process metrics
    'total_process_days': CRITICAL_PATH_DAYS,  # Optimal: 78 days
    'total_process_target': 100,                # Realistic target: 100 days
    'intake_turnaround': 7,
    'pe_turnaround': 30,
    'pe_to_fe_turnaround': 20,
    'data_request_submission': 14,
    'data_request_finalization': 30
}

# ============================================================================
# AADB DATASET INFORMATION
# ============================================================================

AADB_DATASETS = {
    'curated_library': {
        'Maternal Health': {
            'size': '25k+',
            'description': 'Adult patients with newborn delivery 2021-2023',
            'features': 'Postnatal follow-up, SDOH data (SVI/AVI)'
        },
        'Chronic Disease': {
            'size': '500k+',
            'description': 'Adult patients with ≥1 chronic disease',
            'features': 'Diabetes, Hypertension, CKD, etc.'
        },
        'Behavioral Health': {
            'size': '150k+',
            'description': 'Patients with mental health diseases',
            'features': 'Depression, PTSD, Anxiety, Schizophrenia, SDOH (ADI/SVI)'
        },
        'Opioid Use and Misuse': {
            'size': '693k+',
            'description': 'Adult patients with outpatient encounters',
            'features': 'Opioid use and misuse analysis'
        },
        'Medical Images': {
            'Breast & Lung Cancer': '61k+ radiographic images (Mammograms, CT Scans)',
            'Brain Images': '~3000 patients with CT and MRI',
            'Cardiac Images': '1000 patients with echocardiograms',
            'Thyroid Images': '~2000 patients with ultrasound'
        }
    },
    'custom_curated': {
        'description': 'Tailored data specific to research needs',
        'capabilities': [
            'Limited datasets and full PHI',
            'Clinical notes',
            'Longitudinal/temporal data',
            'Custom cohort definitions'
        ]
    },
    'total_data_points': '~2 billion',
    'african_american_representation': '31%'
}

# ============================================================================
# PROCESS FLOW STAGES (from Monday.com tracking)
# ============================================================================

MONDAY_TRACKING_BOARDS = {
    'Data Consult': {
        'board': 'AADB Request',
        'fields': ['PI Info', 'Institution', 'Project Description']
    },
    'Intake & Data Specification': {
        'board': 'AADB_RequestForm Responses',
        'fields': ['Data Specifications', 'Methodology', 'Data Feasibility Status']
    },
    'IRB Process': {
        'board': 'AADB IRB Tracker',
        'fields': ['IRB Consult', 'Submission Date', 'Determination Date', 'Determination']
    },
    'DUA Completion': {
        'board': 'AADB DUA Tracker',
        'fields': ['DUA Contact Info', 'Drafting Date', 'Submission Date', 
                  'Time to PE/FE', 'DUA Expiration Date']
    },
    'Database Curation': {
        'board': 'Active AADB Data Curations',
        'fields': ['Assigned Staff', 'Status', 'Cohort size and details']
    },
    'Data Access Provision': {
        'board': 'AADB Data Access Tracker',
        'fields': ['PI Team Info', 'Regulatory Status', 'DB Name', 'Server Access']
    }
}

# ============================================================================
# RISK ASSESSMENT THRESHOLDS
# ============================================================================

RISK_THRESHOLDS = {
    # Risk score calculation weights
    'early_warning_weight': 1,      # Early delays (consult, intake)
    'major_delay_weight': 2,        # IRB, DUA delays
    'critical_delay_weight': 3,     # Extreme total delays
    
    # Risk level thresholds
    'low_risk_max': 2,
    'medium_risk_max': 4,
    'high_risk_min': 5,
    
    # Delay multipliers for severity
    'moderate_delay_multiplier': 1.0,  # 1.0-1.5x benchmark
    'severe_delay_multiplier': 1.5,    # 1.5-2.0x benchmark
    'critical_delay_multiplier': 2.0,  # >2.0x benchmark
    
    # Early warning indicators
    'consult_delay_threshold': 30,      # Days from NOA to consult
    'intake_delay_threshold': 20,       # Days from intake to submission
    'extreme_total_delay': 180          # Days for critical flag
}

# ============================================================================
# REPORTING CONFIGURATION
# ============================================================================

REPORT_CONFIG = {
    'nih_meeting_frequency': 'monthly',
    'key_stakeholders': [
        'NIH Program Officers',
        'AI CoLab Leadership',
        'AADB Project Management',
        'AIM-AHEAD Coordinating Center'
    ],
    'key_metrics': [
        'IRB compliance rate',
        'DUA compliance rate',
        'Average total process time',
        'Number of high-risk projects',
        'Projects exceeding 150 days'
    ],
    'visualization_types': [
        'Executive dashboard',
        'Process flow diagram',
        'Bottleneck analysis',
        'Risk distribution',
        'Temporal trends'
    ]
}

# ============================================================================
# PROJECT DELIVERABLES (from Kickoff Presentation)
# ============================================================================

PROJECT_DELIVERABLES = {
    'objective_1': {
        'title': 'Process, Analyze, and Report on Historical Data',
        'tasks': [
            'Review Monday.com exports and historical data',
            'Identify trends, bottlenecks, and high-risk factors',
            'Develop visualized process maps highlighting inefficiencies',
            'Focus on IRB, DUA, and Database Build delays'
        ]
    },
    'objective_2': {
        'title': 'Predictive & Optimization Model for At-Risk Awardees',
        'tasks': [
            'Use historical data to develop risk assessment model',
            'Identify awardees likely to experience delays',
            'Provide dashboard/early warning system',
            'Enable proactive intervention'
        ]
    },
    'objective_3': {
        'title': 'Operational Optimization & Reporting Enhancements',
        'tasks': [
            'Develop data-driven recommendations',
            'Streamline AADB access process',
            'Focus on reducing IRB & DUA delays',
            'Propose reporting enhancements for tracking and forecasting'
        ]
    }
}

# ============================================================================
# PROGRAM TYPES
# ============================================================================

PROGRAM_TYPES = [
    'Leadership Fellowship',
    'Research Fellowship',
    'PAIR',
    'DICB',
    'CLINAQ Fellowship',
    'Consortium Development Pilot Program',
    'Hub Pilot Program',
    'Mentor - Research Fellowship'
]

# ============================================================================
# DATA FILE PATHS
# ============================================================================

DATA_FILES = {
    'master_tracker': 'YEAR_3_AADB_DATA_ACCESS_MASTER_TRACKER.csv',
    'dua_tracker': 'AADB_DUA_TRACKER.csv',
    'irb_tracker': 'AADB_IRB_TRACKER.csv'
}

# ============================================================================
# CONTACT INFORMATION (from Kickoff)
# ============================================================================

PROJECT_CONTACTS = {
    'Primary Contact': {
        'name': 'Prabhjeet Singh',
        'title': 'Project Manager',
        'email': 'Prabhjeet.Singh@medstar.net'
    },
    'Secondary Contacts': [
        {
            'name': 'Sara Stienecker',
            'title': 'Associate Director',
            'email': 'Sara.L.Stienecker@medstar.net'
        },
        {
            'name': 'Stephen Fernandez',
            'title': 'Manager, Informatics Core'
        }
    ],
    'Leadership': {
        'name': 'Nawar Shara, PhD',
        'title': 'AIM-AHEAD MPI, Co-Founder AI CoLab',
        'email': 'Nawar.Shara@medstar.net'
    }
}

# ============================================================================
# ANALYSIS SETTINGS
# ============================================================================

ANALYSIS_SETTINGS = {
    'ml_test_size': 0.25,
    'ml_random_state': 42,
    'ml_cv_folds': 5,
    'min_samples_for_ml': 10,
    'visualization_dpi': 300,
    'report_date_format': '%B %d, %Y',
    'timestamp_format': '%Y%m%d_%H%M'
}

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUTPUT_DIRS = {
    'processed_data': 'processed_data',
    'models': 'models',
    'reports': 'reports',
    'visualizations': 'visualizations'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_benchmark(stage_name):
    """Get benchmark for a specific stage"""
    return STAGE_BENCHMARKS.get(stage_name, None)

def get_risk_level(risk_score):
    """Determine risk level from risk score"""
    if risk_score <= RISK_THRESHOLDS['low_risk_max']:
        return 'Low'
    elif risk_score <= RISK_THRESHOLDS['medium_risk_max']:
        return 'Medium'
    else:
        return 'High'

def is_compliant_with_nih(irb_days=None, dua_days=None):
    """Check if metrics meet NIH targets"""
    compliant = {}
    
    if irb_days is not None:
        compliant['irb'] = irb_days <= NIH_TARGETS['IRB_DAYS']
    
    if dua_days is not None:
        compliant['dua'] = dua_days <= NIH_TARGETS['DUA_DAYS']
    
    return compliant


if __name__ == "__main__":
    # Display configuration summary
    print("="*80)
    print("AADB ANALYSIS CONFIGURATION SUMMARY")
    print("="*80)
    print(f"\nCritical Path: {CRITICAL_PATH_DAYS} days")
    print(f"\nNIH Targets:")
    for key, val in NIH_TARGETS.items():
        print(f"  {key}: {val}")
    print(f"\nKey Stage Benchmarks:")
    for stage in ['intake', 'local_irb', 'dua_process', 'database_build', 'total_process_days']:
        print(f"  {stage}: {STAGE_BENCHMARKS[stage]} days")
    print(f"\nAADB Data: {AADB_DATASETS['total_data_points']} data points")
    print(f"African American Representation: {AADB_DATASETS['african_american_representation']}")
    print("\n" + "="*80)

