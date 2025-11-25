"""
AADB Data Loader and Preprocessor
Loads and cleans all AADB tracker datasets for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime


class AADBDataLoader:
    """Loads and preprocesses AADB tracking data"""
    
    def __init__(self, data_dir=None):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing the CSV files (default: auto-detect from data/ or ../)
        """
        import os
        
        # Auto-detect data directory
        if data_dir is None:
            if os.path.exists('data'):
                self.data_dir = 'data'
            elif os.path.exists('../data'):
                self.data_dir = '../data'
            else:
                self.data_dir = '../'
        else:
            self.data_dir = data_dir
        self.master_tracker = None
        self.dua_tracker = None
        self.irb_tracker = None
        
        # NIH Performance Benchmarks
        self.BENCHMARKS = {
            'total_process_days': 78,
            'irb_days': 60,
            'dua_days': 90,
            'intake_turnaround': 7,
            'pe_turnaround': 30,
            'pe_to_fe_turnaround': 20,
            'noa_to_consult': 30,
            'data_request_submission': 14,
            'data_request_finalization': 30
        }
    
    def load_all_data(self):
        """Load all three tracker datasets"""
        print("Loading AADB Tracker Data...")
        
        # Load Master Tracker
        try:
            self.master_tracker = pd.read_csv(f'{self.data_dir}/YEAR_3_AADB_DATA_ACCESS_MASTER_TRACKER.csv')
            # Clean column names (remove newlines)
            self.master_tracker.columns = [col.strip().replace('\n', ' ') for col in self.master_tracker.columns]
            print(f"✓ Master Tracker loaded: {len(self.master_tracker)} records")
        except Exception as e:
            print(f"✗ Error loading Master Tracker: {e}")
        
        # Load DUA Tracker
        try:
            self.dua_tracker = pd.read_csv(f'{self.data_dir}/AADB_DUA_TRACKER.csv')
            print(f"✓ DUA Tracker loaded: {len(self.dua_tracker)} records")
        except Exception as e:
            print(f"✗ Error loading DUA Tracker: {e}")
        
        # Load IRB Tracker
        try:
            self.irb_tracker = pd.read_csv(f'{self.data_dir}/AADB_IRB_TRACKER.csv')
            print(f"✓ IRB Tracker loaded: {len(self.irb_tracker)} records")
        except Exception as e:
            print(f"✗ Error loading IRB Tracker: {e}")
        
        return self
    
    def preprocess_master_tracker(self):
        """Preprocess and clean the master tracker data"""
        if self.master_tracker is None:
            print("Master tracker not loaded")
            return self
        
        df = self.master_tracker.copy()
        
        # Convert date columns
        date_cols = ['NOA', 'Intake Consult', 'Data Request Submitted', 
                    'Data Consult (if any)', 'Data Request Finalized',
                    'IRB Submitted*', 'IRB Determination shared with AADB',
                    'DUA information shared', 'DUA Draft Sent', 'FE DUA',
                    'Data Access Provided', 'Data Access Confirmed']
        
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Extract and clean numeric metrics
        numeric_cols = {
            '# of days to consult from NOA': 'days_noa_to_consult',
            '# of days to submit request from Intake': 'days_intake_to_submit',
            '# of days to finalize request from submission': 'days_submit_to_finalize',
            '# of days for IRB (NIH Target=60)': 'days_irb',
            '# of days to share DUA information from Intake': 'days_intake_to_dua_info',
            '# of days for DUA (NIH Target=90)': 'days_dua',
            '# of days to Data Access from NOA': 'days_total_process',
        }
        
        for orig_col, new_col in numeric_cols.items():
            if orig_col in df.columns:
                df[new_col] = pd.to_numeric(df[orig_col], errors='coerce')
        
        # Create performance flags
        df['irb_on_target'] = df['days_irb'] <= self.BENCHMARKS['irb_days']
        df['dua_on_target'] = df['days_dua'] <= self.BENCHMARKS['dua_days']
        df['total_on_target'] = df['days_total_process'] <= 100  # Overall target: 100 days from NOA
        
        # Create overall performance flag
        df['meets_all_targets'] = (
            df['irb_on_target'].fillna(False) & 
            df['dua_on_target'].fillna(False) & 
            df['total_on_target'].fillna(False)
        )
        
        # Extract temporal features
        df['intake_month'] = df['Intake Consult'].dt.month
        df['intake_year'] = df['Intake Consult'].dt.year
        df['intake_quarter'] = df['Intake Consult'].dt.quarter
        df['intake_day_of_week'] = df['Intake Consult'].dt.dayofweek
        
        # Calculate completion status
        df['is_completed'] = df['Data Access Confirmed'].notna()
        df['is_data_provided'] = df['Data Access Provided'].notna()
        
        self.master_tracker = df
        print(f"✓ Master tracker preprocessed: {len(df)} records")
        
        return self
    
    def preprocess_dua_tracker(self):
        """Preprocess and clean the DUA tracker data"""
        if self.dua_tracker is None:
            print("DUA tracker not loaded")
            return self
        
        df = self.dua_tracker.copy()
        
        # Convert date columns
        date_cols = ['OCGM intake', 'OCGM sent request', 'Recipient Acknowledgment',
                    'PE', 'FE', 'DUA Expiration Date']
        
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Extract numeric turnaround times
        turnaround_cols = ['OCGM intake turnaround', 'Recipient PE turnaround',
                          'OCGM PE to FE turnaround', 'Intake to FE (days)']
        
        for col in turnaround_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create delay flags
        df['intake_delayed'] = df['OCGM intake turnaround'] > self.BENCHMARKS['intake_turnaround']
        df['pe_delayed'] = df['Recipient PE turnaround'] > self.BENCHMARKS['pe_turnaround']
        df['fe_delayed'] = df['OCGM PE to FE turnaround'] > self.BENCHMARKS['pe_to_fe_turnaround']
        df['total_delayed'] = df['Intake to FE (days)'] > self.BENCHMARKS['total_process_days']
        
        # Create overall delay flag
        df['any_delay'] = (
            df['intake_delayed'].fillna(False) |
            df['pe_delayed'].fillna(False) |
            df['fe_delayed'].fillna(False) |
            df['total_delayed'].fillna(False)
        )
        
        # Add features
        df['has_acknowledgment'] = df['Recipient Acknowledgment'].notna()
        
        # Extract temporal features
        df['intake_month'] = df['OCGM intake'].dt.month
        df['intake_year'] = df['OCGM intake'].dt.year
        df['intake_quarter'] = df['OCGM intake'].dt.quarter
        
        self.dua_tracker = df
        print(f"✓ DUA tracker preprocessed: {len(df)} records")
        
        return self
    
    def preprocess_irb_tracker(self):
        """Preprocess and clean the IRB tracker data"""
        if self.irb_tracker is None:
            print("IRB tracker not loaded")
            return self
        
        df = self.irb_tracker.copy()
        
        # Convert date columns
        date_cols = ['Initial 1:1 consult', 'Submit IRB', 'Local Determination',
                    'AADB Receives Letter']
        
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate turnaround times
        df['days_consult_to_submit'] = (df['Submit IRB'] - df['Initial 1:1 consult']).dt.days
        df['days_submit_to_determination'] = (df['Local Determination'] - df['Submit IRB']).dt.days
        df['days_determination_to_letter'] = (df['AADB Receives Letter'] - df['Local Determination']).dt.days
        df['days_total_irb'] = (df['AADB Receives Letter'] - df['Initial 1:1 consult']).dt.days
        
        # Create status flags
        df['has_letter'] = df['IRB Status'] == 'Letter Received'
        df['is_submitted'] = df['IRB Status'] == 'Submitted'
        df['is_exempt'] = df['IRB Determination'] == 'Exempt'
        df['is_nhsr'] = df['IRB Determination'] == 'NHSR'
        
        # IRB delay flag
        df['irb_delayed'] = df['days_total_irb'] > self.BENCHMARKS['irb_days']
        
        self.irb_tracker = df
        print(f"✓ IRB tracker preprocessed: {len(df)} records")
        
        return self
    
    def preprocess_all(self):
        """Preprocess all loaded datasets"""
        self.preprocess_master_tracker()
        self.preprocess_dua_tracker()
        self.preprocess_irb_tracker()
        return self
    
    def get_summary_statistics(self):
        """Get summary statistics for all datasets"""
        print("\n" + "="*60)
        print("AADB DATA SUMMARY STATISTICS")
        print("="*60)
        
        if self.master_tracker is not None:
            print("\nMASTER TRACKER:")
            print(f"  Total Projects: {len(self.master_tracker)}")
            print(f"  Completed Projects: {self.master_tracker['is_completed'].sum()}")
            print(f"  Average Total Process Time: {self.master_tracker['days_total_process'].mean():.1f} days")
            print(f"  Projects Meeting All Targets: {self.master_tracker['meets_all_targets'].sum()}")
            
        if self.dua_tracker is not None:
            print("\nDUA TRACKER:")
            print(f"  Total Records: {len(self.dua_tracker)}")
            print(f"  Average DUA Time: {self.dua_tracker['Intake to FE (days)'].mean():.1f} days")
            print(f"  Records with Delays: {self.dua_tracker['any_delay'].sum()}")
            
        if self.irb_tracker is not None:
            print("\nIRB TRACKER:")
            print(f"  Total Records: {len(self.irb_tracker)}")
            print(f"  Letters Received: {self.irb_tracker['has_letter'].sum()}")
            print(f"  Exempt Determinations: {self.irb_tracker['is_exempt'].sum()}")
        
        print("="*60)
    
    def save_processed_data(self, output_dir='../aadb_analysis/processed_data'):
        """Save preprocessed data for future use"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.master_tracker is not None:
            self.master_tracker.to_csv(f'{output_dir}/master_tracker_processed.csv', index=False)
            print(f"✓ Saved processed master tracker")
        
        if self.dua_tracker is not None:
            self.dua_tracker.to_csv(f'{output_dir}/dua_tracker_processed.csv', index=False)
            print(f"✓ Saved processed DUA tracker")
        
        if self.irb_tracker is not None:
            self.irb_tracker.to_csv(f'{output_dir}/irb_tracker_processed.csv', index=False)
            print(f"✓ Saved processed IRB tracker")


if __name__ == "__main__":
    # Example usage
    loader = AADBDataLoader()
    loader.load_all_data()
    loader.preprocess_all()
    loader.get_summary_statistics()
    loader.save_processed_data()

