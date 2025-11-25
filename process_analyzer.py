"""
AADB Process Analyzer
Comprehensive analysis of AADB data access workflow including:
- Process flow analysis
- Bottleneck identification  
- Performance benchmarking
- Trend analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AADBProcessAnalyzer:
    """Analyzes AADB process flows and identifies bottlenecks"""
    
    def __init__(self, data_loader):
        """
        Initialize with preprocessed data from AADBDataLoader
        
        Args:
            data_loader: AADBDataLoader instance with loaded data
        """
        self.loader = data_loader
        self.master_data = data_loader.master_tracker
        self.dua_data = data_loader.dua_tracker
        self.irb_data = data_loader.irb_tracker
        self.benchmarks = data_loader.BENCHMARKS
        
        self.analysis_results = {}
    
    def analyze_process_stages(self):
        """Analyze each stage of the data access process"""
        print("\n" + "="*70)
        print("PROCESS STAGE ANALYSIS")
        print("="*70)
        
        if self.master_data is None:
            print("No master data available")
            return None
        
        df = self.master_data
        
        # Define stages with their benchmarks
        stages = [
            ('days_noa_to_consult', 'NOA to Initial Consult', self.benchmarks['noa_to_consult']),
            ('days_intake_to_submit', 'Intake to Request Submission', self.benchmarks['data_request_submission']),
            ('days_submit_to_finalize', 'Submission to Finalization', self.benchmarks['data_request_finalization']),
            ('days_irb', 'IRB Process', self.benchmarks['irb_days']),
            ('days_dua', 'DUA Process', self.benchmarks['dua_days']),
            ('days_total_process', 'Total Process (NOA to Access)', 100)
        ]
        
        stage_stats = []
        
        print("\nSTAGE-BY-STAGE PERFORMANCE:")
        print("-" * 70)
        
        for col, stage_name, benchmark in stages:
            if col in df.columns:
                values = df[col].dropna()
                
                if len(values) > 0:
                    mean_val = values.mean()
                    median_val = values.median()
                    std_val = values.std()
                    min_val = values.min()
                    max_val = values.max()
                    
                    # Calculate benchmark performance
                    exceeds = (values > benchmark).sum()
                    exceeds_pct = (exceeds / len(values)) * 100
                    
                    # Calculate severity score (how much worse than benchmark)
                    severity = mean_val / benchmark if benchmark > 0 else 1
                    
                    stage_stats.append({
                        'stage': stage_name,
                        'column': col,
                        'count': len(values),
                        'mean': mean_val,
                        'median': median_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val,
                        'benchmark': benchmark,
                        'exceeds_count': exceeds,
                        'exceeds_pct': exceeds_pct,
                        'severity': severity
                    })
                    
                    # Print results
                    status = "✓" if exceeds_pct < 20 else "⚠" if exceeds_pct < 50 else "✗"
                    print(f"\n{status} {stage_name}")
                    print(f"   Benchmark: {benchmark} days")
                    print(f"   Average: {mean_val:.1f} days (Median: {median_val:.1f})")
                    print(f"   Range: {min_val:.0f} - {max_val:.0f} days")
                    print(f"   Exceeding Benchmark: {exceeds}/{len(values)} ({exceeds_pct:.1f}%)")
                    
                    if severity > 1.5:
                        print(f"   ⚠ CRITICAL: {severity:.1f}x benchmark performance")
        
        # Store results
        self.analysis_results['stage_stats'] = pd.DataFrame(stage_stats)
        
        # Identify top bottlenecks
        print("\n" + "-" * 70)
        print("TOP 3 BOTTLENECKS (by severity):")
        print("-" * 70)
        
        top_bottlenecks = sorted(stage_stats, key=lambda x: x['severity'], reverse=True)[:3]
        for i, bottleneck in enumerate(top_bottlenecks, 1):
            print(f"{i}. {bottleneck['stage']}")
            print(f"   Severity: {bottleneck['severity']:.2f}x benchmark")
            print(f"   Average Duration: {bottleneck['mean']:.1f} days (Target: {bottleneck['benchmark']})")
            print(f"   Impact: {bottleneck['exceeds_count']} projects ({bottleneck['exceeds_pct']:.1f}%) exceeded target")
        
        return self.analysis_results['stage_stats']
    
    def analyze_by_program(self):
        """Analyze performance by program type"""
        print("\n" + "="*70)
        print("PERFORMANCE BY PROGRAM")
        print("="*70)
        
        if self.master_data is None or 'Program' not in self.master_data.columns:
            print("No program data available")
            return None
        
        df = self.master_data
        
        # Aggregate by program
        program_stats = df.groupby('Program').agg({
            'days_total_process': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'days_irb': 'mean',
            'days_dua': 'mean',
            'irb_on_target': 'mean',
            'dua_on_target': 'mean',
            'meets_all_targets': 'mean',
            'is_completed': 'sum'
        }).round(1)
        
        print("\nPROGRAM PERFORMANCE SUMMARY:")
        print("-" * 70)
        
        for program in program_stats.index:
            count = int(program_stats.loc[program, ('days_total_process', 'count')])
            avg_days = program_stats.loc[program, ('days_total_process', 'mean')]
            median_days = program_stats.loc[program, ('days_total_process', 'median')]
            completed = int(program_stats.loc[program, ('is_completed', 'sum')])
            
            irb_compliance = program_stats.loc[program, ('irb_on_target', 'mean')] * 100
            dua_compliance = program_stats.loc[program, ('dua_on_target', 'mean')] * 100
            overall_compliance = program_stats.loc[program, ('meets_all_targets', 'mean')] * 100
            
            print(f"\n{program}")
            print(f"  Projects: {count} (Completed: {completed})")
            print(f"  Avg Duration: {avg_days:.1f} days (Median: {median_days:.1f})")
            print(f"  IRB On-Target: {irb_compliance:.1f}%")
            print(f"  DUA On-Target: {dua_compliance:.1f}%")
            print(f"  Overall Compliance: {overall_compliance:.1f}%")
            
            # Performance rating
            if overall_compliance >= 70:
                rating = "✓ GOOD"
            elif overall_compliance >= 40:
                rating = "⚠ MODERATE"
            else:
                rating = "✗ NEEDS IMPROVEMENT"
            print(f"  Performance: {rating}")
        
        self.analysis_results['program_stats'] = program_stats
        return program_stats
    
    def analyze_temporal_trends(self):
        """Analyze trends over time"""
        print("\n" + "="*70)
        print("TEMPORAL TRENDS ANALYSIS")
        print("="*70)
        
        if self.master_data is None:
            print("No master data available")
            return None
        
        df = self.master_data
        
        # Quarterly trends
        if 'intake_quarter' in df.columns and 'intake_year' in df.columns:
            df['year_quarter'] = df['intake_year'].astype(str) + '-Q' + df['intake_quarter'].astype(str)
            
            quarterly_stats = df.groupby('year_quarter').agg({
                'days_total_process': ['count', 'mean', 'median'],
                'irb_on_target': 'mean',
                'dua_on_target': 'mean'
            }).round(1)
            
            print("\nQUARTERLY PERFORMANCE TRENDS:")
            print("-" * 70)
            print(quarterly_stats.to_string())
            
            self.analysis_results['quarterly_trends'] = quarterly_stats
        
        # Monthly trends
        if 'intake_month' in df.columns:
            monthly_avg = df.groupby('intake_month')['days_total_process'].agg(['mean', 'count']).round(1)
            
            print("\n\nMONTHLY SEASONALITY:")
            print("-" * 70)
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for month in monthly_avg.index:
                if not np.isnan(month):
                    month_idx = int(month) - 1
                    if 0 <= month_idx < 12:
                        avg_days = monthly_avg.loc[month, 'mean']
                        count = int(monthly_avg.loc[month, 'count'])
                        print(f"{month_names[month_idx]:>3}: {avg_days:6.1f} days avg ({count} projects)")
            
            self.analysis_results['monthly_trends'] = monthly_avg
        
        return self.analysis_results
    
    def analyze_risk_factors(self):
        """Identify risk factors for delays"""
        print("\n" + "="*70)
        print("RISK FACTOR ANALYSIS")
        print("="*70)
        
        if self.master_data is None:
            print("No master data available")
            return None
        
        df = self.master_data.copy()
        
        # Calculate risk score for each project
        risk_scores = []
        risk_details = []
        
        for idx, row in df.iterrows():
            score = 0
            factors = []
            
            # Early warning indicators (1 point each)
            if pd.notna(row.get('days_noa_to_consult')) and row['days_noa_to_consult'] > 30:
                score += 1
                factors.append(f"Late initial consult ({row['days_noa_to_consult']:.0f}d)")
            
            if pd.notna(row.get('days_intake_to_submit')) and row['days_intake_to_submit'] > 20:
                score += 1
                factors.append(f"Late request submission ({row['days_intake_to_submit']:.0f}d)")
            
            # Major delay indicators (2 points each)
            if pd.notna(row.get('days_irb')) and row['days_irb'] > self.benchmarks['irb_days']:
                score += 2
                factors.append(f"IRB delay ({row['days_irb']:.0f}d)")
            
            if pd.notna(row.get('days_dua')) and row['days_dua'] > self.benchmarks['dua_days']:
                score += 2
                factors.append(f"DUA delay ({row['days_dua']:.0f}d)")
            
            # Critical delay indicator (3 points)
            if pd.notna(row.get('days_total_process')) and row['days_total_process'] > 180:
                score += 3
                factors.append(f"Critical total delay ({row['days_total_process']:.0f}d)")
            
            risk_scores.append(score)
            risk_details.append('; '.join(factors) if factors else 'None')
        
        df['risk_score'] = risk_scores
        df['risk_factors'] = risk_details
        df['risk_level'] = pd.cut(df['risk_score'], 
                                   bins=[-1, 0, 2, 4, 100],
                                   labels=['None', 'Low', 'Medium', 'High'])
        
        # Risk distribution
        risk_dist = df['risk_level'].value_counts().sort_index()
        
        print("\nRISK SCORE DISTRIBUTION:")
        print("-" * 70)
        for level in ['None', 'Low', 'Medium', 'High']:
            if level in risk_dist.index:
                count = risk_dist[level]
                pct = (count / len(df)) * 100
                print(f"{level:>6}: {count:3d} projects ({pct:5.1f}%)")
        
        # High-risk projects
        high_risk = df[df['risk_level'] == 'High'][['Name', 'Program', 'risk_score', 'risk_factors']]
        
        if len(high_risk) > 0:
            print("\n\nHIGH-RISK PROJECTS REQUIRING INTERVENTION:")
            print("-" * 70)
            for idx, row in high_risk.iterrows():
                print(f"\n• {row['Name']} ({row['Program']})")
                print(f"  Risk Score: {row['risk_score']}")
                print(f"  Factors: {row['risk_factors']}")
        else:
            print("\n✓ No high-risk projects identified")
        
        self.master_data = df
        self.analysis_results['risk_distribution'] = risk_dist
        self.analysis_results['high_risk_projects'] = high_risk
        
        return df[['Name', 'Program', 'risk_score', 'risk_level', 'risk_factors']]
    
    def identify_best_practices(self):
        """Identify characteristics of successful projects"""
        print("\n" + "="*70)
        print("BEST PRACTICES ANALYSIS")
        print("="*70)
        
        if self.master_data is None:
            print("No master data available")
            return None
        
        df = self.master_data
        
        # Identify fast-track projects (completed in less than benchmark)
        fast_track = df[df['days_total_process'] <= 100].copy()
        
        if len(fast_track) > 0:
            print(f"\nFAST-TRACK PROJECTS (≤100 days): {len(fast_track)}")
            print("-" * 70)
            
            # Analyze common characteristics
            print("\nCommon characteristics of fast-track projects:")
            
            # Program distribution
            if 'Program' in fast_track.columns:
                program_dist = fast_track['Program'].value_counts()
                print("\n  Programs:")
                for program, count in program_dist.items():
                    pct = (count / len(fast_track)) * 100
                    print(f"    • {program}: {count} ({pct:.1f}%)")
            
            # Average stage durations
            print("\n  Average Stage Durations:")
            stage_cols = [
                ('days_noa_to_consult', 'NOA to Consult'),
                ('days_intake_to_submit', 'Intake to Submit'),
                ('days_irb', 'IRB Process'),
                ('days_dua', 'DUA Process')
            ]
            
            for col, name in stage_cols:
                if col in fast_track.columns:
                    avg = fast_track[col].mean()
                    print(f"    • {name}: {avg:.1f} days")
        
        # Compare fast vs slow projects
        slow_track = df[df['days_total_process'] > 100].copy()
        
        if len(fast_track) > 0 and len(slow_track) > 0:
            print(f"\n\nCOMPARISON: Fast (≤100d) vs Slow (>100d) Projects")
            print("-" * 70)
            
            for col, name in stage_cols:
                if col in df.columns:
                    fast_avg = fast_track[col].mean()
                    slow_avg = slow_track[col].mean()
                    diff = slow_avg - fast_avg
                    
                    print(f"{name:20} | Fast: {fast_avg:5.1f}d | Slow: {slow_avg:5.1f}d | Diff: +{diff:.1f}d")
        
        self.analysis_results['fast_track'] = fast_track
        self.analysis_results['slow_track'] = slow_track
        
        return fast_track
    
    def generate_summary_report(self):
        """Generate executive summary of key findings"""
        print("\n" + "="*70)
        print("EXECUTIVE SUMMARY - KEY FINDINGS")
        print("="*70)
        
        if self.master_data is None:
            print("No data available for summary")
            return None
        
        df = self.master_data
        
        # Overall metrics
        total_projects = len(df)
        completed_projects = df['is_completed'].sum()
        avg_duration = df['days_total_process'].mean()
        median_duration = df['days_total_process'].median()
        
        print(f"\nOVERALL METRICS:")
        print(f"  Total Projects: {total_projects}")
        print(f"  Completed: {completed_projects}")
        print(f"  Average Duration: {avg_duration:.1f} days")
        print(f"  Median Duration: {median_duration:.1f} days")
        
        # NIH target compliance
        irb_compliance = (df['irb_on_target'].sum() / df['irb_on_target'].count()) * 100 if df['irb_on_target'].count() > 0 else 0
        dua_compliance = (df['dua_on_target'].sum() / df['dua_on_target'].count()) * 100 if df['dua_on_target'].count() > 0 else 0
        overall_compliance = (df['meets_all_targets'].sum() / len(df)) * 100
        
        print(f"\nNIH TARGET COMPLIANCE:")
        print(f"  IRB (≤60 days): {irb_compliance:.1f}%")
        print(f"  DUA (≤90 days): {dua_compliance:.1f}%")
        print(f"  Overall: {overall_compliance:.1f}%")
        
        # Key insights
        print(f"\nKEY INSIGHTS:")
        
        if 'stage_stats' in self.analysis_results:
            top_bottleneck = self.analysis_results['stage_stats'].nlargest(1, 'severity')
            if len(top_bottleneck) > 0:
                bottleneck_name = top_bottleneck.iloc[0]['stage']
                bottleneck_severity = top_bottleneck.iloc[0]['severity']
                print(f"  1. Primary Bottleneck: {bottleneck_name} ({bottleneck_severity:.1f}x benchmark)")
        
        if 'risk_distribution' in self.analysis_results:
            high_risk_count = self.analysis_results['risk_distribution'].get('High', 0)
            high_risk_pct = (high_risk_count / total_projects) * 100
            print(f"  2. High-Risk Projects: {high_risk_count} ({high_risk_pct:.1f}%)")
        
        projects_over_100 = (df['days_total_process'] > 100).sum()
        print(f"  3. Projects Exceeding 100 Days: {projects_over_100} ({projects_over_100/total_projects*100:.1f}%)")
        
        # Recommendations
        print(f"\nTOP 3 RECOMMENDATIONS:")
        
        recommendations = []
        
        # Based on bottlenecks
        if 'stage_stats' in self.analysis_results:
            top_3_bottlenecks = self.analysis_results['stage_stats'].nlargest(3, 'severity')
            for i, row in top_3_bottlenecks.iterrows():
                if row['severity'] > 1.3:
                    recommendations.append(
                        f"Address {row['stage']} delays ({row['severity']:.1f}x target)"
                    )
        
        # Based on compliance
        if irb_compliance < 60:
            recommendations.append("URGENT: Implement IRB fast-track process")
        if dua_compliance < 60:
            recommendations.append("URGENT: Streamline DUA approval workflow")
        
        # Ensure we have at least 3 recommendations
        if len(recommendations) < 3:
            recommendations.append("Implement early warning system for at-risk projects")
            recommendations.append("Standardize onboarding procedures across programs")
        
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec}")
        
        return self.analysis_results


if __name__ == "__main__":
    from data_loader import AADBDataLoader
    
    # Load and preprocess data
    loader = AADBDataLoader()
    loader.load_all_data().preprocess_all()
    
    # Run analysis
    analyzer = AADBProcessAnalyzer(loader)
    analyzer.analyze_process_stages()
    analyzer.analyze_by_program()
    analyzer.analyze_temporal_trends()
    analyzer.analyze_risk_factors()
    analyzer.identify_best_practices()
    analyzer.generate_summary_report()

