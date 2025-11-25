"""
AADB Report Generator
Generates comprehensive reports for NIH and stakeholders including:
- Executive summaries
- Detailed analytics
- Actionable recommendations
- Export to various formats (Excel, PDF-ready)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AADBReportGenerator:
    """Generates comprehensive reports for AADB analysis"""
    
    def __init__(self, analyzer, predictor=None):
        """
        Initialize report generator
        
        Args:
            analyzer: AADBProcessAnalyzer instance
            predictor: AADBRiskPredictor instance (optional)
        """
        self.analyzer = analyzer
        self.predictor = predictor
        self.master_data = analyzer.master_data
        self.benchmarks = analyzer.benchmarks
        self.report_date = datetime.now()
    
    def generate_executive_summary(self):
        """Generate executive summary text"""
        lines = []
        lines.append("="*80)
        lines.append("AADB DATA ACCESS PROCESS - EXECUTIVE SUMMARY")
        lines.append(f"Report Date: {self.report_date.strftime('%B %d, %Y')}")
        lines.append("="*80)
        lines.append("")
        
        # Overall Performance
        lines.append("1. OVERALL PERFORMANCE")
        lines.append("-" * 80)
        
        total = len(self.master_data)
        completed = self.master_data['is_completed'].sum() if 'is_completed' in self.master_data.columns else 0
        
        lines.append(f"   Total Projects Tracked: {total}")
        lines.append(f"   Completed Projects: {completed}")
        
        if 'days_total_process' in self.master_data.columns:
            avg_days = self.master_data['days_total_process'].mean()
            median_days = self.master_data['days_total_process'].median()
            lines.append(f"   Average Process Duration: {avg_days:.1f} days")
            lines.append(f"   Median Process Duration: {median_days:.1f} days")
        
        lines.append("")
        
        # NIH Compliance
        lines.append("2. NIH TARGET COMPLIANCE")
        lines.append("-" * 80)
        
        if 'irb_on_target' in self.master_data.columns:
            irb_compliance = (self.master_data['irb_on_target'].sum() / self.master_data['irb_on_target'].count() * 100) if self.master_data['irb_on_target'].count() > 0 else 0
            status = "✓ ON TARGET" if irb_compliance >= 80 else "⚠ NEEDS IMPROVEMENT"
            lines.append(f"   IRB Process (Target: ≤60 days): {irb_compliance:.1f}% {status}")
        
        if 'dua_on_target' in self.master_data.columns:
            dua_compliance = (self.master_data['dua_on_target'].sum() / self.master_data['dua_on_target'].count() * 100) if self.master_data['dua_on_target'].count() > 0 else 0
            status = "✓ ON TARGET" if dua_compliance >= 80 else "⚠ NEEDS IMPROVEMENT"
            lines.append(f"   DUA Process (Target: ≤90 days): {dua_compliance:.1f}% {status}")
        
        lines.append("")
        
        # Critical Issues
        lines.append("3. CRITICAL ISSUES IDENTIFIED")
        lines.append("-" * 80)
        
        issues = self._identify_critical_issues()
        if issues:
            for i, issue in enumerate(issues, 1):
                lines.append(f"   {i}. {issue}")
        else:
            lines.append("   No critical issues identified")
        
        lines.append("")
        
        # Top Recommendations
        lines.append("4. TOP RECOMMENDATIONS")
        lines.append("-" * 80)
        
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations[:5], 1):
            lines.append(f"   {i}. {rec}")
        
        lines.append("")
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def _identify_critical_issues(self):
        """Identify critical issues requiring immediate attention"""
        issues = []
        
        # Check compliance rates
        if 'irb_on_target' in self.master_data.columns:
            irb_compliance = (self.master_data['irb_on_target'].sum() / self.master_data['irb_on_target'].count() * 100) if self.master_data['irb_on_target'].count() > 0 else 0
            if irb_compliance < 50:
                issues.append(f"CRITICAL: IRB compliance at {irb_compliance:.1f}% (Target: 80%)")
        
        if 'dua_on_target' in self.master_data.columns:
            dua_compliance = (self.master_data['dua_on_target'].sum() / self.master_data['dua_on_target'].count() * 100) if self.master_data['dua_on_target'].count() > 0 else 0
            if dua_compliance < 50:
                issues.append(f"CRITICAL: DUA compliance at {dua_compliance:.1f}% (Target: 80%)")
        
        # Check for extreme delays
        if 'days_total_process' in self.master_data.columns:
            extreme_delays = (self.master_data['days_total_process'] > 180).sum()
            if extreme_delays > 0:
                pct = (extreme_delays / len(self.master_data)) * 100
                issues.append(f"WARNING: {extreme_delays} projects ({pct:.1f}%) exceeding 180 days")
        
        # Check bottlenecks
        if 'stage_stats' in self.analyzer.analysis_results:
            stage_stats = self.analyzer.analysis_results['stage_stats']
            critical_bottlenecks = stage_stats[stage_stats['severity'] > 1.5]
            if len(critical_bottlenecks) > 0:
                for idx, row in critical_bottlenecks.iterrows():
                    issues.append(f"BOTTLENECK: {row['stage']} at {row['severity']:.1f}x benchmark")
        
        return issues
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on bottlenecks
        if 'stage_stats' in self.analyzer.analysis_results:
            stage_stats = self.analyzer.analysis_results['stage_stats'].sort_values('severity', ascending=False)
            
            for idx, row in stage_stats.head(3).iterrows():
                if row['severity'] > 1.3:
                    if 'IRB' in row['stage']:
                        recommendations.append(
                            f"Streamline IRB process: Currently {row['mean']:.0f} days avg "
                            f"(target: {row['benchmark']} days). Consider expedited review procedures."
                        )
                    elif 'DUA' in row['stage']:
                        recommendations.append(
                            f"Optimize DUA workflow: Currently {row['mean']:.0f} days avg "
                            f"(target: {row['benchmark']} days). Implement parallel processing."
                        )
                    elif 'Consult' in row['stage']:
                        recommendations.append(
                            f"Accelerate initial consultation scheduling: Currently {row['mean']:.0f} days avg "
                            f"(target: {row['benchmark']} days). Increase staff capacity."
                        )
                    else:
                        recommendations.append(
                            f"Address {row['stage']} delays: Reduce from {row['mean']:.0f} to "
                            f"{row['benchmark']} days through process optimization."
                        )
        
        # Program-specific recommendations
        if 'program_stats' in self.analyzer.analysis_results:
            program_stats = self.analyzer.analysis_results['program_stats']
            
            # Find programs with low compliance
            for program in program_stats.index:
                overall_compliance = program_stats.loc[program, ('meets_all_targets', 'mean')] * 100
                if overall_compliance < 40:
                    recommendations.append(
                        f"Focus improvement efforts on {program}: Only {overall_compliance:.0f}% "
                        "meeting all targets. Conduct detailed process review."
                    )
        
        # Risk-based recommendations
        if 'risk_distribution' in self.analyzer.analysis_results:
            high_risk = self.analyzer.analysis_results['risk_distribution'].get('High', 0)
            total = len(self.master_data)
            if high_risk > total * 0.15:
                recommendations.append(
                    f"Implement early warning system: {high_risk} high-risk projects identified. "
                    "Proactive intervention could prevent delays."
                )
        
        # General recommendations
        if not recommendations or len(recommendations) < 3:
            recommendations.extend([
                "Standardize onboarding procedures across all programs to reduce variability",
                "Implement automated tracking and alerts for projects approaching deadlines",
                "Establish monthly review meetings with key stakeholders to address bottlenecks",
                "Create templates and checklists to reduce processing time",
                "Provide additional training for staff on compliance requirements"
            ])
        
        return recommendations[:5]
    
    def export_to_excel(self, output_path=None):
        """Export comprehensive analysis to Excel"""
        if output_path is None:
            output_path = f'../aadb_analysis/AADB_Analysis_Report_{self.report_date.strftime("%Y%m%d")}.xlsx'
        
        print(f"\nGenerating Excel report...")
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Executive Summary
                exec_data = self._prepare_executive_summary_data()
                exec_data.to_excel(writer, sheet_name='Executive_Summary', index=False)
                
                # Stage Analysis
                if 'stage_stats' in self.analyzer.analysis_results:
                    stage_df = self.analyzer.analysis_results['stage_stats'].copy()
                    stage_df = stage_df.round(2)
                    stage_df.to_excel(writer, sheet_name='Stage_Analysis', index=False)
                
                # Program Performance
                if 'program_stats' in self.analyzer.analysis_results:
                    program_df = self.analyzer.analysis_results['program_stats'].copy()
                    program_df.to_excel(writer, sheet_name='Program_Performance')
                
                # Risk Assessment
                if 'risk_score' in self.master_data.columns:
                    risk_df = self.master_data[
                        ['Name', 'Program', 'days_total_process', 'days_irb', 'days_dua', 
                         'risk_score', 'risk_level', 'risk_factors']
                    ].copy()
                    risk_df = risk_df.sort_values('risk_score', ascending=False)
                    risk_df.to_excel(writer, sheet_name='Risk_Assessment', index=False)
                
                # Detailed Project Data
                detail_cols = [col for col in self.master_data.columns 
                              if not col.startswith('intake_') and col not in ['risk_factors']]
                detail_df = self.master_data[detail_cols].copy()
                detail_df.to_excel(writer, sheet_name='Project_Details', index=False)
                
                # Recommendations
                rec_df = pd.DataFrame({
                    'Priority': range(1, len(self._generate_recommendations()) + 1),
                    'Recommendation': self._generate_recommendations()
                })
                rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
                
                print(f"✓ Excel report saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"✗ Error generating Excel report: {e}")
            return None
    
    def _prepare_executive_summary_data(self):
        """Prepare executive summary data for Excel"""
        metrics = []
        
        # Overall metrics
        metrics.append({
            'Category': 'Overall Performance',
            'Metric': 'Total Projects',
            'Value': len(self.master_data),
            'Target': 'N/A',
            'Status': 'Info'
        })
        
        if 'is_completed' in self.master_data.columns:
            completed = self.master_data['is_completed'].sum()
            metrics.append({
                'Category': 'Overall Performance',
                'Metric': 'Completed Projects',
                'Value': completed,
                'Target': 'N/A',
                'Status': 'Info'
            })
        
        if 'days_total_process' in self.master_data.columns:
            avg_days = self.master_data['days_total_process'].mean()
            metrics.append({
                'Category': 'Overall Performance',
                'Metric': 'Average Duration (days)',
                'Value': f'{avg_days:.1f}',
                'Target': '≤100',
                'Status': 'On Target' if avg_days <= 100 else 'Over Target'
            })
        
        # NIH Compliance
        if 'irb_on_target' in self.master_data.columns:
            irb_compliance = (self.master_data['irb_on_target'].sum() / self.master_data['irb_on_target'].count() * 100) if self.master_data['irb_on_target'].count() > 0 else 0
            metrics.append({
                'Category': 'NIH Compliance',
                'Metric': 'IRB Compliance Rate (%)',
                'Value': f'{irb_compliance:.1f}',
                'Target': '≥80',
                'Status': 'On Target' if irb_compliance >= 80 else 'Needs Improvement'
            })
        
        if 'dua_on_target' in self.master_data.columns:
            dua_compliance = (self.master_data['dua_on_target'].sum() / self.master_data['dua_on_target'].count() * 100) if self.master_data['dua_on_target'].count() > 0 else 0
            metrics.append({
                'Category': 'NIH Compliance',
                'Metric': 'DUA Compliance Rate (%)',
                'Value': f'{dua_compliance:.1f}',
                'Target': '≥80',
                'Status': 'On Target' if dua_compliance >= 80 else 'Needs Improvement'
            })
        
        # Risk metrics
        if 'risk_level' in self.master_data.columns:
            high_risk = (self.master_data['risk_level'] == 'High').sum()
            high_risk_pct = (high_risk / len(self.master_data)) * 100
            metrics.append({
                'Category': 'Risk Assessment',
                'Metric': 'High-Risk Projects',
                'Value': f'{high_risk} ({high_risk_pct:.1f}%)',
                'Target': '<10%',
                'Status': 'Acceptable' if high_risk_pct < 10 else 'Elevated'
            })
        
        return pd.DataFrame(metrics)
    
    def generate_nih_monthly_report(self):
        """Generate formatted report for NIH monthly meetings"""
        lines = []
        lines.append("="*80)
        lines.append("AADB DATA ACCESS - MONTHLY REPORT FOR NIH")
        lines.append(f"Reporting Period: {self.report_date.strftime('%B %Y')}")
        lines.append("="*80)
        lines.append("")
        
        # Key Performance Indicators
        lines.append("KEY PERFORMANCE INDICATORS")
        lines.append("-" * 80)
        lines.append("")
        
        # Create KPI table
        kpi_data = []
        
        if 'days_total_process' in self.master_data.columns:
            avg_total = self.master_data['days_total_process'].mean()
            kpi_data.append(['Average Total Process Time', f'{avg_total:.0f} days', '100 days', 
                           '✓' if avg_total <= 100 else '✗'])
        
        if 'days_irb' in self.master_data.columns:
            avg_irb = self.master_data['days_irb'].mean()
            irb_on_target = (self.master_data['days_irb'] <= 60).sum() / self.master_data['days_irb'].count() * 100 if self.master_data['days_irb'].count() > 0 else 0
            kpi_data.append(['IRB Process Time', f'{avg_irb:.0f} days ({irb_on_target:.0f}% on target)', 
                           '≤60 days (80% target)', '✓' if irb_on_target >= 80 else '✗'])
        
        if 'days_dua' in self.master_data.columns:
            avg_dua = self.master_data['days_dua'].mean()
            dua_on_target = (self.master_data['days_dua'] <= 90).sum() / self.master_data['days_dua'].count() * 100 if self.master_data['days_dua'].count() > 0 else 0
            kpi_data.append(['DUA Process Time', f'{avg_dua:.0f} days ({dua_on_target:.0f}% on target)', 
                           '≤90 days (80% target)', '✓' if dua_on_target >= 80 else '✗'])
        
        # Format KPI table
        if kpi_data:
            lines.append(f"{'Metric':<30} {'Current':<25} {'Target':<25} {'Status':<10}")
            lines.append("-" * 80)
            for row in kpi_data:
                lines.append(f"{row[0]:<30} {row[1]:<25} {row[2]:<25} {row[3]:<10}")
        
        lines.append("")
        lines.append("")
        
        # Progress Updates
        lines.append("PROGRESS UPDATES")
        lines.append("-" * 80)
        lines.append("")
        
        # Projects completed this period
        if 'is_completed' in self.master_data.columns:
            completed = self.master_data['is_completed'].sum()
            lines.append(f"• Projects Completed: {completed}")
        
        # New projects
        lines.append(f"• Total Active Projects: {len(self.master_data)}")
        
        # At-risk projects
        if 'risk_level' in self.master_data.columns:
            high_risk = (self.master_data['risk_level'] == 'High').sum()
            lines.append(f"• High-Risk Projects Requiring Intervention: {high_risk}")
        
        lines.append("")
        lines.append("")
        
        # Challenges and Mitigation
        lines.append("CHALLENGES AND MITIGATION STRATEGIES")
        lines.append("-" * 80)
        lines.append("")
        
        issues = self._identify_critical_issues()
        if issues:
            for i, issue in enumerate(issues, 1):
                lines.append(f"{i}. {issue}")
        else:
            lines.append("No critical challenges identified this period.")
        
        lines.append("")
        lines.append("")
        
        # Action Items
        lines.append("ACTION ITEMS FOR NEXT PERIOD")
        lines.append("-" * 80)
        lines.append("")
        
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations[:5], 1):
            lines.append(f"{i}. {rec}")
        
        lines.append("")
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def save_reports(self, output_dir='../aadb_analysis/reports'):
        """Generate and save all reports"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = self.report_date.strftime('%Y%m%d_%H%M')
        
        print("\nGenerating reports...")
        
        # Executive Summary
        exec_summary = self.generate_executive_summary()
        exec_path = f'{output_dir}/Executive_Summary_{timestamp}.txt'
        with open(exec_path, 'w') as f:
            f.write(exec_summary)
        print(f"✓ Executive summary saved to {exec_path}")
        
        # NIH Monthly Report
        nih_report = self.generate_nih_monthly_report()
        nih_path = f'{output_dir}/NIH_Monthly_Report_{timestamp}.txt'
        with open(nih_path, 'w') as f:
            f.write(nih_report)
        print(f"✓ NIH monthly report saved to {nih_path}")
        
        # Excel Report
        excel_path = self.export_to_excel(f'{output_dir}/Comprehensive_Analysis_{timestamp}.xlsx')
        
        print(f"\n✓ All reports generated successfully")
        print(f"  Output directory: {output_dir}")
        
        return {
            'executive_summary': exec_path,
            'nih_report': nih_path,
            'excel_report': excel_path
        }


if __name__ == "__main__":
    from data_loader import AADBDataLoader
    from process_analyzer import AADBProcessAnalyzer
    
    # Load and analyze data
    loader = AADBDataLoader()
    loader.load_all_data().preprocess_all()
    
    analyzer = AADBProcessAnalyzer(loader)
    analyzer.analyze_process_stages()
    analyzer.analyze_by_program()
    analyzer.analyze_risk_factors()
    
    # Generate reports
    reporter = AADBReportGenerator(analyzer)
    reporter.save_reports()

