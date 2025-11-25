"""
AADB Main Analysis Script
Comprehensive analysis of AADB data access workflow

This script orchestrates the complete analysis including:
1. Data loading and preprocessing
2. Process flow analysis and bottleneck identification
3. Risk prediction modeling
4. Visualization generation
5. Report creation

Usage:
    python main_analysis.py
"""

import sys
import os
from datetime import datetime

# Import AADB analysis modules
from data_loader import AADBDataLoader
from process_analyzer import AADBProcessAnalyzer
from risk_predictor import AADBRiskPredictor
from visualizations import AADBVisualizer
from report_generator import AADBReportGenerator


def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def main():
    """Run complete AADB analysis"""
    
    print_header("AADB DATA ACCESS WORKFLOW ANALYSIS")
    print(f"Analysis Date: {datetime.now().strftime('%B %d, %Y at %H:%M')}")
    print("\nThis analysis addresses the following objectives:")
    print("1. Process, Analyze, and Report on Historical Data")
    print("2. Predictive & Optimization Model for At-Risk Awardees")
    print("3. Operational Optimization & Reporting Enhancements")
    print("\n" + "-"*80)
    
    # Create output directories
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    try:
        # ========================================================================
        # STEP 1: DATA LOADING AND PREPROCESSING
        # ========================================================================
        print_header("STEP 1: DATA LOADING AND PREPROCESSING")
        
        loader = AADBDataLoader()  # Auto-detects data directory
        loader.load_all_data()
        loader.preprocess_all()
        loader.get_summary_statistics()
        loader.save_processed_data(output_dir='processed_data')
        
        print("\n✓ Data loading and preprocessing complete")
        
        # ========================================================================
        # STEP 2: PROCESS FLOW ANALYSIS
        # ========================================================================
        print_header("STEP 2: PROCESS FLOW ANALYSIS AND BOTTLENECK IDENTIFICATION")
        
        analyzer = AADBProcessAnalyzer(loader)
        
        # Run comprehensive analysis
        print("\n[2.1] Analyzing Process Stages...")
        analyzer.analyze_process_stages()
        
        print("\n[2.2] Analyzing Performance by Program...")
        analyzer.analyze_by_program()
        
        print("\n[2.3] Analyzing Temporal Trends...")
        analyzer.analyze_temporal_trends()
        
        print("\n[2.4] Analyzing Risk Factors...")
        analyzer.analyze_risk_factors()
        
        print("\n[2.5] Identifying Best Practices...")
        analyzer.identify_best_practices()
        
        print("\n[2.6] Generating Summary...")
        analyzer.generate_summary_report()
        
        print("\n✓ Process analysis complete")
        
        # ========================================================================
        # STEP 3: RISK PREDICTION MODEL
        # ========================================================================
        print_header("STEP 3: PREDICTIVE RISK MODEL FOR AT-RISK AWARDEES")
        
        predictor = AADBRiskPredictor(loader)
        
        print("\n[3.1] Training Risk Prediction Model...")
        predictor.train_model()
        
        print("\n[3.2] Saving Model...")
        predictor.save_model(filepath='models/risk_predictor.pkl')
        
        # Example predictions
        print("\n[3.3] Example Risk Predictions:")
        print("-" * 80)
        
        example_projects = [
            {
                'Name': 'Example: Fast-Track Project',
                'Program': 'Research Fellowship',
                'days_noa_to_consult': 15,
                'days_intake_to_submit': 7,
                'intake_month': 3,
                'intake_quarter': 1
            },
            {
                'Name': 'Example: At-Risk Project',
                'Program': 'Leadership Fellowship',
                'days_noa_to_consult': 60,
                'days_intake_to_submit': 30,
                'intake_month': 12,
                'intake_quarter': 4
            }
        ]
        
        for project in example_projects:
            risk_result = predictor.predict_risk(project)
            print(f"\n{project['Name']}:")
            print(f"  Risk Level: {risk_result['risk_level']}")
            print(f"  Risk Probability: {risk_result['risk_probability']:.1%}")
            print(f"  Confidence: {risk_result['confidence']:.1%}")
        
        # Early warning for current projects
        if len(loader.master_tracker) > 0:
            print("\n[3.4] Generating Early Warning Dashboard...")
            # Use incomplete projects for early warning
            current_projects = loader.master_tracker[
                ~loader.master_tracker['is_completed']
            ] if 'is_completed' in loader.master_tracker.columns else loader.master_tracker
            
            if len(current_projects) > 0:
                dashboard_df = predictor.create_early_warning_dashboard(current_projects)
                if dashboard_df is not None:
                    dashboard_df.to_csv('reports/early_warning_dashboard.csv', index=False)
                    print("✓ Early warning dashboard saved to reports/early_warning_dashboard.csv")
        
        print("\n✓ Risk prediction model complete")
        
        # ========================================================================
        # STEP 4: VISUALIZATION GENERATION
        # ========================================================================
        print_header("STEP 4: VISUALIZATION GENERATION")
        
        visualizer = AADBVisualizer(analyzer, predictor)
        
        print("\n[4.1] Creating Executive Dashboard...")
        try:
            visualizer.create_executive_dashboard(
                save_path='visualizations/executive_dashboard.png'
            )
            print("✓ Executive dashboard created")
        except Exception as e:
            print(f"⚠ Warning: Could not create executive dashboard: {e}")
        
        print("\n[4.2] Creating Process Flow Diagram...")
        try:
            visualizer.create_process_flow_diagram(
                save_path='visualizations/process_flow_diagram.png'
            )
            print("✓ Process flow diagram created")
        except Exception as e:
            print(f"⚠ Warning: Could not create process flow diagram: {e}")
        
        print("\n✓ Visualization generation complete")
        
        # ========================================================================
        # STEP 5: REPORT GENERATION
        # ========================================================================
        print_header("STEP 5: COMPREHENSIVE REPORT GENERATION")
        
        reporter = AADBReportGenerator(analyzer, predictor)
        
        print("\n[5.1] Generating Reports...")
        report_paths = reporter.save_reports(output_dir='reports')
        
        print("\n[5.2] Report Summary:")
        for report_type, path in report_paths.items():
            print(f"  • {report_type}: {path}")
        
        print("\n✓ Report generation complete")
        
        # ========================================================================
        # FINAL SUMMARY
        # ========================================================================
        print_header("ANALYSIS COMPLETE")
        
        print("Summary of Deliverables:")
        print("\n1. HISTORICAL DATA ANALYSIS")
        print("   ✓ Process flow analysis with bottleneck identification")
        print("   ✓ Performance benchmarking against NIH targets")
        print("   ✓ Trend analysis and risk factor identification")
        print("   ✓ Visual process maps highlighting inefficiencies")
        
        print("\n2. PREDICTIVE MODEL")
        print("   ✓ Risk assessment model trained and validated")
        print("   ✓ Early warning system for at-risk awardees")
        print("   ✓ Model saved for future predictions")
        
        print("\n3. OPERATIONAL OPTIMIZATION")
        print("   ✓ Data-driven recommendations for process improvement")
        print("   ✓ Enhanced reporting for NIH monthly meetings")
        print("   ✓ Executive dashboards and comprehensive analytics")
        
        print("\nOutput Files:")
        print("  • Processed Data: processed_data/")
        print("  • Risk Model: models/risk_predictor.pkl")
        print("  • Visualizations: visualizations/")
        print("  • Reports: reports/")
        
        print("\n" + "="*80)
        print("Next Steps:")
        print("  1. Review executive summary in reports/")
        print("  2. Examine visualizations in visualizations/")
        print("  3. Use risk model to predict new project risks")
        print("  4. Implement top recommendations from analysis")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

