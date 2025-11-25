"""
AADB Visualization Module
Creates comprehensive visualizations including:
- Process flow diagrams
- Performance dashboards
- Bottleneck identification charts
- Risk assessment visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import warnings
import os
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


class AADBVisualizer:
    """Creates comprehensive visualizations for AADB analysis"""
    
    def __init__(self, analyzer, predictor=None):
        """
        Initialize visualizer
        
        Args:
            analyzer: AADBProcessAnalyzer instance
            predictor: AADBRiskPredictor instance (optional)
        """
        self.analyzer = analyzer
        self.predictor = predictor
        self.master_data = analyzer.master_data
        self.benchmarks = analyzer.benchmarks
    
    def create_executive_dashboard(self, save_path=None):
        """Create comprehensive executive dashboard"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('AADB Data Access Process - Executive Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Create grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Overall Performance Metrics (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_performance_metrics(ax1)
        
        # 2. NIH Target Compliance (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_nih_compliance(ax2)
        
        # 3. Process Duration Distribution (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_duration_distribution(ax3)
        
        # 4. Stage Performance (middle center-left)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_stage_performance(ax4)
        
        # 5. Risk Distribution (middle center-right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_risk_distribution(ax5)
        
        # 6. Program Performance (middle right)
        ax6 = fig.add_subplot(gs[1, 3])
        self._plot_program_performance(ax6)
        
        # 7. Temporal Trends (bottom left)
        ax7 = fig.add_subplot(gs[2, :2])
        self._plot_temporal_trends(ax7)
        
        # 8. Bottleneck Analysis (bottom right)
        ax8 = fig.add_subplot(gs[2, 2:])
        self._plot_bottleneck_analysis(ax8)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dashboard saved to {save_path}")
        
        plt.show()
        return fig
    
    def _plot_performance_metrics(self, ax):
        """Plot key performance metrics"""
        df = self.master_data
        
        metrics = {
            'Total\nProjects': len(df),
            'Completed': df['is_completed'].sum() if 'is_completed' in df.columns else 0,
            'Avg Duration\n(days)': df['days_total_process'].mean() if 'days_total_process' in df.columns else 0,
            'Projects\n>100 days': (df['days_total_process'] > 100).sum() if 'days_total_process' in df.columns else 0
        }
        
        x_pos = np.arange(len(metrics))
        values = list(metrics.values())
        
        bars = ax.bar(x_pos, values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics.keys())
        ax.set_title('Key Performance Metrics', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_nih_compliance(self, ax):
        """Plot NIH target compliance rates"""
        df = self.master_data
        
        # Calculate compliance rates
        compliance_data = []
        
        if 'irb_on_target' in df.columns:
            irb_rate = (df['irb_on_target'].sum() / df['irb_on_target'].count() * 100) if df['irb_on_target'].count() > 0 else 0
            compliance_data.append(('IRB\n(≤60d)', irb_rate, 80))
        
        if 'dua_on_target' in df.columns:
            dua_rate = (df['dua_on_target'].sum() / df['dua_on_target'].count() * 100) if df['dua_on_target'].count() > 0 else 0
            compliance_data.append(('DUA\n(≤90d)', dua_rate, 80))
        
        if 'total_on_target' in df.columns:
            total_rate = (df['total_on_target'].sum() / df['total_on_target'].count() * 100) if df['total_on_target'].count() > 0 else 0
            compliance_data.append(('Total\n(≤100d)', total_rate, 80))
        
        if compliance_data:
            categories = [x[0] for x in compliance_data]
            rates = [x[1] for x in compliance_data]
            targets = [x[2] for x in compliance_data]
            
            x_pos = np.arange(len(categories))
            colors = ['#e74c3c' if r < 50 else '#f39c12' if r < 80 else '#2ecc71' for r in rates]
            
            bars = ax.bar(x_pos, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.axhline(y=80, color='green', linestyle='--', linewidth=2, label='Target (80%)', alpha=0.7)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories)
            ax.set_ylabel('Compliance Rate (%)')
            ax.set_title('NIH Target Compliance', fontweight='bold', fontsize=12)
            ax.set_ylim(0, 110)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def _plot_duration_distribution(self, ax):
        """Plot distribution of process durations"""
        if 'days_total_process' in self.master_data.columns:
            data = self.master_data['days_total_process'].dropna()
            
            ax.hist(data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.0f}d')
            ax.axvline(data.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {data.median():.0f}d')
            ax.axvline(100, color='green', linestyle='--', linewidth=2, label='Target: 100d', alpha=0.7)
            
            ax.set_xlabel('Total Process Days')
            ax.set_ylabel('Number of Projects')
            ax.set_title('Process Duration Distribution', fontweight='bold', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
    
    def _plot_stage_performance(self, ax):
        """Plot performance by stage"""
        stages = [
            ('days_noa_to_consult', 'NOA→Consult', 30),
            ('days_intake_to_submit', 'Intake→Submit', 14),
            ('days_irb', 'IRB', 60),
            ('days_dua', 'DUA', 90)
        ]
        
        stage_names = []
        avg_days = []
        colors = []
        
        for col, name, benchmark in stages:
            if col in self.master_data.columns:
                mean_val = self.master_data[col].mean()
                stage_names.append(name)
                avg_days.append(mean_val)
                
                # Color by performance
                ratio = mean_val / benchmark
                if ratio > 1.5:
                    colors.append('#e74c3c')  # Red
                elif ratio > 1.0:
                    colors.append('#f39c12')  # Orange
                else:
                    colors.append('#2ecc71')  # Green
        
        if stage_names:
            y_pos = np.arange(len(stage_names))
            bars = ax.barh(y_pos, avg_days, color=colors, alpha=0.7, edgecolor='black')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(stage_names)
            ax.set_xlabel('Average Days')
            ax.set_title('Avg Duration by Stage', fontweight='bold', fontsize=11)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, days in zip(bars, avg_days):
                width = bar.get_width()
                ax.text(width + 2, bar.get_y() + bar.get_height()/2.,
                       f'{days:.0f}d', ha='left', va='center', fontweight='bold')
    
    def _plot_risk_distribution(self, ax):
        """Plot risk score distribution"""
        if 'risk_score' in self.master_data.columns:
            risk_counts = self.master_data['risk_score'].value_counts().sort_index()
            
            colors_map = {0: '#2ecc71', 1: '#2ecc71', 2: '#f39c12', 
                         3: '#f39c12', 4: '#e74c3c', 5: '#e74c3c', 6: '#e74c3c'}
            
            colors = [colors_map.get(score, '#95a5a6') for score in risk_counts.index]
            
            bars = ax.bar(risk_counts.index, risk_counts.values, color=colors, 
                         alpha=0.7, edgecolor='black')
            
            ax.set_xlabel('Risk Score')
            ax.set_ylabel('Number of Projects')
            ax.set_title('Risk Distribution', fontweight='bold', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        elif 'risk_level' in self.master_data.columns:
            risk_counts = self.master_data['risk_level'].value_counts()
            colors = ['#2ecc71', '#f39c12', '#f39c12', '#e74c3c']
            
            ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                  colors=colors[:len(risk_counts)], startangle=90)
            ax.set_title('Risk Distribution', fontweight='bold', fontsize=11)
    
    def _plot_program_performance(self, ax):
        """Plot performance by program"""
        if 'Program' in self.master_data.columns and 'days_total_process' in self.master_data.columns:
            program_avg = self.master_data.groupby('Program')['days_total_process'].mean().sort_values(ascending=True)
            
            if len(program_avg) > 0:
                # Truncate long names
                labels = [p[:20] + '...' if len(p) > 20 else p for p in program_avg.index]
                
                colors = ['#e74c3c' if v > 100 else '#f39c12' if v > 80 else '#2ecc71' 
                         for v in program_avg.values]
                
                y_pos = np.arange(len(program_avg))
                bars = ax.barh(y_pos, program_avg.values, color=colors, alpha=0.7, edgecolor='black')
                
                ax.axvline(x=100, color='green', linestyle='--', linewidth=2, 
                          label='Target', alpha=0.7)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_xlabel('Average Days')
                ax.set_title('Program Performance', fontweight='bold', fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(axis='x', alpha=0.3)
    
    def _plot_temporal_trends(self, ax):
        """Plot trends over time"""
        if 'intake_month' in self.master_data.columns and 'intake_year' in self.master_data.columns:
            # Create year-month column
            df = self.master_data.copy()
            df['year_month'] = pd.to_datetime(df['intake_year'].astype(str) + '-' + 
                                             df['intake_month'].astype(str), 
                                             format='%Y-%m', errors='coerce')
            
            if df['year_month'].notna().any():
                monthly_stats = df.groupby('year_month').agg({
                    'days_total_process': ['mean', 'count']
                }).sort_index()
                
                if len(monthly_stats) > 0:
                    # Plot average duration
                    ax.plot(monthly_stats.index, 
                           monthly_stats[('days_total_process', 'mean')],
                           marker='o', linewidth=2, markersize=6, color='#3498db', 
                           label='Avg Duration')
                    
                    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, 
                              label='Target (100d)', alpha=0.7)
                    
                    # Add count as bar chart
                    ax2 = ax.twinx()
                    ax2.bar(monthly_stats.index, 
                           monthly_stats[('days_total_process', 'count')],
                           alpha=0.3, color='gray', label='Project Count')
                    ax2.set_ylabel('Project Count', fontsize=9)
                    
                    ax.set_xlabel('Month')
                    ax.set_ylabel('Average Days')
                    ax.set_title('Temporal Trends', fontweight='bold', fontsize=12)
                    ax.legend(loc='upper left', fontsize=9)
                    ax2.legend(loc='upper right', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    # Format x-axis
                    ax.tick_params(axis='x', rotation=45)
    
    def _plot_bottleneck_analysis(self, ax):
        """Plot bottleneck severity analysis"""
        if 'stage_stats' in self.analyzer.analysis_results:
            stage_stats = self.analyzer.analysis_results['stage_stats']
            
            # Sort by severity
            stage_stats_sorted = stage_stats.sort_values('severity', ascending=True)
            
            stages = stage_stats_sorted['stage'].values
            severities = stage_stats_sorted['severity'].values
            
            # Color by severity
            colors = ['#e74c3c' if s > 1.5 else '#f39c12' if s > 1.0 else '#2ecc71' 
                     for s in severities]
            
            y_pos = np.arange(len(stages))
            bars = ax.barh(y_pos, severities, color=colors, alpha=0.7, edgecolor='black')
            
            ax.axvline(x=1.0, color='green', linestyle='--', linewidth=2, 
                      label='Benchmark', alpha=0.7)
            ax.axvline(x=1.5, color='orange', linestyle='--', linewidth=2, 
                      label='Critical Threshold', alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(stages, fontsize=9)
            ax.set_xlabel('Severity Ratio (Actual / Benchmark)')
            ax.set_title('Bottleneck Severity Analysis', fontweight='bold', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, severity in zip(bars, severities):
                width = bar.get_width()
                ax.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                       f'{severity:.2f}x', ha='left', va='center', 
                       fontweight='bold', fontsize=9)
    
    def create_process_flow_diagram(self, save_path=None):
        """Create visual process flow diagram with bottleneck highlighting"""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        fig.suptitle('AADB Data Access Process Flow - Bottleneck Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Define process stages
        stages = [
            {'name': 'NOA Received', 'x': 1, 'y': 6, 'w': 1.5, 'h': 0.8, 'color': '#3498db'},
            {'name': 'Initial Consult', 'x': 1, 'y': 4.5, 'w': 1.5, 'h': 0.8, 'color': self._get_stage_color('days_noa_to_consult', 30)},
            {'name': 'Data Request\nSubmitted', 'x': 3.5, 'y': 4.5, 'w': 1.5, 'h': 0.8, 'color': self._get_stage_color('days_intake_to_submit', 14)},
            {'name': 'IRB Process', 'x': 6, 'y': 6.5, 'w': 1.5, 'h': 0.8, 'color': self._get_stage_color('days_irb', 60)},
            {'name': 'DUA Process', 'x': 6, 'y': 4.5, 'w': 1.5, 'h': 0.8, 'color': self._get_stage_color('days_dua', 90)},
            {'name': 'Data Access\nProvided', 'x': 8.5, 'y': 5.5, 'w': 1.5, 'h': 0.8, 'color': '#2ecc71'}
        ]
        
        # Draw stages
        for stage in stages:
            box = FancyBboxPatch((stage['x'], stage['y']), stage['w'], stage['h'],
                                boxstyle="round,pad=0.05", 
                                facecolor=stage['color'],
                                edgecolor='black', linewidth=2, alpha=0.7)
            ax.add_patch(box)
            
            # Add stage name
            ax.text(stage['x'] + stage['w']/2, stage['y'] + stage['h']/2,
                   stage['name'], ha='center', va='center', 
                   fontweight='bold', fontsize=10, wrap=True)
            
            # Add duration if available
            duration = self._get_stage_duration(stage['name'])
            if duration:
                ax.text(stage['x'] + stage['w']/2, stage['y'] - 0.3,
                       f'{duration:.0f} days avg', ha='center', va='top',
                       fontsize=8, style='italic')
        
        # Draw arrows
        arrows = [
            {'from': (1.75, 6), 'to': (1.75, 5.3)},  # NOA to Consult
            {'from': (2.5, 4.9), 'to': (3.5, 4.9)},  # Consult to Request
            {'from': (5, 4.9), 'to': (6, 4.9)},  # Request to DUA
            {'from': (4.25, 5.2), 'to': (6, 6.7)},  # Request to IRB
            {'from': (7.5, 6.9), 'to': (9, 6.2)},  # IRB to Access
            {'from': (7.5, 4.9), 'to': (8.5, 5.3)},  # DUA to Access
        ]
        
        for arrow in arrows:
            arr = FancyArrowPatch(arrow['from'], arrow['to'],
                                 arrowstyle='->', mutation_scale=20,
                                 linewidth=2, color='black', alpha=0.6)
            ax.add_patch(arr)
        
        # Add legend
        legend_y = 2.5
        ax.text(0.5, legend_y + 0.5, 'Color Key:', fontweight='bold', fontsize=11)
        
        legend_items = [
            ('#2ecc71', 'On Target (≤benchmark)'),
            ('#f39c12', 'Moderate Delay (1-1.5x)'),
            ('#e74c3c', 'Critical Bottleneck (>1.5x)')
        ]
        
        for i, (color, label) in enumerate(legend_items):
            y = legend_y - (i * 0.4)
            box = Rectangle((0.5, y - 0.15), 0.3, 0.3, 
                          facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(box)
            ax.text(0.9, y, label, va='center', fontsize=9)
        
        # Add statistics box
        stats_text = self._get_process_stats_text()
        ax.text(0.5, 0.8, stats_text, fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               verticalalignment='top', family='monospace')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Process flow diagram saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def _get_stage_color(self, col, benchmark):
        """Get color based on stage performance"""
        if col in self.master_data.columns:
            avg = self.master_data[col].mean()
            if pd.notna(avg):
                ratio = avg / benchmark
                if ratio <= 1.0:
                    return '#2ecc71'  # Green
                elif ratio <= 1.5:
                    return '#f39c12'  # Orange
                else:
                    return '#e74c3c'  # Red
        return '#95a5a6'  # Gray (no data)
    
    def _get_stage_duration(self, stage_name):
        """Get average duration for a stage"""
        stage_map = {
            'Initial Consult': 'days_noa_to_consult',
            'Data Request\nSubmitted': 'days_intake_to_submit',
            'IRB Process': 'days_irb',
            'DUA Process': 'days_dua'
        }
        
        col = stage_map.get(stage_name)
        if col and col in self.master_data.columns:
            return self.master_data[col].mean()
        return None
    
    def _get_process_stats_text(self):
        """Get process statistics as formatted text"""
        stats = []
        stats.append("PROCESS STATISTICS")
        stats.append("-" * 30)
        
        if 'days_total_process' in self.master_data.columns:
            avg = self.master_data['days_total_process'].mean()
            median = self.master_data['days_total_process'].median()
            stats.append(f"Avg Duration:    {avg:6.1f} days")
            stats.append(f"Median Duration: {median:6.1f} days")
        
        total = len(self.master_data)
        if 'days_total_process' in self.master_data.columns:
            over_100 = (self.master_data['days_total_process'] > 100).sum()
            stats.append(f"Over 100 days:   {over_100:3d}/{total}")
        
        return '\n'.join(stats)
    
    def create_risk_prediction_chart(self, predictions_df, save_path=None):
        """Create visualization of risk predictions"""
        if self.predictor is None or predictions_df is None:
            print("No risk predictions available")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Risk Prediction Analysis', fontsize=16, fontweight='bold')
        
        # Risk level distribution
        risk_counts = predictions_df['Risk_Level'].value_counts()
        colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 
                 'High': '#e74c3c', 'Critical': '#c0392b'}
        
        risk_colors = [colors.get(level, '#95a5a6') for level in risk_counts.index]
        
        ax1.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
               colors=risk_colors, startangle=90)
        ax1.set_title('Risk Level Distribution')
        
        # Risk probability histogram
        ax2.hist(predictions_df['Risk_Probability'], bins=20, 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, 
                   label='High Risk Threshold')
        ax2.set_xlabel('Risk Probability')
        ax2.set_ylabel('Number of Projects')
        ax2.set_title('Risk Probability Distribution')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Risk prediction chart saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        return fig


if __name__ == "__main__":
    from data_loader import AADBDataLoader
    from process_analyzer import AADBProcessAnalyzer
    
    # Load and analyze data
    loader = AADBDataLoader()
    loader.load_all_data().preprocess_all()
    
    analyzer = AADBProcessAnalyzer(loader)
    analyzer.analyze_process_stages()
    analyzer.analyze_risk_factors()
    
    # Create visualizations
    viz = AADBVisualizer(analyzer)
    viz.create_executive_dashboard(save_path='../aadb_analysis/dashboard.png')
    viz.create_process_flow_diagram(save_path='../aadb_analysis/process_flow.png')

