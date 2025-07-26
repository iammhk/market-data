"""
Enhanced Spot vs Expiry Analysis Module

This module provides comprehensive, robust analysis functionality for comparing
Bank Nifty spot prices with options data across different expiry dates.

Features:
- Interactive dual y-axis plotting with real-time controls
- Comprehensive correlation and statistical analysis
- Automatic data validation and cleaning
- Support for multiple option types and price metrics
- Export capabilities and programmatic access
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from ipywidgets import widgets, VBox, HBox, Button, Output, HTML
from IPython.display import display, clear_output
from typing import Optional, Tuple, List, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')


class EnhancedSpotVsExpiryAnalyzer:
    """
    Enhanced interactive analyzer for comparing Bank Nifty spot prices with options data.
    
    This class provides:
    - Robust data validation and preprocessing
    - Interactive dual y-axis plotting with real-time updates
    - Comprehensive statistical analysis and correlation metrics
    - Multiple visualization modes and export capabilities
    - Error handling and user-friendly feedback
    """
    
    def __init__(self, spot_data: pd.DataFrame, call_data: pd.DataFrame, put_data: pd.DataFrame):
        """
        Initialize the enhanced analyzer with comprehensive data validation.
        
        Args:
            spot_data (pd.DataFrame): Bank Nifty spot data with Date and OHLC columns
            call_data (pd.DataFrame): Call options data with Date, Expiry, Strike, and OHLC columns
            put_data (pd.DataFrame): Put options data with Date, Expiry, Strike, and OHLC columns
        """
        print("🚀 Initializing Enhanced Spot vs Expiry Analyzer...")
        
        # Initialize data containers
        self.original_spot_data = spot_data.copy() if not spot_data.empty else pd.DataFrame()
        self.original_call_data = call_data.copy() if not call_data.empty else pd.DataFrame()
        self.original_put_data = put_data.copy() if not put_data.empty else pd.DataFrame()
        
        # Process and validate data
        self._validate_and_process_data()
        
        # Setup analysis components
        self._setup_expiry_options()
        self._setup_analysis_metrics()
        self._create_interface_components()
        
        # Initialize state variables
        self.current_analysis_data = {}
        self.export_data = {}
        
        print(f"✅ Analyzer initialized successfully!")
        if self.has_valid_data:
            print(f"📊 Available expiries: {len(self.expiry_options)}")
            print(f"📈 Spot data: {len(self.spot_data)} records")
            print(f"📞 Call options: {len(self.call_data)} records")
            print(f"📉 Put options: {len(self.put_data)} records")
    
    def _validate_and_process_data(self):
        """Validate and preprocess all input data with comprehensive error handling."""
        print("🔍 Validating and processing data...")
        
        self.validation_results = {
            'spot_data': {'valid': False, 'issues': []},
            'call_data': {'valid': False, 'issues': []},
            'put_data': {'valid': False, 'issues': []}
        }
        
        # Process spot data
        self.spot_data = self._process_spot_data()
        
        # Process options data
        self.call_data = self._process_options_data(self.original_call_data, 'call')
        self.put_data = self._process_options_data(self.original_put_data, 'put')
        
        # Determine overall data validity
        self.has_valid_data = (
            self.validation_results['spot_data']['valid'] and 
            (self.validation_results['call_data']['valid'] or self.validation_results['put_data']['valid'])
        )
        
        if self.has_valid_data:
            print("✅ Data validation completed successfully")
        else:
            print("⚠️ Data validation found issues:")
            for data_type, results in self.validation_results.items():
                if not results['valid'] and results['issues']:
                    print(f"   {data_type}: {', '.join(results['issues'])}")
    
    def _process_spot_data(self) -> pd.DataFrame:
        """Process and validate spot data."""
        if self.original_spot_data.empty:
            self.validation_results['spot_data']['issues'].append("No spot data provided")
            return pd.DataFrame()
        
        data = self.original_spot_data.copy()
        required_columns = ['Date', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.validation_results['spot_data']['issues'].append(f"Missing columns: {missing_columns}")
            return pd.DataFrame()
        
        # Convert Date column
        try:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            invalid_dates = data['Date'].isna().sum()
            if invalid_dates > 0:
                print(f"⚠️ Removed {invalid_dates} records with invalid dates from spot data")
                data = data.dropna(subset=['Date'])
        except Exception as e:
            self.validation_results['spot_data']['issues'].append(f"Date conversion error: {str(e)}")
            return pd.DataFrame()
        
        # Validate numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove rows with all NaN numeric values
        data = data.dropna(subset=['Close'])
        
        if len(data) == 0:
            self.validation_results['spot_data']['issues'].append("No valid records after cleaning")
            return pd.DataFrame()
        
        # Sort by date
        data = data.sort_values('Date').reset_index(drop=True)
        
        self.validation_results['spot_data']['valid'] = True
        return data
    
    def _process_options_data(self, data: pd.DataFrame, option_type: str) -> pd.DataFrame:
        """Process and validate options data."""
        validation_key = f'{option_type}_data'
        
        if data.empty:
            self.validation_results[validation_key]['issues'].append(f"No {option_type} options data provided")
            return pd.DataFrame()
        
        processed_data = data.copy()
        required_columns = ['Date', 'Expiry', 'Close']
        
        # Check for Strike column (could be 'Strike' or 'Strike Price')
        strike_col = None
        if 'Strike' in processed_data.columns:
            strike_col = 'Strike'
        elif 'Strike Price' in processed_data.columns:
            strike_col = 'Strike Price'
            processed_data = processed_data.rename(columns={'Strike Price': 'Strike'})
        
        if strike_col is None:
            required_columns.append('Strike')  # Will fail validation below
        
        missing_columns = [col for col in required_columns if col not in processed_data.columns]
        if missing_columns:
            self.validation_results[validation_key]['issues'].append(f"Missing columns: {missing_columns}")
            return pd.DataFrame()
        
        # Convert Date and Expiry columns
        try:
            processed_data['Date'] = pd.to_datetime(processed_data['Date'], errors='coerce')
            processed_data['Expiry'] = pd.to_datetime(processed_data['Expiry'], errors='coerce')
            
            # Remove rows with invalid dates
            before_count = len(processed_data)
            processed_data = processed_data.dropna(subset=['Date', 'Expiry'])
            after_count = len(processed_data)
            
            if before_count > after_count:
                print(f"⚠️ Removed {before_count - after_count} records with invalid dates from {option_type} data")
                
        except Exception as e:
            self.validation_results[validation_key]['issues'].append(f"Date conversion error: {str(e)}")
            return pd.DataFrame()
        
        # Convert numeric columns
        numeric_columns = ['Strike', 'Open', 'High', 'Low', 'Close']
        for col in numeric_columns:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        # Remove rows with invalid essential data
        processed_data = processed_data.dropna(subset=['Strike', 'Close'])
        
        if len(processed_data) == 0:
            self.validation_results[validation_key]['issues'].append("No valid records after cleaning")
            return pd.DataFrame()
        
        # Sort by date and expiry
        processed_data = processed_data.sort_values(['Date', 'Expiry', 'Strike']).reset_index(drop=True)
        
        self.validation_results[validation_key]['valid'] = True
        return processed_data
    
    def _setup_expiry_options(self):
        """Setup available expiry dates from validated options data."""
        all_expiries = set()
        
        if not self.call_data.empty and 'Expiry' in self.call_data.columns:
            all_expiries.update(self.call_data['Expiry'].dropna().unique())
        if not self.put_data.empty and 'Expiry' in self.put_data.columns:
            all_expiries.update(self.put_data['Expiry'].dropna().unique())
        
        # Sort expiries and create options
        sorted_expiries = sorted([exp for exp in all_expiries if pd.notna(exp)])
        self.expiry_options = [(exp.strftime('%d-%b-%Y'), exp) for exp in sorted_expiries]
        
        # Add summary statistics
        self.expiry_stats = {}
        for exp_label, exp_date in self.expiry_options:
            call_count = len(self.call_data[self.call_data['Expiry'] == exp_date]) if not self.call_data.empty else 0
            put_count = len(self.put_data[self.put_data['Expiry'] == exp_date]) if not self.put_data.empty else 0
            self.expiry_stats[exp_label] = {'calls': call_count, 'puts': put_count, 'total': call_count + put_count}
    
    def _setup_analysis_metrics(self):
        """Setup analysis metrics and calculation methods."""
        self.price_columns = ['Open', 'High', 'Low', 'Close']
        self.correlation_methods = ['pearson', 'spearman', 'kendall']
        self.analysis_modes = ['basic', 'detailed', 'correlation_matrix']
    
    def _create_interface_components(self):
        """Create all interface components with enhanced functionality."""
        if not self.has_valid_data:
            self.widgets_created = False
            return
        
        # Enhanced expiry dropdown with statistics
        expiry_display_options = []
        for exp_label, exp_date in self.expiry_options:
            stats = self.expiry_stats[exp_label]
            display_text = f"{exp_label} (C:{stats['calls']}, P:{stats['puts']})"
            expiry_display_options.append((display_text, exp_date))
        
        self.expiry_dropdown = widgets.Dropdown(
            options=expiry_display_options,
            value=expiry_display_options[0][1] if expiry_display_options else None,
            description='📅 Expiry:',
            style={'description_width': '80px'},
            layout={'width': '350px'}
        )
        
        # Enhanced option type selection
        self.option_type_dropdown = widgets.Dropdown(
            options=[
                ('📞 Call Options Only', 'call'), 
                ('📉 Put Options Only', 'put'), 
                ('📊 Both Call & Put', 'both'),
                ('📈 Call vs Put Comparison', 'comparison')
            ],
            value='both',
            description='📊 Type:',
            style={'description_width': '80px'},
            layout={'width': '250px'}
        )
        
        # Price type selectors
        self.spot_price_dropdown = widgets.Dropdown(
            options=[(col, col) for col in self.price_columns if col in self.spot_data.columns],
            value='Close',
            description='💰 Spot:',
            style={'description_width': '80px'},
            layout={'width': '140px'}
        )
        
        self.options_price_dropdown = widgets.Dropdown(
            options=[(col, col) for col in self.price_columns],
            value='Close',
            description='💎 Options:',
            style={'description_width': '80px'},
            layout={'width': '150px'}
        )
        
        # Analysis mode selector
        self.analysis_mode_dropdown = widgets.Dropdown(
            options=[
                ('Basic Analysis', 'basic'),
                ('Detailed Statistics', 'detailed'),
                ('Correlation Matrix', 'correlation_matrix')
            ],
            value='detailed',
            description='🔍 Mode:',
            style={'description_width': '80px'},
            layout={'width': '180px'}
        )
        
        # Action buttons
        self.plot_button = widgets.Button(
            description='📈 Analyze',
            button_style='success',
            tooltip='Generate comprehensive analysis',
            layout={'width': '120px'}
        )
        
        self.export_button = widgets.Button(
            description='💾 Export',
            button_style='info',
            tooltip='Export analysis data',
            layout={'width': '100px'}
        )
        
        # Connect event handlers
        self.plot_button.on_click(self._on_analyze_button_click)
        self.export_button.on_click(self._on_export_button_click)
        
        # Output areas
        self.plot_output = Output()
        self.stats_output = Output()
        
        # Status indicator
        self.status_html = HTML(value="<b>📊 Ready for analysis</b>")
        
        self.widgets_created = True
    
    def _on_analyze_button_click(self, button):
        """Handle analyze button click with comprehensive error handling."""
        self.status_html.value = "<b>🔄 Analyzing...</b>"
        
        try:
            with self.plot_output:
                self.plot_output.clear_output(wait=True)
                self._perform_comprehensive_analysis()
            self.status_html.value = "<b>✅ Analysis completed</b>"
        except Exception as e:
            self.status_html.value = f"<b>❌ Analysis failed: {str(e)}</b>"
            with self.plot_output:
                print(f"❌ Error during analysis: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _on_export_button_click(self, button):
        """Handle export button click."""
        if hasattr(self, 'current_analysis_data') and self.current_analysis_data:
            print("💾 Export functionality would be implemented here")
            print("📊 Available data for export:")
            for key, value in self.current_analysis_data.items():
                if isinstance(value, pd.DataFrame):
                    print(f"   {key}: {len(value)} records")
                else:
                    print(f"   {key}: {type(value)}")
        else:
            print("❌ No analysis data available for export. Please run analysis first.")
    
    def _perform_comprehensive_analysis(self):
        """Perform comprehensive analysis based on selected parameters."""
        expiry = self.expiry_dropdown.value
        option_type = self.option_type_dropdown.value
        spot_price_type = self.spot_price_dropdown.value
        options_price_type = self.options_price_dropdown.value
        analysis_mode = self.analysis_mode_dropdown.value
        
        print(f"🔍 Performing {analysis_mode} analysis for {expiry.strftime('%d-%b-%Y')}")
        
        # Get filtered data
        analysis_data = self._prepare_analysis_data(expiry, option_type, spot_price_type, options_price_type)
        
        if not analysis_data['has_data']:
            print("❌ No data available for selected parameters")
            return
        
        # Store for export
        self.current_analysis_data = analysis_data
        
        # Create visualization based on option type
        if option_type == 'comparison':
            self._create_comparison_plot(analysis_data)
        else:
            self._create_standard_plot(analysis_data)
        
        # Display statistics based on mode
        if analysis_mode in ['detailed', 'correlation_matrix']:
            self._display_detailed_statistics(analysis_data, analysis_mode)
    
    def _prepare_analysis_data(self, expiry: datetime, option_type: str, 
                             spot_price_type: str, options_price_type: str) -> Dict:
        """Prepare all data needed for analysis."""
        # Filter options data by expiry
        call_data = pd.DataFrame()
        put_data = pd.DataFrame()
        
        if option_type in ['call', 'both', 'comparison'] and not self.call_data.empty:
            call_data = self.call_data[self.call_data['Expiry'] == expiry].copy()
        
        if option_type in ['put', 'both', 'comparison'] and not self.put_data.empty:
            put_data = self.put_data[self.put_data['Expiry'] == expiry].copy()
        
        # Check if we have any data
        has_data = not call_data.empty or not put_data.empty
        
        if not has_data:
            return {'has_data': False}
        
        # Get date range from options data
        all_dates = []
        if not call_data.empty:
            all_dates.extend(call_data['Date'].tolist())
        if not put_data.empty:
            all_dates.extend(put_data['Date'].tolist())
        
        date_range = {
            'start': min(all_dates),
            'end': max(all_dates)
        }
        
        # Filter spot data for the same period
        spot_mask = (
            (self.spot_data['Date'] >= date_range['start']) & 
            (self.spot_data['Date'] <= date_range['end'])
        )
        spot_data = self.spot_data.loc[spot_mask].copy()
        
        # Calculate daily aggregates for options
        call_daily, put_daily = pd.DataFrame(), pd.DataFrame()
        
        if not call_data.empty:
            call_daily = call_data.groupby('Date').agg({
                options_price_type: ['mean', 'std', 'min', 'max', 'count'],
                'Strike': ['min', 'max', 'count']
            }).round(2)
            call_daily.columns = ['price_mean', 'price_std', 'price_min', 'price_max', 'price_count',
                                'strike_min', 'strike_max', 'strike_count']
            call_daily = call_daily.reset_index()
        
        if not put_data.empty:
            put_daily = put_data.groupby('Date').agg({
                options_price_type: ['mean', 'std', 'min', 'max', 'count'],
                'Strike': ['min', 'max', 'count']
            }).round(2)
            put_daily.columns = ['price_mean', 'price_std', 'price_min', 'price_max', 'price_count',
                               'strike_min', 'strike_max', 'strike_count']
            put_daily = put_daily.reset_index()
        
        return {
            'has_data': True,
            'expiry': expiry,
            'option_type': option_type,
            'spot_price_type': spot_price_type,
            'options_price_type': options_price_type,
            'date_range': date_range,
            'spot_data': spot_data,
            'call_data': call_data,
            'put_data': put_data,
            'call_daily': call_daily,
            'put_daily': put_daily
        }
    
    def _create_standard_plot(self, data: Dict):
        """Create standard dual y-axis plot."""
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]],
            subplot_titles=[f"Spot vs Options Analysis - {data['expiry'].strftime('%d-%b-%Y')}"]
        )
        
        # Add spot price line
        fig.add_trace(
            go.Scatter(
                x=data['spot_data']['Date'],
                y=data['spot_data'][data['spot_price_type']],
                mode='lines',
                name=f"Bank Nifty Spot ({data['spot_price_type']})",
                line=dict(color='#2E8B57', width=3),
                hovertemplate=f"Date: %{{x}}<br>Spot {data['spot_price_type']}: ₹%{{y:,.0f}}<extra></extra>"
            ),
            secondary_y=False
        )
        
        # Add options traces
        if data['option_type'] in ['call', 'both'] and not data['call_daily'].empty:
            fig.add_trace(
                go.Scatter(
                    x=data['call_daily']['Date'],
                    y=data['call_daily']['price_mean'],
                    mode='lines+markers',
                    name=f"Call Options (Avg {data['options_price_type']})",
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4),
                    hovertemplate=f"Date: %{{x}}<br>Call Avg: ₹%{{y:,.0f}}<extra></extra>"
                ),
                secondary_y=True
            )
        
        if data['option_type'] in ['put', 'both'] and not data['put_daily'].empty:
            fig.add_trace(
                go.Scatter(
                    x=data['put_daily']['Date'],
                    y=data['put_daily']['price_mean'],
                    mode='lines+markers',
                    name=f"Put Options (Avg {data['options_price_type']})",
                    line=dict(color='#d62728', width=2),
                    marker=dict(size=4),
                    hovertemplate=f"Date: %{{x}}<br>Put Avg: ₹%{{y:,.0f}}<extra></extra>"
                ),
                secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            height=600,
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98),
            template='plotly_white',
            title_x=0.5
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Spot Price (₹)", secondary_y=False, title_font=dict(color='#2E8B57'))
        fig.update_yaxes(title_text="Options Price (₹)", secondary_y=True, title_font=dict(color='#ff7f0e'))
        
        fig.show()
    
    def _create_comparison_plot(self, data: Dict):
        """Create comparison plot for call vs put analysis."""
        if data['call_daily'].empty or data['put_daily'].empty:
            print("❌ Both call and put data required for comparison mode")
            return
        
        # Merge call and put data
        merged = pd.merge(
            data['call_daily'][['Date', 'price_mean']].rename(columns={'price_mean': 'call_price'}),
            data['put_daily'][['Date', 'price_mean']].rename(columns={'price_mean': 'put_price'}),
            on='Date', how='outer'
        )
        
        # Add spot data
        merged = pd.merge(
            merged,
            data['spot_data'][['Date', data['spot_price_type']]].rename(columns={data['spot_price_type']: 'spot_price'}),
            on='Date', how='outer'
        ).dropna()
        
        if merged.empty:
            print("❌ No overlapping dates found for comparison")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Call vs Put Prices', 'Call-Put Spread', 'Price Ratios', 'Correlation Timeline'],
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Call vs Put prices with spot
        fig.add_trace(
            go.Scatter(x=merged['Date'], y=merged['spot_price'], name='Spot', line=dict(color='green')),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=merged['Date'], y=merged['call_price'], name='Call', line=dict(color='blue')),
            row=1, col=1, secondary_y=True
        )
        fig.add_trace(
            go.Scatter(x=merged['Date'], y=merged['put_price'], name='Put', line=dict(color='red')),
            row=1, col=1, secondary_y=True
        )
        
        # Plot 2: Call-Put spread
        merged['call_put_spread'] = merged['call_price'] - merged['put_price']
        fig.add_trace(
            go.Scatter(x=merged['Date'], y=merged['call_put_spread'], name='Call-Put Spread', 
                      line=dict(color='purple'), showlegend=False),
            row=1, col=2
        )
        
        # Plot 3: Price ratios
        merged['call_spot_ratio'] = merged['call_price'] / merged['spot_price']
        merged['put_spot_ratio'] = merged['put_price'] / merged['spot_price']
        fig.add_trace(
            go.Scatter(x=merged['Date'], y=merged['call_spot_ratio'], name='Call/Spot Ratio', 
                      line=dict(color='lightblue'), showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=merged['Date'], y=merged['put_spot_ratio'], name='Put/Spot Ratio', 
                      line=dict(color='lightcoral'), showlegend=False),
            row=2, col=1
        )
        
        # Plot 4: Rolling correlations
        window = min(10, len(merged) // 2)
        if window >= 3:
            merged['call_spot_corr'] = merged['call_price'].rolling(window).corr(merged['spot_price'])
            merged['put_spot_corr'] = merged['put_price'].rolling(window).corr(merged['spot_price'])
            
            fig.add_trace(
                go.Scatter(x=merged['Date'], y=merged['call_spot_corr'], name='Call-Spot Correlation', 
                          line=dict(color='blue', dash='dash'), showlegend=False),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=merged['Date'], y=merged['put_spot_corr'], name='Put-Spot Correlation', 
                          line=dict(color='red', dash='dash'), showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Comprehensive Call vs Put Analysis", title_x=0.5)
        fig.show()
    
    def _display_detailed_statistics(self, data: Dict, mode: str):
        """Display detailed statistical analysis."""
        print(f"\n📊 DETAILED STATISTICAL ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print(f"🎯 Expiry Date: {data['expiry'].strftime('%d-%b-%Y')}")
        print(f"📅 Analysis Period: {data['date_range']['start'].strftime('%d-%b-%Y')} to {data['date_range']['end'].strftime('%d-%b-%Y')}")
        print(f"📈 Spot Data Points: {len(data['spot_data'])}")
        
        if not data['call_data'].empty:
            print(f"📞 Call Options Records: {len(data['call_data'])}")
            print(f"🎯 Call Strike Range: ₹{data['call_data']['Strike'].min():,.0f} - ₹{data['call_data']['Strike'].max():,.0f}")
        
        if not data['put_data'].empty:
            print(f"📉 Put Options Records: {len(data['put_data'])}")
            print(f"🎯 Put Strike Range: ₹{data['put_data']['Strike'].min():,.0f} - ₹{data['put_data']['Strike'].max():,.0f}")
        
        # Price statistics
        spot_stats = data['spot_data'][data['spot_price_type']].describe()
        print(f"\n💰 SPOT PRICE STATISTICS ({data['spot_price_type']}):")
        print(f"   Average: ₹{spot_stats['mean']:,.2f}")
        print(f"   Std Dev: ₹{spot_stats['std']:,.2f}")
        print(f"   Range: ₹{spot_stats['min']:,.2f} - ₹{spot_stats['max']:,.2f}")
        
        # Options statistics
        if not data['call_daily'].empty:
            call_stats = data['call_daily']['price_mean'].describe()
            print(f"\n📞 CALL OPTIONS STATISTICS (Avg {data['options_price_type']}):")
            print(f"   Average: ₹{call_stats['mean']:,.2f}")
            print(f"   Std Dev: ₹{call_stats['std']:,.2f}")
            print(f"   Range: ₹{call_stats['min']:,.2f} - ₹{call_stats['max']:,.2f}")
        
        if not data['put_daily'].empty:
            put_stats = data['put_daily']['price_mean'].describe()
            print(f"\n📉 PUT OPTIONS STATISTICS (Avg {data['options_price_type']}):")
            print(f"   Average: ₹{put_stats['mean']:,.2f}")
            print(f"   Std Dev: ₹{put_stats['std']:,.2f}")
            print(f"   Range: ₹{put_stats['min']:,.2f} - ₹{put_stats['max']:,.2f}")
        
        # Correlation analysis
        if mode == 'correlation_matrix':
            self._display_correlation_matrix(data)
    
    def _display_correlation_matrix(self, data: Dict):
        """Display comprehensive correlation analysis."""
        print(f"\n🔗 CORRELATION ANALYSIS")
        print("-" * 30)
        
        # Prepare data for correlation
        corr_data = data['spot_data'][['Date', data['spot_price_type']]].copy()
        corr_data.columns = ['Date', 'Spot']
        
        if not data['call_daily'].empty:
            call_corr = data['call_daily'][['Date', 'price_mean']].copy()
            call_corr.columns = ['Date', 'Call']
            corr_data = pd.merge(corr_data, call_corr, on='Date', how='outer')
        
        if not data['put_daily'].empty:
            put_corr = data['put_daily'][['Date', 'price_mean']].copy()
            put_corr.columns = ['Date', 'Put']
            corr_data = pd.merge(corr_data, put_corr, on='Date', how='outer')
        
        # Calculate correlations
        numeric_cols = [col for col in corr_data.columns if col != 'Date']
        if len(numeric_cols) >= 2:
            for method in ['pearson', 'spearman']:
                print(f"\n📊 {method.title()} Correlations:")
                corr_matrix = corr_data[numeric_cols].corr(method=method)
                
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:  # Only show upper triangle
                            corr_val = corr_matrix.loc[col1, col2]
                            if not pd.isna(corr_val):
                                print(f"   {col1} vs {col2}: {corr_val:.4f}")
    
    def display_interface(self):
        """Display the complete enhanced interactive interface."""
        print("🚀 ENHANCED SPOT vs EXPIRY ANALYSIS")
        print("=" * 60)
        
        if not self.has_valid_data:
            print("❌ Insufficient data for analysis")
            print("\n🔍 Data Validation Results:")
            for data_type, results in self.validation_results.items():
                status = "✅" if results['valid'] else "❌"
                print(f"   {status} {data_type}: {', '.join(results['issues']) if results['issues'] else 'Valid'}")
            return
        
        if not self.widgets_created:
            print("❌ Unable to create interface components")
            return
        
        print("✅ All data validated successfully")
        print(f"📊 Ready for analysis with {len(self.expiry_options)} expiry dates")
        
        # Create enhanced interface layout
        print(f"\n🎛️ ENHANCED ANALYSIS CONTROLS:")
        print("-" * 40)
        
        # Controls layout
        controls_row1 = HBox([
            self.expiry_dropdown,
            self.option_type_dropdown,
            self.analysis_mode_dropdown
        ])
        
        controls_row2 = HBox([
            self.spot_price_dropdown,
            self.options_price_dropdown,
            self.plot_button,
            self.export_button
        ])
        
        # Display interface
        display(self.status_html)
        display(controls_row1)
        display(controls_row2)
        display(self.plot_output)
        
        # Auto-run initial analysis
        print("\n📈 Running initial analysis...")
        self._on_analyze_button_click(None)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary."""
        summary = {
            "analyzer_version": "Enhanced v2.0",
            "has_valid_data": self.has_valid_data,
            "validation_results": self.validation_results,
            "available_expiries": len(self.expiry_options),
            "expiry_dates": [exp[0] for exp in self.expiry_options],
            "expiry_statistics": self.expiry_stats
        }
        
        if not self.spot_data.empty:
            summary["spot_data_info"] = {
                "records": len(self.spot_data),
                "date_range": {
                    "start": self.spot_data['Date'].min().strftime('%Y-%m-%d'),
                    "end": self.spot_data['Date'].max().strftime('%Y-%m-%d')
                },
                "available_columns": list(self.spot_data.columns)
            }
        
        for option_type, data in [("call_data", self.call_data), ("put_data", self.put_data)]:
            if not data.empty:
                summary[f"{option_type}_info"] = {
                    "records": len(data),
                    "unique_expiries": data['Expiry'].nunique(),
                    "unique_strikes": data['Strike'].nunique(),
                    "strike_range": {
                        "min": float(data['Strike'].min()),
                        "max": float(data['Strike'].max())
                    }
                }
        
        return summary


# Factory and convenience functions
def create_enhanced_spot_vs_expiry_analyzer(spot_data: pd.DataFrame, call_data: pd.DataFrame, 
                                          put_data: pd.DataFrame) -> EnhancedSpotVsExpiryAnalyzer:
    """
    Factory function to create an enhanced spot vs expiry analyzer.
    
    Args:
        spot_data (pd.DataFrame): Bank Nifty spot data
        call_data (pd.DataFrame): Call options data  
        put_data (pd.DataFrame): Put options data
        
    Returns:
        EnhancedSpotVsExpiryAnalyzer: Configured analyzer instance
    """
    return EnhancedSpotVsExpiryAnalyzer(spot_data, call_data, put_data)


def display_spot_vs_expiry_analysis(spot_data: pd.DataFrame, call_data: pd.DataFrame, 
                                   put_data: pd.DataFrame) -> EnhancedSpotVsExpiryAnalyzer:
    """
    Convenience function to create and immediately display an enhanced analyzer.
    
    Args:
        spot_data (pd.DataFrame): Bank Nifty spot data
        call_data (pd.DataFrame): Call options data
        put_data (pd.DataFrame): Put options data
        
    Returns:
        EnhancedSpotVsExpiryAnalyzer: Analyzer instance for further use
    """
    analyzer = create_enhanced_spot_vs_expiry_analyzer(spot_data, call_data, put_data)
    analyzer.display_interface()
    return analyzer


def plot_spot_vs_expiry_simple(spot_data: pd.DataFrame, call_data: pd.DataFrame, 
                               put_data: pd.DataFrame, expiry_date: str, 
                               option_type: str = 'call', spot_price_type: str = 'Close',
                               options_price_type: str = 'Close'):
    """
    Create a simple spot vs expiry plot without interactive controls.
    
    Args:
        spot_data (pd.DataFrame): Bank Nifty spot data
        call_data (pd.DataFrame): Call options data
        put_data (pd.DataFrame): Put options data
        expiry_date (str): Expiry date in 'YYYY-MM-DD' format
        option_type (str): Type of options ('call', 'put', 'both')
        spot_price_type (str): Spot price column to use
        options_price_type (str): Options price column to use
    """
    try:
        expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
        analyzer = create_enhanced_spot_vs_expiry_analyzer(spot_data, call_data, put_data)
        
        if analyzer.has_valid_data:
            # Use the enhanced analyzer's analysis method
            analysis_data = analyzer._prepare_analysis_data(expiry_dt, option_type, spot_price_type, options_price_type)
            if analysis_data['has_data']:
                analyzer._create_standard_plot(analysis_data)
                analyzer._display_detailed_statistics(analysis_data, 'detailed')
            else:
                print("❌ No data available for the specified parameters")
        else:
            print("❌ Insufficient data for analysis")
            
    except ValueError as e:
        print(f"❌ Date format error: {e}")
        print("Please use 'YYYY-MM-DD' format for expiry_date")


# Legacy compatibility aliases
SpotVsExpiryAnalyzer = EnhancedSpotVsExpiryAnalyzer
create_spot_vs_expiry_analyzer = create_enhanced_spot_vs_expiry_analyzer
