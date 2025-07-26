"""
Options Interactive Visualization Module

This module provides comprehensive interactive visualization capabilities for Bank Nifty options data,
including scatter plots, heatmaps, correlation matrices, timeline plots, and strike distributions
with superimposed fading effects and interactive controls.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import widgets, VBox, HBox, Output
from IPython.display import display, clear_output
from typing import Tuple, Optional, Dict, Any


class OptionsVisualizer:
    """
    Interactive Options Data Visualizer with superimposed fading effects.
    
    This class provides comprehensive visualization capabilities for options data
    with interactive controls for blending Call (CE) and Put (PE) options analysis.
    """
    
    def __init__(self, df_call: pd.DataFrame, df_put: pd.DataFrame):
        """
        Initialize the Options Visualizer.
        
        Args:
            df_call (pd.DataFrame): Call options data
            df_put (pd.DataFrame): Put options data
        """
        self.df_call = df_call.copy() if not df_call.empty else pd.DataFrame()
        self.df_put = df_put.copy() if not df_put.empty else pd.DataFrame()
        
        # Initialize widgets
        self.option_type_slider = None
        self.slider_labels = None
        self.viz_type_selector = None
        self.generate_button = None
        self.plot_output = None
        
        self._setup_widgets()
    
    def _setup_widgets(self):
        """Setup interactive widgets for the visualizer."""
        # Create interactive controls
        self.option_type_slider = widgets.FloatSlider(
            value=0.0,
            min=0.0,
            max=1.0,
            step=0.1,
            description='Call ← → Put:',
            style={'description_width': '100px'},
            layout={'width': '300px'},
            readout_format='.1f'
        )
        
        # Add custom labels for slider
        self.slider_labels = widgets.HTML(
            value='<div style="display: flex; justify-content: space-between; width: 280px; margin-top: -5px; font-size: 11px; color: #666;"><span>📞 Call (CE)</span><span>📊 Mixed</span><span>📉 Put (PE)</span></div>',
            layout={'width': '300px'}
        )
        
        self.viz_type_selector = widgets.Dropdown(
            options=[
                ('Volume vs Price Scatter', 'scatter'),
                ('Returns Heatmap', 'heatmap'),
                ('Correlation Matrix', 'correlation'),
                ('Volume & OI Timeline', 'timeline'),
                ('Strike Distribution', 'strike'),
                ('All Visualizations', 'all')
            ],
            value='all',
            description='Visualization:',
            style={'description_width': '100px'},
            layout={'width': '220px'}
        )
        
        self.generate_button = widgets.Button(
            description='📈 Generate Plots',
            button_style='success',
            layout={'width': '140px'}
        )
        
        # Create output area
        self.plot_output = widgets.Output()
        
        # Set up button click handler
        self.generate_button.on_click(self._on_generate_click)
    
    def get_data_for_analysis(self, slider_value: float) -> Tuple[pd.DataFrame, str, float, float]:
        """
        Get blended dataframe based on slider position with smooth transition.
        
        Args:
            slider_value (float): Slider position (0.0 = Call, 1.0 = Put)
            
        Returns:
            Tuple containing:
                - Combined DataFrame
                - Title suffix string
                - Call ratio (1.0 to 0.0)
                - Put ratio (0.0 to 1.0)
        """
        if self.df_call.empty and self.df_put.empty:
            return pd.DataFrame(), 'No Data', 1.0, 0.0
        elif self.df_call.empty:
            return self.df_put.copy(), 'Put Options (PE)', 0.0, 1.0
        elif self.df_put.empty:
            return self.df_call.copy(), 'Call Options (CE)', 1.0, 0.0
        
        # Calculate blend ratios
        call_ratio = 1.0 - slider_value
        put_ratio = slider_value
        
        # Determine primary dataset and title
        if slider_value <= 0.2:
            primary_data = self.df_call.copy()
            title_suffix = 'Call Options (CE)'
        elif slider_value >= 0.8:
            primary_data = self.df_put.copy()
            title_suffix = 'Put Options (PE)'
        else:
            # Blend the datasets for mixed analysis
            primary_data = pd.concat([self.df_call, self.df_put], ignore_index=True)
            title_suffix = f'Mixed Options (CE: {call_ratio:.1f}, PE: {put_ratio:.1f})'
        
        return primary_data, title_suffix, call_ratio, put_ratio
    
    def create_scatter_plot(self, df_clean: pd.DataFrame, title_suffix: str, call_ratio: float, put_ratio: float):
        """Create volume vs price scatter plot with superimposed fading effect."""
        if ('No. of contracts' not in self.df_call.columns and 'No. of contracts' not in self.df_put.columns):
            print("❌ No data available for scatter plot")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Get top volume data for both call and put options separately
        call_top = self.df_call.nlargest(100, 'No. of contracts') if not self.df_call.empty else pd.DataFrame()
        put_top = self.df_put.nlargest(100, 'No. of contracts') if not self.df_put.empty else pd.DataFrame()
        
        # Add Contract_Month column for both datasets
        if not call_top.empty:
            call_top = call_top.copy()
            call_top['Contract_Month'] = call_top['Expiry'].dt.strftime('%b-%Y')
        if not put_top.empty:
            put_top = put_top.copy()
            put_top['Contract_Month'] = put_top['Expiry'].dt.strftime('%b-%Y')
        
        # Plot Call Options with fading alpha based on slider position
        if not call_top.empty and 'Strike Price' in call_top.columns:
            call_top['Strike_Group'] = pd.cut(call_top['Strike Price'], bins=5, 
                                            labels=['Low', 'Med-Low', 'Med', 'Med-High', 'High'])
            
            # Call options - blue palette with alpha based on call_ratio
            call_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
            call_alpha = call_ratio
            if call_alpha > 0.01:  # Only plot if not fully transparent
                sns.scatterplot(data=call_top, x='Close', y='No. of contracts', 
                              hue='Strike_Group', style='Contract_Month',
                              palette=call_colors, alpha=call_alpha, s=60,
                              legend=False)
        
        # Plot Put Options with fading alpha based on slider position  
        if not put_top.empty and 'Strike Price' in put_top.columns:
            put_top['Strike_Group'] = pd.cut(put_top['Strike Price'], bins=5,
                                           labels=['Low', 'Med-Low', 'Med', 'Med-High', 'High'])
            
            # Put options - red palette with alpha based on put_ratio
            put_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 5))
            put_alpha = put_ratio
            if put_alpha > 0.01:  # Only plot if not fully transparent
                sns.scatterplot(data=put_top, x='Close', y='No. of contracts',
                              hue='Strike_Group', style='Contract_Month', 
                              palette=put_colors, alpha=put_alpha, s=60,
                              legend=False)
        
        # Create custom legend showing both call and put with current transparency levels
        legend_elements = []
        if not call_top.empty and call_ratio > 0.01:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor='blue', markersize=8, 
                                            alpha=call_ratio,
                                            label=f'Call Options (α={call_ratio:.1f})'))
        if not put_top.empty and put_ratio > 0.01:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor='red', markersize=8,
                                            alpha=put_ratio, 
                                            label=f'Put Options (α={put_ratio:.1f})'))
        
        if legend_elements:
            plt.legend(handles=legend_elements, title='Option Types (Superimposed)', 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(f'Top 100 Days: Volume vs. Close Price - Superimposed Fade Effect\\n({title_suffix})')
        plt.xlabel('Close Price (₹)')
        plt.ylabel('Contracts Traded')
        plt.tight_layout()
        plt.show()
    
    def create_returns_heatmap(self, df_clean: pd.DataFrame, title_suffix: str, call_ratio: float, put_ratio: float):
        """Create daily returns heatmap with superimposed fading effect."""
        if self.df_call.empty and self.df_put.empty:
            print("❌ No data available for returns heatmap")
            return
            
        plt.figure(figsize=(10, 6))
        
        # Calculate returns for both call and put options separately
        call_returns = pd.DataFrame()
        put_returns = pd.DataFrame()
        
        if not self.df_call.empty:
            df_call_copy = self.df_call.copy()
            if 'Strike Price' in df_call_copy.columns:
                df_call_copy['Daily_Return'] = df_call_copy.groupby(['Expiry', 'Strike Price'])['Close'].pct_change() * 100
            else:
                df_call_copy['Daily_Return'] = df_call_copy.groupby('Expiry')['Close'].pct_change() * 100
            
            df_call_copy['Year'] = df_call_copy['Date'].dt.year
            df_call_copy['Month'] = df_call_copy['Date'].dt.strftime('%b')
            call_returns = df_call_copy.pivot_table(index='Month', columns='Year', values='Daily_Return', aggfunc='mean')
        
        if not self.df_put.empty:
            df_put_copy = self.df_put.copy()
            if 'Strike Price' in df_put_copy.columns:
                df_put_copy['Daily_Return'] = df_put_copy.groupby(['Expiry', 'Strike Price'])['Close'].pct_change() * 100
            else:
                df_put_copy['Daily_Return'] = df_put_copy.groupby('Expiry')['Close'].pct_change() * 100
                
            df_put_copy['Year'] = df_put_copy['Date'].dt.year
            df_put_copy['Month'] = df_put_copy['Date'].dt.strftime('%b')
            put_returns = df_put_copy.pivot_table(index='Month', columns='Year', values='Daily_Return', aggfunc='mean')
        
        # Reindex both to standard month order
        month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        if not call_returns.empty:
            call_returns = call_returns.reindex(month_order)
        if not put_returns.empty:
            put_returns = put_returns.reindex(month_order)
        
        # Create superimposed heatmaps with fading effect
        if not call_returns.empty and not put_returns.empty:
            # Align the dataframes to have same index/columns
            combined_index = call_returns.index.union(put_returns.index)
            combined_columns = call_returns.columns.union(put_returns.columns)
            
            call_aligned = call_returns.reindex(index=combined_index, columns=combined_columns)
            put_aligned = put_returns.reindex(index=combined_index, columns=combined_columns)
            
            # Create base plot with call data
            call_alpha = call_ratio
            put_alpha = put_ratio
            
            if call_alpha > 0.01:  # Only plot if not fully transparent
                ax = sns.heatmap(call_aligned, annot=True, fmt=".2f", cmap='Blues', 
                               center=0, alpha=call_alpha, cbar=False)
            else:
                ax = plt.gca()
            
            # Overlay put data
            if put_alpha > 0.01:  # Only plot if not fully transparent
                sns.heatmap(put_aligned, annot=True, fmt=".2f", cmap='Reds',
                           center=0, alpha=put_alpha, cbar=True, ax=ax)
            
        elif not call_returns.empty:
            call_alpha = max(call_ratio, 0.2)
            sns.heatmap(call_returns, annot=True, fmt=".2f", cmap='Blues', 
                       center=0, alpha=call_alpha)
        elif not put_returns.empty:
            put_alpha = max(put_ratio, 0.2)
            sns.heatmap(put_returns, annot=True, fmt=".2f", cmap='Reds',
                       center=0, alpha=put_alpha)
        
        plt.title(f'Daily Returns Heatmap - Superimposed Fade Effect\\n({title_suffix})\\nCall α={call_ratio:.1f}, Put α={put_ratio:.1f}')
        plt.xlabel('Year')
        plt.ylabel('Month')
        plt.tight_layout()
        plt.show()
    
    def create_correlation_matrix(self, df_clean: pd.DataFrame, title_suffix: str, call_ratio: float, put_ratio: float):
        """Create correlation matrix heatmap with superimposed fading effect."""
        if self.df_call.empty and self.df_put.empty:
            print("❌ No data available for correlation matrix")
            return
            
        plt.figure(figsize=(12, 9))
        
        option_metrics = ['Open', 'High', 'Low', 'Close', 'No. of contracts', 'Turnover * in   ₹ Lakhs', 'Open Int']
        
        # Add optional metrics if available
        if not self.df_call.empty and 'Change in OI' in self.df_call.columns:
            option_metrics.append('Change in OI')
        if not self.df_call.empty and 'Strike Price' in self.df_call.columns:
            option_metrics.append('Strike Price')
        
        # Calculate correlation matrices for both call and put options
        call_corr = pd.DataFrame()
        put_corr = pd.DataFrame()
        
        if not self.df_call.empty:
            # Add daily return for calls
            df_call_copy = self.df_call.copy()
            if 'Strike Price' in df_call_copy.columns:
                df_call_copy['Daily_Return'] = df_call_copy.groupby(['Expiry', 'Strike Price'])['Close'].pct_change() * 100
            else:
                df_call_copy['Daily_Return'] = df_call_copy.groupby('Expiry')['Close'].pct_change() * 100
            
            available_call_metrics = [col for col in option_metrics + ['Daily_Return'] if col in df_call_copy.columns]
            call_corr = df_call_copy[available_call_metrics].corr()
        
        if not self.df_put.empty:
            # Add daily return for puts  
            df_put_copy = self.df_put.copy()
            if 'Strike Price' in df_put_copy.columns:
                df_put_copy['Daily_Return'] = df_put_copy.groupby(['Expiry', 'Strike Price'])['Close'].pct_change() * 100
            else:
                df_put_copy['Daily_Return'] = df_put_copy.groupby('Expiry')['Close'].pct_change() * 100
                
            available_put_metrics = [col for col in option_metrics + ['Daily_Return'] if col in df_put_copy.columns]
            put_corr = df_put_copy[available_put_metrics].corr()
        
        # Create superimposed correlation heatmaps
        if not call_corr.empty and not put_corr.empty:
            # Ensure both matrices have same dimensions
            common_cols = call_corr.columns.intersection(put_corr.columns)
            call_aligned = call_corr.loc[common_cols, common_cols]
            put_aligned = put_corr.loc[common_cols, common_cols]
            
            # Calculate alpha values
            call_alpha = call_ratio
            put_alpha = put_ratio
            
            # Create base plot with call correlations
            if call_alpha > 0.01:  # Only plot if not fully transparent
                ax = sns.heatmap(call_aligned, annot=True, cmap='Blues', center=0, fmt='.2f', 
                               alpha=call_alpha, cbar=False, square=True)
            else:
                ax = plt.gca()
            
            # Overlay put correlations
            if put_alpha > 0.01:  # Only plot if not fully transparent
                sns.heatmap(put_aligned, annot=True, cmap='Reds', center=0, fmt='.2f',
                           alpha=put_alpha, cbar=True, ax=ax, square=True)
            
        elif not call_corr.empty:
            call_alpha = max(call_ratio, 0.2)
            sns.heatmap(call_corr, annot=True, cmap='Blues', center=0, fmt='.2f', 
                       alpha=call_alpha, square=True)
        elif not put_corr.empty:
            put_alpha = max(put_ratio, 0.2)
            sns.heatmap(put_corr, annot=True, cmap='Reds', center=0, fmt='.2f',
                       alpha=put_alpha, square=True)
        
        plt.title(f'Correlation Matrix - Superimposed Fade Effect\\n({title_suffix})\\nCall α={call_ratio:.1f}, Put α={put_ratio:.1f}')
        plt.tight_layout()
        plt.show()
    
    def create_timeline_plot(self, df_clean: pd.DataFrame, title_suffix: str, call_ratio: float, put_ratio: float):
        """Create volume and open interest timeline with superimposed fading effect."""
        if self.df_call.empty and self.df_put.empty:
            print("❌ No data available for timeline plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Process Call Options Data
        call_daily_data = pd.DataFrame()
        if not self.df_call.empty:
            df_call_copy = self.df_call.copy()
            df_call_copy['Contract_Month'] = df_call_copy['Expiry'].dt.strftime('%b-%Y')
            most_active_call = df_call_copy.groupby('Contract_Month')['No. of contracts'].sum().idxmax()
            active_call_data = df_call_copy[df_call_copy['Contract_Month'] == most_active_call]
            call_daily_data = active_call_data.groupby('Date').agg({
                'Open Int': 'sum',
                'No. of contracts': 'sum'
            }).reset_index()
        
        # Process Put Options Data
        put_daily_data = pd.DataFrame()
        if not self.df_put.empty:
            df_put_copy = self.df_put.copy()
            df_put_copy['Contract_Month'] = df_put_copy['Expiry'].dt.strftime('%b-%Y')
            most_active_put = df_put_copy.groupby('Contract_Month')['No. of contracts'].sum().idxmax()
            active_put_data = df_put_copy[df_put_copy['Contract_Month'] == most_active_put]
            put_daily_data = active_put_data.groupby('Date').agg({
                'Open Int': 'sum',
                'No. of contracts': 'sum'
            }).reset_index()
        
        # Plot Open Interest with superimposed fading effect
        if not call_daily_data.empty:
            call_alpha = call_ratio
            if call_alpha > 0.01:
                ax1.plot(call_daily_data['Date'], call_daily_data['Open Int'], 
                        color='blue', label='Call Open Interest', linewidth=3, 
                        alpha=call_alpha)
        
        if not put_daily_data.empty:
            put_alpha = put_ratio
            if put_alpha > 0.01:
                ax1.plot(put_daily_data['Date'], put_daily_data['Open Int'],
                        color='red', label='Put Open Interest', linewidth=3,
                        alpha=put_alpha)
        
        ax1.set_ylabel('Open Interest', fontsize=12)
        ax1.set_title(f'Open Interest Over Time - Superimposed Fade Effect\\n({title_suffix})\\nCall α={call_ratio:.1f}, Put α={put_ratio:.1f}', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Volume (Contracts) with superimposed fading effect
        if not call_daily_data.empty:
            call_alpha = call_ratio
            if call_alpha > 0.01:
                ax2.plot(call_daily_data['Date'], call_daily_data['No. of contracts'],
                        color='blue', label='Call Volume', linewidth=3,
                        alpha=call_alpha)
        
        if not put_daily_data.empty:
            put_alpha = put_ratio
            if put_alpha > 0.01:
                ax2.plot(put_daily_data['Date'], put_daily_data['No. of contracts'],
                        color='red', label='Put Volume', linewidth=3, 
                        alpha=put_alpha)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Contracts (Volume)', fontsize=12)
        ax2.set_title('Volume Over Time - Superimposed Fade Effect', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_strike_distribution(self, df_clean: pd.DataFrame, title_suffix: str, call_ratio: float, put_ratio: float):
        """Create strike price distribution plots with superimposed fading effect."""
        if ((self.df_call.empty or 'Strike Price' not in self.df_call.columns) and 
            (self.df_put.empty or 'Strike Price' not in self.df_put.columns)):
            print("❌ No Strike Price data available for distribution plots")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Volume Distribution by Strike Price
        call_vol_dist = pd.Series()
        put_vol_dist = pd.Series()
        
        if not self.df_call.empty and 'Strike Price' in self.df_call.columns:
            call_vol_dist = self.df_call.groupby('Strike Price')['No. of contracts'].sum().sort_index()
        if not self.df_put.empty and 'Strike Price' in self.df_put.columns:
            put_vol_dist = self.df_put.groupby('Strike Price')['No. of contracts'].sum().sort_index()
        
        # Plot volume distributions with superimposed fading effect
        if not call_vol_dist.empty:
            call_alpha = call_ratio
            if call_alpha > 0.01:
                ax1.bar(range(len(call_vol_dist)), call_vol_dist.values, 
                       alpha=call_alpha, color='blue', 
                       label=f'Call Volume (α={call_ratio:.1f})', width=0.8)
        
        if not put_vol_dist.empty:
            put_alpha = put_ratio
            # Align put data to same x-axis as call data for superimposition
            if not call_vol_dist.empty:
                put_aligned = put_vol_dist.reindex(call_vol_dist.index, fill_value=0)
                if put_alpha > 0.01:
                    ax1.bar(range(len(put_aligned)), put_aligned.values,
                           alpha=put_alpha, color='red',
                           label=f'Put Volume (α={put_ratio:.1f})', width=0.8)
            else:
                if put_alpha > 0.01:
                    ax1.bar(range(len(put_vol_dist)), put_vol_dist.values,
                           alpha=put_alpha, color='red',
                           label=f'Put Volume (α={put_ratio:.1f})', width=0.8)
        
        ax1.set_xlabel('Strike Price Index')
        ax1.set_ylabel('Total Volume')
        ax1.set_title(f'Volume Distribution - Superimposed Fade\\n({title_suffix})')
        ax1.legend()
        
        # Set x-axis labels for volume plot
        if not call_vol_dist.empty:
            ax1.set_xticks(range(0, len(call_vol_dist), max(1, len(call_vol_dist)//10)))
            ax1.set_xticklabels([f"₹{int(strike)}" for strike in call_vol_dist.index[::max(1, len(call_vol_dist)//10)]], 
                               rotation=45)
        elif not put_vol_dist.empty:
            ax1.set_xticks(range(0, len(put_vol_dist), max(1, len(put_vol_dist)//10)))
            ax1.set_xticklabels([f"₹{int(strike)}" for strike in put_vol_dist.index[::max(1, len(put_vol_dist)//10)]], 
                               rotation=45)
        
        # Open Interest Distribution by Strike Price
        call_oi_dist = pd.Series()
        put_oi_dist = pd.Series()
        
        if not self.df_call.empty and 'Strike Price' in self.df_call.columns:
            call_oi_dist = self.df_call.groupby('Strike Price')['Open Int'].sum().sort_index()
        if not self.df_put.empty and 'Strike Price' in self.df_put.columns:
            put_oi_dist = self.df_put.groupby('Strike Price')['Open Int'].sum().sort_index()
        
        # Plot OI distributions with superimposed fading effect
        if not call_oi_dist.empty:
            call_alpha = call_ratio
            if call_alpha > 0.01:
                ax2.bar(range(len(call_oi_dist)), call_oi_dist.values,
                       alpha=call_alpha, color='blue',
                       label=f'Call OI (α={call_ratio:.1f})', width=0.8)
        
        if not put_oi_dist.empty:
            put_alpha = put_ratio
            # Align put data to same x-axis as call data for superimposition
            if not call_oi_dist.empty:
                put_oi_aligned = put_oi_dist.reindex(call_oi_dist.index, fill_value=0)
                if put_alpha > 0.01:
                    ax2.bar(range(len(put_oi_aligned)), put_oi_aligned.values,
                           alpha=put_alpha, color='red', 
                           label=f'Put OI (α={put_ratio:.1f})', width=0.8)
            else:
                if put_alpha > 0.01:
                    ax2.bar(range(len(put_oi_dist)), put_oi_dist.values,
                           alpha=put_alpha, color='red',
                           label=f'Put OI (α={put_ratio:.1f})', width=0.8)
        
        ax2.set_xlabel('Strike Price Index')
        ax2.set_ylabel('Total Open Interest')
        ax2.set_title(f'Open Interest Distribution - Superimposed Fade\\n({title_suffix})')
        ax2.legend()
        
        # Set x-axis labels for OI plot
        if not call_oi_dist.empty:
            ax2.set_xticks(range(0, len(call_oi_dist), max(1, len(call_oi_dist)//10)))
            ax2.set_xticklabels([f"₹{int(strike)}" for strike in call_oi_dist.index[::max(1, len(call_oi_dist)//10)]], 
                               rotation=45)
        elif not put_oi_dist.empty:
            ax2.set_xticks(range(0, len(put_oi_dist), max(1, len(put_oi_dist)//10)))
            ax2.set_xticklabels([f"₹{int(strike)}" for strike in put_oi_dist.index[::max(1, len(put_oi_dist)//10)]], 
                               rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_visualizations(self):
        """Generate selected visualizations with smooth color transitions."""
        slider_value = self.option_type_slider.value
        viz_type = self.viz_type_selector.value
        
        with self.plot_output:
            clear_output(wait=True)
            
            # Get data with blend ratios
            df_clean, title_suffix, call_ratio, put_ratio = self.get_data_for_analysis(slider_value)
            
            if df_clean.empty:
                print(f"❌ No data available for {title_suffix}")
                return
            
            print(f"📊 Analyzing: {title_suffix}")
            print(f"📈 Records: {len(df_clean):,}")
            print("✅ All '-' values already replaced with NaN during data loading")
            print(f"🎛️ Slider Position: {slider_value:.1f} (Call: {call_ratio:.1f}, Put: {put_ratio:.1f})")
            print("-" * 60)
            
            # Generate selected visualizations with color transitions
            if viz_type == 'scatter' or viz_type == 'all':
                self.create_scatter_plot(df_clean, title_suffix, call_ratio, put_ratio)
                
            if viz_type == 'heatmap' or viz_type == 'all':
                self.create_returns_heatmap(df_clean, title_suffix, call_ratio, put_ratio)
                
            if viz_type == 'correlation' or viz_type == 'all':
                self.create_correlation_matrix(df_clean, title_suffix, call_ratio, put_ratio)
                
            if viz_type == 'timeline' or viz_type == 'all':
                self.create_timeline_plot(df_clean, title_suffix, call_ratio, put_ratio)
                
            if viz_type == 'strike' or viz_type == 'all':
                self.create_strike_distribution(df_clean, title_suffix, call_ratio, put_ratio)
            
            print(f"\\n✅ Visualization completed for {title_suffix}!")
            print(f"📊 Available DataFrames:")
            print(f"   📞 df_call: {len(self.df_call)} Call options records")
            print(f"   📉 df_put: {len(self.df_put)} Put options records")
    
    def _on_generate_click(self, b):
        """Handle generate button click."""
        self.generate_visualizations()
    
    def display_interface(self):
        """Display the interactive visualization interface."""
        print("📊 INTERACTIVE OPTIONS DATA VISUALIZATION")
        print("=" * 45)
        
        # Check if we have options data
        if self.df_call.empty and self.df_put.empty:
            print("❌ No options data available for visualization")
            return
        else:
            print(f"📞 Call Options Available: {len(self.df_call):,} records")
            print(f"📉 Put Options Available: {len(self.df_put):,} records")
        
        # Arrange controls with slider labels  
        slider_container = VBox([
            self.option_type_slider,
            self.slider_labels
        ])
        
        controls_row = HBox([
            slider_container,
            self.viz_type_selector,
            self.generate_button
        ])
        
        print("\\n🎛️ INTERACTIVE CONTROLS:")
        print("-" * 25)
        print("🎚️ Use the slider to smoothly transition between Call and Put options")
        print("   • 0.0 = Pure Call Options (CE)")
        print("   • 0.5 = Mixed Analysis") 
        print("   • 1.0 = Pure Put Options (PE)")
        display(controls_row)
        display(self.plot_output)
        
        # Generate initial visualization
        print("\\n📈 Initial Analysis (Call Options, All Visualizations):")
        self.generate_visualizations()


def create_interactive_options_visualizer(df_call: pd.DataFrame, df_put: pd.DataFrame) -> OptionsVisualizer:
    """
    Create and display an interactive options data visualizer.
    
    This function creates a comprehensive interactive visualization interface for options data
    with superimposed fading effects and smooth transitions between Call and Put analysis.
    
    Args:
        df_call (pd.DataFrame): Call options data with columns like 'Date', 'Expiry', 'Close', etc.
        df_put (pd.DataFrame): Put options data with columns like 'Date', 'Expiry', 'Close', etc.
    
    Returns:
        OptionsVisualizer: The visualizer instance for further customization if needed
        
    Example:
        ```python
        # Load your options data
        df_call, df_put, options_merged = load_banknifty_options_data(data_path)
        
        # Create the interactive visualizer
        visualizer = create_interactive_options_visualizer(df_call, df_put)
        ```
    """
    visualizer = OptionsVisualizer(df_call, df_put)
    visualizer.display_interface()
    return visualizer


def get_visualization_summary(df_call: pd.DataFrame, df_put: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of visualization capabilities for the given options data.
    
    Args:
        df_call (pd.DataFrame): Call options data
        df_put (pd.DataFrame): Put options data
        
    Returns:
        Dict[str, Any]: Summary of available visualization features
    """
    summary = {
        'data_availability': {
            'call_options': not df_call.empty,
            'put_options': not df_put.empty,
            'call_records': len(df_call),
            'put_records': len(df_put)
        },
        'available_visualizations': [
            'Volume vs Price Scatter Plot',
            'Daily Returns Heatmap',
            'Correlation Matrix',
            'Volume & Open Interest Timeline',
            'Strike Price Distribution'
        ],
        'interactive_features': [
            'Call/Put Blend Slider',
            'Visualization Type Selection',
            'Superimposed Fading Effects',
            'Real-time Plot Updates'
        ],
        'required_columns': {
            'basic': ['Date', 'Expiry', 'Close', 'Open', 'High', 'Low'],
            'volume': ['No. of contracts', 'Open Int'],
            'optional': ['Strike Price', 'Change in OI', 'Turnover * in   ₹ Lakhs']
        }
    }
    
    # Check column availability
    summary['column_availability'] = {}
    if not df_call.empty:
        summary['column_availability']['call_columns'] = list(df_call.columns)
    if not df_put.empty:
        summary['column_availability']['put_columns'] = list(df_put.columns)
    
    return summary
