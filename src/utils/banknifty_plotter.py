"""
Bank Nifty Interactive Plotter Module

This module provides interactive plotting functionality for Bank Nifty data
using Plotly and IPython widgets for Jupyter notebooks.
"""

import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from ipywidgets import widgets, VBox, HBox, Button, Output
from IPython.display import display, clear_output
from typing import Optional, List


class BankNiftyInteractivePlotter:
    """
    Interactive plotter class for Bank Nifty data with customizable controls.
    
    This class creates an interactive plotting interface with date pickers,
    price type selection, and automatic statistics display.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the interactive plotter with Bank Nifty data.
        
        Args:
            data (pd.DataFrame): Bank Nifty data with Date and OHLC columns
        """
        self.data = data.copy() if not data.empty else pd.DataFrame()
        self.plot_area = Output()
        self._setup_widgets()
    
    def _setup_widgets(self):
        """Setup all interactive widgets for the plotter."""
        if self.data.empty or 'Date' not in self.data.columns:
            self.widgets_created = False
            return
        
        # Get available price columns
        self.price_columns = [col for col in ['Open', 'High', 'Low', 'Close'] if col in self.data.columns]
        
        if not self.price_columns:
            self.widgets_created = False
            return
        
        # Create control widgets
        self.price_selector = widgets.Dropdown(
            options=self.price_columns,
            value='Close' if 'Close' in self.price_columns else self.price_columns[0],
            description='Price Type:',
            style={'description_width': '80px'},
            layout={'width': '150px'}
        )
        
        self.start_date_picker = widgets.DatePicker(
            description='From:',
            value=self.data['Date'].min().date(),
            style={'description_width': '50px'},
            layout={'width': '160px'}
        )
        
        self.end_date_picker = widgets.DatePicker(
            description='To:',
            value=self.data['Date'].max().date(),
            style={'description_width': '50px'},
            layout={'width': '160px'}
        )
        
        self.plot_button = Button(
            description='📈 Plot',
            button_style='success',
            layout={'width': '80px'}
        )
        
        # Connect button click handler
        self.plot_button.on_click(self._on_plot_button_clicked)
        
        self.widgets_created = True
    
    def _on_plot_button_clicked(self, b):
        """Handle plot button click event."""
        self._make_single_plot()
    
    def _make_single_plot(self):
        """Generate a single plot based on current widget values."""
        if not self.widgets_created:
            print("❌ Widgets not properly initialized")
            return
        
        # Get current widget values
        price_type = self.price_selector.value
        start_date = datetime.combine(self.start_date_picker.value, datetime.min.time())
        end_date = datetime.combine(self.end_date_picker.value, datetime.min.time())
        
        # Clear previous plot
        with self.plot_area:
            clear_output(wait=True)
            
            # Filter data by date range
            mask = (self.data['Date'] >= start_date) & (self.data['Date'] <= end_date)
            filtered_data = self.data.loc[mask].copy()
            
            if filtered_data.empty:
                print("❌ No data for selected date range")
                return
            
            # Create single plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data[price_type],
                mode='lines+markers',
                name=f'Bank Nifty {price_type}',
                line=dict(color='#2E86C1', width=2.5),
                marker=dict(size=3),
                hovertemplate=f'<b>%{{x}}</b><br>{price_type}: ₹%{{y:,.0f}}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Bank Nifty {price_type} Price Analysis',
                xaxis_title='Date',
                yaxis_title=f'{price_type} Price (₹)',
                height=450,
                template='plotly_white',
                showlegend=False,
                hovermode='x unified'
            )
            
            fig.show()
            
            # Display statistics
            stats = filtered_data[price_type].describe()
            print(f"\n📊 {price_type} Analysis Summary:")
            print(f"   📅 Period: {start_date:%d-%b-%Y} to {end_date:%d-%b-%Y}")
            print(f"   📈 Data Points: {len(filtered_data):,}")
            print(f"   💰 Price Range: ₹{stats['min']:,.0f} - ₹{stats['max']:,.0f}")
            print(f"   📊 Average Price: ₹{stats['mean']:,.0f}")
            print(f"   📈 Standard Dev: ₹{stats['std']:,.0f}")
    
    def display_interface(self):
        """Display the complete interactive interface."""
        print("📊 BANK NIFTY INTERACTIVE PLOTTER")
        print("=" * 40)
        
        if self.data.empty:
            print("❌ No Bank Nifty data available")
            return
        
        if 'Date' not in self.data.columns:
            print("❌ Date column not found in data")
            return
        
        if not self.widgets_created:
            print("❌ No price columns found")
            return
        
        print(f"✅ Available: {', '.join(self.price_columns)}")
        print(f"📅 Range: {self.data['Date'].min():%d-%b-%Y} to {self.data['Date'].max():%d-%b-%Y}")
        print(f"📊 Records: {len(self.data):,}")
        
        # Arrange controls in horizontal layout
        controls = HBox([
            self.price_selector,
            self.start_date_picker,
            self.end_date_picker,
            self.plot_button
        ])
        
        # Display interface
        print("\n🎛️ Controls:")
        display(controls)
        display(self.plot_area)
        
        # Create initial plot automatically
        print("\n📈 Initial plot (Close price, full range):")
        self._make_single_plot()
    
    def plot_specific_range(self, start_date: str, end_date: str, price_type: str = 'Close'):
        """
        Plot data for a specific date range and price type programmatically.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            price_type (str): Price column to plot ('Open', 'High', 'Low', 'Close')
        """
        if self.data.empty:
            print("❌ No data available for plotting")
            return
        
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if price_type not in self.data.columns:
                print(f"❌ Price type '{price_type}' not available. Available: {list(self.data.columns)}")
                return
            
            # Filter data
            mask = (self.data['Date'] >= start_dt) & (self.data['Date'] <= end_dt)
            filtered_data = self.data.loc[mask].copy()
            
            if filtered_data.empty:
                print(f"❌ No data available for range {start_date} to {end_date}")
                return
            
            # Create plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data[price_type],
                mode='lines+markers',
                name=f'Bank Nifty {price_type}',
                line=dict(color='#2E86C1', width=2.5),
                marker=dict(size=3),
                hovertemplate=f'<b>%{{x}}</b><br>{price_type}: ₹%{{y:,.0f}}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Bank Nifty {price_type} Price Analysis ({start_date} to {end_date})',
                xaxis_title='Date',
                yaxis_title=f'{price_type} Price (₹)',
                height=450,
                template='plotly_white',
                showlegend=False,
                hovermode='x unified'
            )
            
            fig.show()
            
            # Display statistics
            stats = filtered_data[price_type].describe()
            print(f"\n📊 {price_type} Analysis Summary:")
            print(f"   📅 Period: {start_date} to {end_date}")
            print(f"   📈 Data Points: {len(filtered_data):,}")
            print(f"   💰 Price Range: ₹{stats['min']:,.0f} - ₹{stats['max']:,.0f}")
            print(f"   📊 Average Price: ₹{stats['mean']:,.0f}")
            print(f"   📈 Standard Dev: ₹{stats['std']:,.0f}")
            
        except ValueError as e:
            print(f"❌ Date format error: {e}")
            print("Please use 'YYYY-MM-DD' format for dates")
    
    def get_data_summary(self) -> dict:
        """
        Get summary information about the loaded data.
        
        Returns:
            dict: Summary statistics and data information
        """
        if self.data.empty:
            return {"error": "No data available"}
        
        summary = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "price_columns": self.price_columns,
            "records": len(self.data)
        }
        
        if 'Date' in self.data.columns:
            summary["date_range"] = {
                "start": self.data['Date'].min(),
                "end": self.data['Date'].max(),
                "days": (self.data['Date'].max() - self.data['Date'].min()).days
            }
        
        # Price statistics for available price columns
        if self.price_columns:
            summary["price_stats"] = {}
            for col in self.price_columns:
                summary["price_stats"][col] = {
                    "mean": self.data[col].mean(),
                    "std": self.data[col].std(),
                    "min": self.data[col].min(),
                    "max": self.data[col].max()
                }
        
        return summary


def create_banknifty_interactive_plotter(data: pd.DataFrame) -> BankNiftyInteractivePlotter:
    """
    Factory function to create a Bank Nifty interactive plotter.
    
    Args:
        data (pd.DataFrame): Bank Nifty data with Date and OHLC columns
        
    Returns:
        BankNiftyInteractivePlotter: Configured plotter instance
    """
    return BankNiftyInteractivePlotter(data)


def display_banknifty_plotter(data: pd.DataFrame) -> BankNiftyInteractivePlotter:
    """
    Convenience function to create and immediately display a Bank Nifty plotter.
    
    Args:
        data (pd.DataFrame): Bank Nifty data with Date and OHLC columns
        
    Returns:
        BankNiftyInteractivePlotter: Plotter instance (for further use if needed)
    """
    plotter = create_banknifty_interactive_plotter(data)
    plotter.display_interface()
    return plotter


def plot_banknifty_simple(data: pd.DataFrame, price_type: str = 'Close', 
                         start_date: Optional[str] = None, end_date: Optional[str] = None):
    """
    Create a simple Bank Nifty plot without interactive controls.
    
    Args:
        data (pd.DataFrame): Bank Nifty data
        price_type (str): Price column to plot
        start_date (Optional[str]): Start date in 'YYYY-MM-DD' format
        end_date (Optional[str]): End date in 'YYYY-MM-DD' format
    """
    if data.empty:
        print("❌ No data available for plotting")
        return
    
    if price_type not in data.columns:
        print(f"❌ Price type '{price_type}' not available. Available: {list(data.columns)}")
        return
    
    plot_data = data.copy()
    
    # Filter by date range if provided
    if start_date or end_date:
        try:
            if start_date:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                plot_data = plot_data[plot_data['Date'] >= start_dt]
            if end_date:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                plot_data = plot_data[plot_data['Date'] <= end_dt]
        except ValueError as e:
            print(f"❌ Date format error: {e}")
            return
    
    if plot_data.empty:
        print("❌ No data available for the specified date range")
        return
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=plot_data[price_type],
        mode='lines',
        name=f'Bank Nifty {price_type}',
        line=dict(color='#2E86C1', width=2),
        hovertemplate=f'<b>%{{x}}</b><br>{price_type}: ₹%{{y:,.0f}}<extra></extra>'
    ))
    
    title_suffix = f" ({start_date} to {end_date})" if start_date or end_date else ""
    fig.update_layout(
        title=f'Bank Nifty {price_type} Price{title_suffix}',
        xaxis_title='Date',
        yaxis_title=f'{price_type} Price (₹)',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    fig.show()
    
    # Display basic statistics
    stats = plot_data[price_type].describe()
    print(f"\n📊 {price_type} Summary: Records: {len(plot_data):,}, Range: ₹{stats['min']:,.0f} - ₹{stats['max']:,.0f}, Avg: ₹{stats['mean']:,.0f}")
