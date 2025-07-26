"""
Unit Tests for Options Visualizer Module

This module contains unit tests for the options_visualizer module to ensure
proper functionality of visualization components and data processing.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.options_visualizer import OptionsVisualizer, create_interactive_options_visualizer, get_visualization_summary


class TestOptionsVisualizer(unittest.TestCase):
    """Test cases for the Options Visualizer module."""
    
    def setUp(self):
        """Set up test data for all test methods."""
        # Create sample call options data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        expiry_date = datetime(2024, 1, 25)
        
        self.df_call = pd.DataFrame({
            'Date': np.tile(dates, 3)[:90],  # Repeat dates for multiple strikes
            'Expiry': expiry_date,
            'Strike Price': np.repeat([50000, 51000, 52000], 30),
            'Open': np.random.uniform(100, 200, 90),
            'High': np.random.uniform(200, 300, 90),
            'Low': np.random.uniform(50, 100, 90),
            'Close': np.random.uniform(150, 250, 90),
            'No. of contracts': np.random.randint(1000, 10000, 90),
            'Open Int': np.random.randint(5000, 50000, 90),
            'Turnover * in   ₹ Lakhs': np.random.uniform(1000, 5000, 90),
            'Change in OI': np.random.randint(-1000, 1000, 90)
        })
        
        # Create sample put options data
        self.df_put = pd.DataFrame({
            'Date': np.tile(dates, 3)[:90],
            'Expiry': expiry_date,
            'Strike Price': np.repeat([50000, 51000, 52000], 30),
            'Open': np.random.uniform(80, 180, 90),
            'High': np.random.uniform(180, 280, 90),
            'Low': np.random.uniform(30, 80, 90),
            'Close': np.random.uniform(120, 220, 90),
            'No. of contracts': np.random.randint(800, 8000, 90),
            'Open Int': np.random.randint(4000, 40000, 90),
            'Turnover * in   ₹ Lakhs': np.random.uniform(800, 4000, 90),
            'Change in OI': np.random.randint(-800, 800, 90)
        })
        
        # Create empty dataframes for edge case testing
        self.df_empty = pd.DataFrame()
    
    def test_visualizer_initialization(self):
        """Test OptionsVisualizer initialization with valid data."""
        visualizer = OptionsVisualizer(self.df_call, self.df_put)
        
        # Check that data is properly stored
        self.assertFalse(visualizer.df_call.empty)
        self.assertFalse(visualizer.df_put.empty)
        self.assertEqual(len(visualizer.df_call), 90)
        self.assertEqual(len(visualizer.df_put), 90)
        
        # Check that widgets are initialized
        self.assertIsNotNone(visualizer.option_type_slider)
        self.assertIsNotNone(visualizer.viz_type_selector)
        self.assertIsNotNone(visualizer.generate_button)
        self.assertIsNotNone(visualizer.plot_output)
    
    def test_visualizer_initialization_with_empty_data(self):
        """Test OptionsVisualizer initialization with empty data."""
        visualizer = OptionsVisualizer(self.df_empty, self.df_empty)
        
        # Check that empty dataframes are handled properly
        self.assertTrue(visualizer.df_call.empty)
        self.assertTrue(visualizer.df_put.empty)
        
        # Widgets should still be initialized
        self.assertIsNotNone(visualizer.option_type_slider)
        self.assertIsNotNone(visualizer.viz_type_selector)
    
    def test_get_data_for_analysis(self):
        """Test the data blending functionality."""
        visualizer = OptionsVisualizer(self.df_call, self.df_put)
        
        # Test pure call analysis (slider = 0.0)
        df_clean, title_suffix, call_ratio, put_ratio = visualizer.get_data_for_analysis(0.0)
        self.assertFalse(df_clean.empty)
        self.assertEqual(call_ratio, 1.0)
        self.assertEqual(put_ratio, 0.0)
        self.assertIn('Call Options', title_suffix)
        
        # Test pure put analysis (slider = 1.0)
        df_clean, title_suffix, call_ratio, put_ratio = visualizer.get_data_for_analysis(1.0)
        self.assertFalse(df_clean.empty)
        self.assertEqual(call_ratio, 0.0)
        self.assertEqual(put_ratio, 1.0)
        self.assertIn('Put Options', title_suffix)
        
        # Test mixed analysis (slider = 0.5)
        df_clean, title_suffix, call_ratio, put_ratio = visualizer.get_data_for_analysis(0.5)
        self.assertFalse(df_clean.empty)
        self.assertEqual(call_ratio, 0.5)
        self.assertEqual(put_ratio, 0.5)
        self.assertIn('Mixed Options', title_suffix)
    
    def test_get_data_for_analysis_with_empty_data(self):
        """Test data blending with empty datasets."""
        visualizer = OptionsVisualizer(self.df_empty, self.df_empty)
        
        df_clean, title_suffix, call_ratio, put_ratio = visualizer.get_data_for_analysis(0.5)
        self.assertTrue(df_clean.empty)
        self.assertEqual(title_suffix, 'No Data')
        self.assertEqual(call_ratio, 1.0)
        self.assertEqual(put_ratio, 0.0)
    
    def test_create_interactive_options_visualizer_function(self):
        """Test the convenience function for creating visualizers."""
        visualizer = create_interactive_options_visualizer(self.df_call, self.df_put)
        
        # Should return an OptionsVisualizer instance
        self.assertIsInstance(visualizer, OptionsVisualizer)
        self.assertFalse(visualizer.df_call.empty)
        self.assertFalse(visualizer.df_put.empty)
    
    def test_get_visualization_summary(self):
        """Test the visualization summary function."""
        summary = get_visualization_summary(self.df_call, self.df_put)
        
        # Check summary structure
        self.assertIn('data_availability', summary)
        self.assertIn('available_visualizations', summary)
        self.assertIn('interactive_features', summary)
        self.assertIn('required_columns', summary)
        self.assertIn('column_availability', summary)
        
        # Check data availability
        data_avail = summary['data_availability']
        self.assertTrue(data_avail['call_options'])
        self.assertTrue(data_avail['put_options'])
        self.assertEqual(data_avail['call_records'], 90)
        self.assertEqual(data_avail['put_records'], 90)
        
        # Check available visualizations
        visualizations = summary['available_visualizations']
        expected_viz = [
            'Volume vs Price Scatter Plot',
            'Daily Returns Heatmap',
            'Correlation Matrix',
            'Volume & Open Interest Timeline',
            'Strike Price Distribution'
        ]
        for viz in expected_viz:
            self.assertIn(viz, visualizations)
    
    def test_get_visualization_summary_with_empty_data(self):
        """Test visualization summary with empty data."""
        summary = get_visualization_summary(self.df_empty, self.df_empty)
        
        data_avail = summary['data_availability']
        self.assertFalse(data_avail['call_options'])
        self.assertFalse(data_avail['put_options'])
        self.assertEqual(data_avail['call_records'], 0)
        self.assertEqual(data_avail['put_records'], 0)
    
    def test_widget_properties(self):
        """Test widget configuration and properties."""
        visualizer = OptionsVisualizer(self.df_call, self.df_put)
        
        # Test slider properties
        slider = visualizer.option_type_slider
        self.assertEqual(slider.min, 0.0)
        self.assertEqual(slider.max, 1.0)
        self.assertEqual(slider.step, 0.1)
        self.assertEqual(slider.value, 0.0)
        
        # Test dropdown options
        dropdown = visualizer.viz_type_selector
        expected_options = ['scatter', 'heatmap', 'correlation', 'timeline', 'strike', 'all']
        actual_options = [opt[1] for opt in dropdown.options]
        for option in expected_options:
            self.assertIn(option, actual_options)
    
    def test_data_integrity(self):
        """Test that data is properly copied and not modified."""
        original_call_len = len(self.df_call)
        original_put_len = len(self.df_put)
        
        visualizer = OptionsVisualizer(self.df_call, self.df_put)
        
        # Original data should remain unchanged
        self.assertEqual(len(self.df_call), original_call_len)
        self.assertEqual(len(self.df_put), original_put_len)
        
        # Visualizer should have copies
        self.assertEqual(len(visualizer.df_call), original_call_len)
        self.assertEqual(len(visualizer.df_put), original_put_len)
        
        # Modifying visualizer data shouldn't affect original
        visualizer.df_call.loc[0, 'Close'] = 999999
        self.assertNotEqual(self.df_call.loc[0, 'Close'], 999999)


class TestVisualizationMethods(unittest.TestCase):
    """Test the individual visualization methods."""
    
    def setUp(self):
        """Set up test data for visualization method tests."""
        # Create minimal test data
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        
        self.df_call_small = pd.DataFrame({
            'Date': dates,
            'Expiry': datetime(2024, 1, 25),
            'Strike Price': 50000,
            'Close': np.random.uniform(150, 250, 10),
            'No. of contracts': np.random.randint(1000, 5000, 10),
            'Open Int': np.random.randint(5000, 25000, 10),
        })
        
        self.df_put_small = pd.DataFrame({
            'Date': dates,
            'Expiry': datetime(2024, 1, 25),
            'Strike Price': 50000,
            'Close': np.random.uniform(120, 220, 10),
            'No. of contracts': np.random.randint(800, 4000, 10),
            'Open Int': np.random.randint(4000, 20000, 10),
        })
    
    def test_visualization_methods_exist(self):
        """Test that all visualization methods exist and are callable."""
        visualizer = OptionsVisualizer(self.df_call_small, self.df_put_small)
        
        # Check that all visualization methods exist
        self.assertTrue(hasattr(visualizer, 'create_scatter_plot'))
        self.assertTrue(hasattr(visualizer, 'create_returns_heatmap'))
        self.assertTrue(hasattr(visualizer, 'create_correlation_matrix'))
        self.assertTrue(hasattr(visualizer, 'create_timeline_plot'))
        self.assertTrue(hasattr(visualizer, 'create_strike_distribution'))
        
        # Check that methods are callable
        self.assertTrue(callable(visualizer.create_scatter_plot))
        self.assertTrue(callable(visualizer.create_returns_heatmap))
        self.assertTrue(callable(visualizer.create_correlation_matrix))
        self.assertTrue(callable(visualizer.create_timeline_plot))
        self.assertTrue(callable(visualizer.create_strike_distribution))


if __name__ == '__main__':
    # Run the tests
    print("🧪 Running Options Visualizer Module Tests")
    print("=" * 45)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestOptionsVisualizer))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizationMethods))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"   ✅ Tests Run: {result.testsRun}")
    print(f"   ❌ Failures: {len(result.failures)}")
    print(f"   ⚠️  Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"\n🎉 All tests passed! The options_visualizer module is working correctly.")
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")
        
    print(f"\n🔧 Module Location: src/utils/options_visualizer.py")
    print(f"📖 Usage: from utils.options_visualizer import create_interactive_options_visualizer")
