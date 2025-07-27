"""
Market Data Validation Utilities

This module provides comprehensive validation functions for market data analysis,
including Bank Nifty spot data and options data validation.

Functions:
- validate_market_data: Main validation function for all market data
- display_validation_summary: Display validation results and next steps
- _validate_options_data: Helper for validating options data
- _analyze_expiry_data: Helper for analyzing expiry information
"""

import pandas as pd
from typing import Optional, Dict, Any


def validate_market_data(bank_nifty: Optional[pd.DataFrame] = None, 
                        df_call: Optional[pd.DataFrame] = None, 
                        df_put: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Comprehensive data validation function for market analysis.
    
    Args:
        bank_nifty (pd.DataFrame, optional): Bank Nifty spot data
        df_call (pd.DataFrame, optional): Call options data
        df_put (pd.DataFrame, optional): Put options data
        
    Returns:
        dict: Validation results with status and details
    """
    print("🔍 Enhanced Market Data Validation:")
    print("-" * 40)
    
    validation_results = {
        'all_valid': False,
        'data_status': {
            'bank_nifty': False,
            'df_call': False,
            'df_put': False
        },
        'validation_details': {},
        'expiry_analysis': {},
        'summary_stats': {}
    }
    
    # Validate Bank Nifty spot data
    if bank_nifty is not None and isinstance(bank_nifty, pd.DataFrame) and not bank_nifty.empty:
        required_columns = ['Date', 'Close']
        if all(col in bank_nifty.columns for col in required_columns):
            validation_results['data_status']['bank_nifty'] = True
            validation_results['validation_details']['bank_nifty'] = f"{len(bank_nifty):,} records ({bank_nifty['Date'].min():%d-%b-%Y} to {bank_nifty['Date'].max():%d-%b-%Y})"
            validation_results['summary_stats']['bank_nifty'] = {
                'records': len(bank_nifty),
                'date_range': {'start': bank_nifty['Date'].min(), 'end': bank_nifty['Date'].max()},
                'price_range': {'min': bank_nifty['Close'].min(), 'max': bank_nifty['Close'].max()},
                'columns': list(bank_nifty.columns)
            }
            print(f"✅ Bank Nifty Spot: {validation_results['validation_details']['bank_nifty']}")
        else:
            missing_cols = [col for col in required_columns if col not in bank_nifty.columns]
            validation_results['validation_details']['bank_nifty'] = f"Missing columns: {missing_cols}"
            print(f"⚠️ Bank Nifty: {validation_results['validation_details']['bank_nifty']}")
    else:
        validation_results['validation_details']['bank_nifty'] = "Not available or empty"
        print("❌ Bank Nifty: Not available or empty")
    
    # Validate Call Options data
    validation_results = _validate_options_data(
        df_call, 'df_call', 'Call', validation_results
    )
    
    # Validate Put Options data
    validation_results = _validate_options_data(
        df_put, 'df_put', 'Put', validation_results
    )
    
    # Analyze expiry information if options data is available
    if validation_results['data_status']['df_call'] or validation_results['data_status']['df_put']:
        validation_results = _analyze_expiry_data(df_call, df_put, validation_results)
    
    # Determine overall validity
    validation_results['all_valid'] = all(validation_results['data_status'].values())
    
    return validation_results


def _validate_options_data(options_df: Optional[pd.DataFrame], 
                          data_key: str, 
                          data_name: str, 
                          validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to validate options data (calls or puts).
    
    Args:
        options_df: Options DataFrame to validate
        data_key: Key for storing validation results
        data_name: Display name for the data type
        validation_results: Current validation results dictionary
        
    Returns:
        dict: Updated validation results
    """
    if options_df is not None and isinstance(options_df, pd.DataFrame) and not options_df.empty:
        # Check for either 'Strike' or 'Strike Price' columns
        has_strike = 'Strike' in options_df.columns or 'Strike Price' in options_df.columns
        required_columns = ['Date', 'Expiry', 'Close']
        
        if all(col in options_df.columns for col in required_columns) and has_strike:
            validation_results['data_status'][data_key] = True
            expiry_count = options_df['Expiry'].nunique()
            validation_results['validation_details'][data_key] = f"{len(options_df):,} records, {expiry_count} unique expiries"
            
            strike_col = 'Strike Price' if 'Strike Price' in options_df.columns else 'Strike'
            strike_min, strike_max = options_df[strike_col].min(), options_df[strike_col].max()
            
            validation_results['summary_stats'][data_key] = {
                'records': len(options_df),
                'expiry_count': expiry_count,
                'strike_column': strike_col,
                'strike_range': {'min': strike_min, 'max': strike_max},
                'columns': list(options_df.columns),
                'date_range': {'start': options_df['Date'].min(), 'end': options_df['Date'].max()}
            }
            
            print(f"✅ {data_name} Options: {validation_results['validation_details'][data_key]}")
            print(f"   📊 Strike column: '{strike_col}' | Strike range: ₹{strike_min:,.0f} - ₹{strike_max:,.0f}")
        else:
            missing_info = []
            if not all(col in options_df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in options_df.columns]
                missing_info.append(f"Missing columns: {missing_cols}")
            if not has_strike:
                missing_info.append("Missing Strike or Strike Price column")
            validation_results['validation_details'][data_key] = "; ".join(missing_info)
            print(f"⚠️ {data_name} Options: {validation_results['validation_details'][data_key]}")
            print(f"   📋 Available columns: {list(options_df.columns)}")
    else:
        validation_results['validation_details'][data_key] = "Not available or empty"
        print(f"❌ {data_name} Options: Not available or empty")
    
    return validation_results


def _analyze_expiry_data(df_call: Optional[pd.DataFrame], 
                        df_put: Optional[pd.DataFrame], 
                        validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to analyze expiry information.
    
    Args:
        df_call: Call options DataFrame
        df_put: Put options DataFrame
        validation_results: Current validation results dictionary
        
    Returns:
        dict: Updated validation results with expiry analysis
    """
    print(f"\n📅 EXPIRY DATES ANALYSIS:")
    print("-" * 25)
    
    all_expiries = set()
    if validation_results['data_status']['df_call'] and df_call is not None:
        call_expiries = set(df_call['Expiry'].dropna().unique())
        all_expiries.update(call_expiries)
    if validation_results['data_status']['df_put'] and df_put is not None:
        put_expiries = set(df_put['Expiry'].dropna().unique())
        all_expiries.update(put_expiries)
    
    sorted_expiries = sorted([exp for exp in all_expiries if pd.notna(exp)])
    
    validation_results['expiry_analysis'] = {
        'total_expiries': len(sorted_expiries),
        'expiry_list': sorted_expiries,
        'expiry_details': {}
    }
    
    print(f"   🎯 Total unique expiries: {len(sorted_expiries)}")
    
    if sorted_expiries:
        print(f"   📅 Expiry range: {sorted_expiries[0]:%d-%b-%Y} to {sorted_expiries[-1]:%d-%b-%Y}")
        print(f"   📋 Available expiries:")
        
        for i, exp in enumerate(sorted_expiries[:10]):  # Show first 10
            call_count = len(df_call[df_call['Expiry'] == exp]) if validation_results['data_status']['df_call'] else 0
            put_count = len(df_put[df_put['Expiry'] == exp]) if validation_results['data_status']['df_put'] else 0
            
            validation_results['expiry_analysis']['expiry_details'][exp] = {
                'calls': call_count,
                'puts': put_count,
                'total': call_count + put_count
            }
            
            print(f"      {exp:%d-%b-%Y}: {call_count} calls, {put_count} puts")
        
        if len(sorted_expiries) > 10:
            print(f"      ... and {len(sorted_expiries) - 10} more expiries")
    
    return validation_results


def display_validation_summary(validation_results: Dict[str, Any]) -> bool:
    """
    Display validation summary and next steps.
    
    Args:
        validation_results: Validation results dictionary
        
    Returns:
        bool: True if all data is valid, False otherwise
    """
    if not validation_results['all_valid']:
        print("\n❌ ENHANCED DATA VALIDATION FAILED")
        print("=" * 40)
        
        missing_data = [key for key, valid in validation_results['data_status'].items() if not valid]
        print(f"🚫 Invalid Data Components: {', '.join(missing_data)}")
        
        print("\n📋 DETAILED VALIDATION RESULTS:")
        for component, details in validation_results['validation_details'].items():
            status = "✅" if validation_results['data_status'][component] else "❌"
            print(f"   {status} {component}: {details}")
        
        print("\n🔧 REQUIRED ACTIONS:")
        action_map = {
            'bank_nifty': "📌 Run Cell 4: Load Bank Nifty Index Data",
            'df_call': "📌 Run Cell 2: Load Bank Nifty Options Data",
            'df_put': "📌 Run Cell 2: Load Bank Nifty Options Data"
        }
        
        for i, data_type in enumerate(missing_data, 1):
            action = action_map.get(data_type, f"📌 Load {data_type} data")
            print(f"   {i}. {action}")
        
        print("\n💡 Enhanced Quick Fix Guide:")
        print("   🔍 Run Cell 1: Enhanced Data Status Checker for comprehensive diagnostics")
        print("   📊 Execute required data loading cells in correct sequence")
        print("   🔄 Return to this cell once all data components are loaded and validated")
        print("   🚀 The enhanced analyzer will automatically detect and optimize your data")
        
        return False
    else:
        print("\n🎯 DATA VALIDATION PASSED - Ready for Enhanced Analysis")
        print("=" * 60)
        return True


def get_validation_summary_stats(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract summary statistics from validation results.
    
    Args:
        validation_results: Validation results dictionary
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'all_valid': validation_results['all_valid'],
        'total_expiries': validation_results.get('expiry_analysis', {}).get('total_expiries', 0),
        'data_counts': {}
    }
    
    for data_type in ['bank_nifty', 'df_call', 'df_put']:
        if data_type in validation_results['summary_stats']:
            summary['data_counts'][data_type] = validation_results['summary_stats'][data_type]['records']
        else:
            summary['data_counts'][data_type] = 0
    
    return summary


def validate_data_compatibility(bank_nifty: Optional[pd.DataFrame] = None,
                               df_call: Optional[pd.DataFrame] = None,
                               df_put: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Check data compatibility for analysis (date ranges, overlaps, etc.).
    
    Args:
        bank_nifty: Bank Nifty spot data
        df_call: Call options data
        df_put: Put options data
        
    Returns:
        dict: Compatibility analysis results
    """
    compatibility_results = {
        'compatible': False,
        'issues': [],
        'recommendations': [],
        'date_analysis': {}
    }
    
    date_ranges = {}
    
    # Collect date ranges
    if bank_nifty is not None and not bank_nifty.empty and 'Date' in bank_nifty.columns:
        date_ranges['spot'] = {
            'start': bank_nifty['Date'].min(),
            'end': bank_nifty['Date'].max(),
            'count': len(bank_nifty)
        }
    
    if df_call is not None and not df_call.empty and 'Date' in df_call.columns:
        date_ranges['calls'] = {
            'start': df_call['Date'].min(),
            'end': df_call['Date'].max(),
            'count': len(df_call)
        }
    
    if df_put is not None and not df_put.empty and 'Date' in df_put.columns:
        date_ranges['puts'] = {
            'start': df_put['Date'].min(),
            'end': df_put['Date'].max(),
            'count': len(df_put)
        }
    
    compatibility_results['date_analysis'] = date_ranges
    
    # Check for date overlaps
    if len(date_ranges) >= 2:
        all_starts = [info['start'] for info in date_ranges.values()]
        all_ends = [info['end'] for info in date_ranges.values()]
        
        overlap_start = max(all_starts)
        overlap_end = min(all_ends)
        
        if overlap_start <= overlap_end:
            compatibility_results['compatible'] = True
            compatibility_results['date_analysis']['overlap'] = {
                'start': overlap_start,
                'end': overlap_end,
                'days': (overlap_end - overlap_start).days + 1
            }
        else:
            compatibility_results['issues'].append("No overlapping date ranges found")
            compatibility_results['recommendations'].append("Ensure data covers overlapping time periods")
    
    return compatibility_results
