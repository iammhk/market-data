"""
Feature Engineering Module for XGBoost Bank Nifty Prediction

This module contains all the feature engineering functions for creating
robust features from options data, spot price data, and technical indicators.

Now includes intelligent caching support for improved performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


def safe_numeric_conversion(series: pd.Series) -> pd.Series:
    """
    Safely convert series to numeric, handling NaN and string values
    
    Args:
        series (pd.Series): Input series to convert
        
    Returns:
        pd.Series: Converted numeric series with NaN values filled with 0
    """
    try:
        # Convert to numeric, coercing errors to NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        # Fill NaN with 0
        return numeric_series.fillna(0)
    except:
        return pd.Series([0] * len(series))


def preprocess_options_data(df_call: pd.DataFrame, df_put: pd.DataFrame, bank_nifty: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess options and spot data by converting to numeric types and formatting dates
    
    Args:
        df_call (pd.DataFrame): Call options data
        df_put (pd.DataFrame): Put options data  
        bank_nifty (pd.DataFrame): Bank Nifty spot data
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Preprocessed dataframes
    """
    # Column mapping based on actual data structure
    column_mapping = {
        'price': 'LTP',  # Last Traded Price
        'volume': 'No. of contracts',  # Trading Volume
        'oi': 'Open Int',  # Open Interest
        'strike': 'Strike Price',  # Strike Price
        'close': 'Close',  # Close Price
        'turnover': 'Turnover * in  ₹ Lakhs'  # Turnover
    }
    
    print(f"📊 Processing data with column mapping: {column_mapping}")
    
    # Data preprocessing - convert to numeric
    print("🔧 Converting data types to numeric...")
    
    for df_name, df in [('Calls', df_call), ('Puts', df_put)]:
        for col in ['LTP', 'Close', 'Open', 'High', 'Low', 'Strike Price', 'No. of contracts', 'Open Int', 'Turnover * in  ₹ Lakhs']:
            if col in df.columns:
                original_type = df[col].dtype
                df[col] = safe_numeric_conversion(df[col])
                print(f"   ✅ {df_name} {col}: {original_type} → numeric")
    
    # Ensure consistent date formatting
    for df in [df_call, df_put, bank_nifty]:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        if 'Expiry' in df.columns:
            df['Expiry'] = pd.to_datetime(df['Expiry'])
    
    return df_call, df_put, bank_nifty


def create_call_options_features(day_calls: pd.DataFrame, spot_price: float) -> Dict:
    """
    Create features from call options data
    
    Args:
        day_calls (pd.DataFrame): Call options data for a specific date
        spot_price (float): Spot price for the date
        
    Returns:
        Dict: Dictionary of call options features
    """
    features = {}
    
    if not day_calls.empty:
        # Aggregate call features
        features.update({
            'call_total_volume': day_calls['No. of contracts'].sum(),
            'call_avg_ltp': day_calls['LTP'].mean(),
            'call_max_ltp': day_calls['LTP'].max(),
            'call_min_ltp': day_calls['LTP'].min(),
            'call_total_oi': day_calls['Open Int'].sum(),
            'call_avg_close': day_calls['Close'].mean(),
            'call_unique_strikes': day_calls['Strike Price'].nunique(),
            'call_total_turnover': day_calls['Turnover * in  ₹ Lakhs'].sum() if 'Turnover * in  ₹ Lakhs' in day_calls.columns else 0,
        })
        
        # ATM and ITM/OTM analysis for calls
        if 'Strike Price' in day_calls.columns:
            strikes = day_calls['Strike Price'].values
            closest_strike_idx = np.argmin(np.abs(strikes - spot_price))
            atm_strike = strikes[closest_strike_idx]
            
            # ATM call features
            atm_calls = day_calls[day_calls['Strike Price'] == atm_strike]
            if not atm_calls.empty:
                features.update({
                    'call_atm_ltp': atm_calls['LTP'].iloc[0],
                    'call_atm_volume': atm_calls['No. of contracts'].iloc[0],
                    'call_atm_oi': atm_calls['Open Int'].iloc[0],
                })
            
            # ITM/OTM analysis
            itm_calls = day_calls[day_calls['Strike Price'] < spot_price]
            otm_calls = day_calls[day_calls['Strike Price'] > spot_price]
            
            features.update({
                'call_itm_volume': itm_calls['No. of contracts'].sum(),
                'call_otm_volume': otm_calls['No. of contracts'].sum(),
                'call_itm_oi': itm_calls['Open Int'].sum(),
                'call_otm_oi': otm_calls['Open Int'].sum(),
                'call_itm_count': len(itm_calls),
                'call_otm_count': len(otm_calls),
            })
            
            # Ratios
            features['call_itm_otm_volume_ratio'] = (
                features['call_itm_volume'] / features['call_otm_volume'] 
                if features['call_otm_volume'] > 0 else 0
            )
            features['call_itm_otm_oi_ratio'] = (
                features['call_itm_oi'] / features['call_otm_oi'] 
                if features['call_otm_oi'] > 0 else 0
            )
    
    return features


def create_put_options_features(day_puts: pd.DataFrame, spot_price: float) -> Dict:
    """
    Create features from put options data
    
    Args:
        day_puts (pd.DataFrame): Put options data for a specific date
        spot_price (float): Spot price for the date
        
    Returns:
        Dict: Dictionary of put options features
    """
    features = {}
    
    if not day_puts.empty:
        # Aggregate put features
        features.update({
            'put_total_volume': day_puts['No. of contracts'].sum(),
            'put_avg_ltp': day_puts['LTP'].mean(),
            'put_max_ltp': day_puts['LTP'].max(),
            'put_min_ltp': day_puts['LTP'].min(),
            'put_total_oi': day_puts['Open Int'].sum(),
            'put_avg_close': day_puts['Close'].mean(),
            'put_unique_strikes': day_puts['Strike Price'].nunique(),
            'put_total_turnover': day_puts['Turnover * in  ₹ Lakhs'].sum() if 'Turnover * in  ₹ Lakhs' in day_puts.columns else 0,
        })
        
        # ATM and ITM/OTM analysis for puts
        if 'Strike Price' in day_puts.columns:
            strikes = day_puts['Strike Price'].values
            closest_strike_idx = np.argmin(np.abs(strikes - spot_price))
            atm_strike = strikes[closest_strike_idx]
            
            # ATM put features
            atm_puts = day_puts[day_puts['Strike Price'] == atm_strike]
            if not atm_puts.empty:
                features.update({
                    'put_atm_ltp': atm_puts['LTP'].iloc[0],
                    'put_atm_volume': atm_puts['No. of contracts'].iloc[0],
                    'put_atm_oi': atm_puts['Open Int'].iloc[0],
                })
            
            # ITM/OTM analysis for puts (opposite to calls)
            itm_puts = day_puts[day_puts['Strike Price'] > spot_price]
            otm_puts = day_puts[day_puts['Strike Price'] < spot_price]
            
            features.update({
                'put_itm_volume': itm_puts['No. of contracts'].sum(),
                'put_otm_volume': otm_puts['No. of contracts'].sum(),
                'put_itm_oi': itm_puts['Open Int'].sum(),
                'put_otm_oi': otm_puts['Open Int'].sum(),
                'put_itm_count': len(itm_puts),
                'put_otm_count': len(otm_puts),
            })
            
            features['put_itm_otm_volume_ratio'] = (
                features['put_itm_volume'] / features['put_otm_volume'] 
                if features['put_otm_volume'] > 0 else 0
            )
            features['put_itm_otm_oi_ratio'] = (
                features['put_itm_oi'] / features['put_otm_oi'] 
                if features['put_otm_oi'] > 0 else 0
            )
    
    return features


def create_combined_options_features(call_features: Dict, put_features: Dict) -> Dict:
    """
    Create combined call-put features including PCR ratios
    
    Args:
        call_features (Dict): Call options features
        put_features (Dict): Put options features
        
    Returns:
        Dict: Dictionary of combined features
    """
    features = {}
    
    # Put-Call Ratios
    features['pcr_volume'] = (
        put_features.get('put_total_volume', 0) / call_features.get('call_total_volume', 1)
        if call_features.get('call_total_volume', 1) > 0 else 0
    )
    features['pcr_oi'] = (
        put_features.get('put_total_oi', 0) / call_features.get('call_total_oi', 1)
        if call_features.get('call_total_oi', 1) > 0 else 0
    )
    features['pcr_ltp'] = (
        put_features.get('put_avg_ltp', 0) / call_features.get('call_avg_ltp', 1)
        if call_features.get('call_avg_ltp', 1) > 0 else 0
    )
    
    # Combined totals
    features['total_volume'] = call_features.get('call_total_volume', 0) + put_features.get('put_total_volume', 0)
    features['total_oi'] = call_features.get('call_total_oi', 0) + put_features.get('put_total_oi', 0)
    features['total_turnover'] = call_features.get('call_total_turnover', 0) + put_features.get('put_total_turnover', 0)
    
    # Market sentiment indicators
    features['volume_weighted_pcr'] = (
        (put_features.get('put_total_volume', 0) * put_features.get('put_avg_ltp', 0)) /
        (call_features.get('call_total_volume', 1) * call_features.get('call_avg_ltp', 1))
        if call_features.get('call_total_volume', 1) > 0 and call_features.get('call_avg_ltp', 1) > 0 else 0
    )
    
    return features


def create_lag_features(bank_nifty: pd.DataFrame, date: pd.Timestamp, spot_price: float) -> Dict:
    """
    Create lag features from previous day spot price and volume data
    
    Args:
        bank_nifty (pd.DataFrame): Bank Nifty spot data
        date (pd.Timestamp): Current date
        spot_price (float): Current spot price
        
    Returns:
        Dict: Dictionary of lag features
    """
    features = {}
    
    # Get previous trading day's data (lag features)
    previous_dates = sorted([d for d in bank_nifty['Date'].dt.date if d < date.date()])
    
    if previous_dates:
        # Get the most recent previous day
        prev_date = previous_dates[-1]
        prev_spot_data = bank_nifty[bank_nifty['Date'].dt.date == prev_date]
        
        if not prev_spot_data.empty:
            prev_close = prev_spot_data['Close'].iloc[-1]
            prev_volume = prev_spot_data['Volume'].iloc[-1]
            prev_high = prev_spot_data['High'].iloc[-1]
            prev_low = prev_spot_data['Low'].iloc[-1]
            prev_open = prev_spot_data['Open'].iloc[-1]
            
            # Previous day price features
            features.update({
                'prev_close': prev_close,
                'prev_volume': prev_volume,
                'prev_high': prev_high,
                'prev_low': prev_low,
                'prev_open': prev_open,
                'prev_range': prev_high - prev_low,
                'prev_body': abs(prev_close - prev_open),
                'prev_upper_shadow': prev_high - max(prev_close, prev_open),
                'prev_lower_shadow': min(prev_close, prev_open) - prev_low,
            })
            
            # Price momentum features
            features.update({
                'price_change_pct': ((spot_price - prev_close) / prev_close) * 100,
                'gap_up_down': ((spot_price - prev_close) / prev_close) * 100,  # Gap from previous close
                'volume_ratio': prev_volume / 1000000,  # Volume in millions for scaling
            })
            
            # Get 2-day lag features if available
            if len(previous_dates) >= 2:
                prev2_date = previous_dates[-2]
                prev2_spot_data = bank_nifty[bank_nifty['Date'].dt.date == prev2_date]
                
                if not prev2_spot_data.empty:
                    prev2_close = prev2_spot_data['Close'].iloc[-1]
                    prev2_volume = prev2_spot_data['Volume'].iloc[-1]
                    
                    features.update({
                        'prev2_close': prev2_close,
                        'prev2_volume': prev2_volume,
                        '2day_price_change_pct': ((prev_close - prev2_close) / prev2_close) * 100,
                        '2day_volume_change_pct': ((prev_volume - prev2_volume) / prev2_volume) * 100 if prev2_volume > 0 else 0,
                        'price_momentum_2day': ((spot_price - prev2_close) / prev2_close) * 100,
                    })
            
            # Get 5-day moving average features if available
            if len(previous_dates) >= 5:
                recent_5_dates = previous_dates[-5:]
                recent_5_data = bank_nifty[bank_nifty['Date'].dt.date.isin(recent_5_dates)]
                
                if len(recent_5_data) >= 5:
                    ma5_close = recent_5_data['Close'].mean()
                    ma5_volume = recent_5_data['Volume'].mean()
                    
                    features.update({
                        'ma5_close': ma5_close,
                        'ma5_volume': ma5_volume,
                        'price_vs_ma5': ((spot_price - ma5_close) / ma5_close) * 100,
                        'volume_vs_ma5': ((prev_volume - ma5_volume) / ma5_volume) * 100 if ma5_volume > 0 else 0,
                    })
        else:
            # Fill with default values if no previous data
            _fill_default_lag_features(features, spot_price)
    else:
        # Fill with default values for first date
        _fill_default_lag_features(features, spot_price)
    
    return features


def _fill_default_lag_features(features: Dict, spot_price: float) -> None:
    """
    Fill default values for lag features when historical data is not available
    
    Args:
        features (Dict): Features dictionary to update
        spot_price (float): Current spot price
    """
    features.update({
        'prev_close': spot_price,
        'prev_volume': 100000,
        'prev_high': spot_price,
        'prev_low': spot_price,
        'prev_open': spot_price,
        'prev_range': 0,
        'prev_body': 0,
        'prev_upper_shadow': 0,
        'prev_lower_shadow': 0,
        'price_change_pct': 0,
        'gap_up_down': 0,
        'volume_ratio': 0.1,
        'prev2_close': spot_price,
        'prev2_volume': 100000,
        '2day_price_change_pct': 0,
        '2day_volume_change_pct': 0,
        'price_momentum_2day': 0,
        'ma5_close': spot_price,
        'ma5_volume': 100000,
        'price_vs_ma5': 0,
        'volume_vs_ma5': 0,
    })


def create_time_features(date_obj: pd.Timestamp) -> Dict:
    """
    Create time-based features from date
    
    Args:
        date_obj (pd.Timestamp): Date object
        
    Returns:
        Dict: Dictionary of time-based features
    """
    return {
        'day_of_week': date_obj.dayofweek,
        'day_of_month': date_obj.day,
        'month': date_obj.month,
        'quarter': date_obj.quarter,
        'is_month_end': 1 if date_obj.day > 25 else 0,
        'is_quarter_end': 1 if date_obj.month in [3, 6, 9, 12] and date_obj.day > 25 else 0
    }


def create_robust_options_features(df_call: pd.DataFrame, df_put: pd.DataFrame, bank_nifty: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to create comprehensive features from options and spot data
    
    Args:
        df_call (pd.DataFrame): Call options data
        df_put (pd.DataFrame): Put options data
        bank_nifty (pd.DataFrame): Bank Nifty spot data
        
    Returns:
        pd.DataFrame: Complete features dataframe ready for ML modeling
    """
    # Preprocess data
    df_call, df_put, bank_nifty = preprocess_options_data(df_call, df_put, bank_nifty)
    
    # Create features by date
    features_list = []
    
    # Get unique dates from options data
    call_dates = set(df_call['Date'].dt.date) if 'Date' in df_call.columns else set()
    put_dates = set(df_put['Date'].dt.date) if 'Date' in df_put.columns else set()
    common_dates = sorted(call_dates.intersection(put_dates))
    
    print(f"📅 Processing {len(common_dates)} common trading dates")
    
    processed_count = 0
    for date in common_dates:
        # Process all available dates for complete dataset
        date_obj = pd.to_datetime(date)
        
        # Get Bank Nifty spot price for this date (target variable)
        spot_data = bank_nifty[bank_nifty['Date'].dt.date == date]
        if spot_data.empty:
            continue
            
        # Use Close price as target
        spot_price = spot_data['Close'].iloc[-1]  # Use last available price for the day
        
        # Get options data for this date
        day_calls = df_call[df_call['Date'].dt.date == date].copy()
        day_puts = df_put[df_put['Date'].dt.date == date].copy()
        
        if day_calls.empty or day_puts.empty:
            continue
        
        # Initialize feature dictionary
        features = {
            'Date': date_obj,
            'target_spot_price': spot_price
        }
        
        # Create all feature types
        call_features = create_call_options_features(day_calls, spot_price)
        put_features = create_put_options_features(day_puts, spot_price)
        combined_features = create_combined_options_features(call_features, put_features)
        lag_features = create_lag_features(bank_nifty, date_obj, spot_price)
        time_features = create_time_features(date_obj)
        
        # Combine all features
        features.update(call_features)
        features.update(put_features)
        features.update(combined_features)
        features.update(lag_features)
        features.update(time_features)
        
        features_list.append(features)
        processed_count += 1
        
        if processed_count % 10 == 0:
            print(f"   ✅ Processed {processed_count} dates...")
    
    return pd.DataFrame(features_list)


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Get categorized feature groups for analysis
    
    Returns:
        Dict[str, List[str]]: Dictionary of feature categories and their feature names
    """
    return {
        'call_features': [
            'call_total_volume', 'call_avg_ltp', 'call_max_ltp', 'call_min_ltp',
            'call_total_oi', 'call_avg_close', 'call_unique_strikes', 'call_total_turnover',
            'call_atm_ltp', 'call_atm_volume', 'call_atm_oi', 'call_itm_volume',
            'call_otm_volume', 'call_itm_oi', 'call_otm_oi', 'call_itm_count',
            'call_otm_count', 'call_itm_otm_volume_ratio', 'call_itm_otm_oi_ratio'
        ],
        'put_features': [
            'put_total_volume', 'put_avg_ltp', 'put_max_ltp', 'put_min_ltp',
            'put_total_oi', 'put_avg_close', 'put_unique_strikes', 'put_total_turnover',
            'put_atm_ltp', 'put_atm_volume', 'put_atm_oi', 'put_itm_volume',
            'put_otm_volume', 'put_itm_oi', 'put_otm_oi', 'put_itm_count',
            'put_otm_count', 'put_itm_otm_volume_ratio', 'put_itm_otm_oi_ratio'
        ],
        'combined_features': [
            'pcr_volume', 'pcr_oi', 'pcr_ltp', 'total_volume', 'total_oi',
            'total_turnover', 'volume_weighted_pcr'
        ],
        'lag_features': [
            'prev_close', 'prev_volume', 'prev_high', 'prev_low', 'prev_open',
            'prev_range', 'prev_body', 'prev_upper_shadow', 'prev_lower_shadow',
            'price_change_pct', 'gap_up_down', 'volume_ratio', 'prev2_close',
            'prev2_volume', '2day_price_change_pct', '2day_volume_change_pct',
            'price_momentum_2day', 'ma5_close', 'ma5_volume', 'price_vs_ma5',
            'volume_vs_ma5'
        ],
        'time_features': [
            'day_of_week', 'day_of_month', 'month', 'quarter',
            'is_month_end', 'is_quarter_end'
        ]
    }


def create_features_with_cache(df_call: pd.DataFrame, df_put: pd.DataFrame, 
                              bank_nifty: pd.DataFrame, cache_dir: Optional[str] = None,
                              max_cache_age_hours: int = 24, force_regenerate: bool = False) -> pd.DataFrame:
    """
    Create features with intelligent caching support
    
    This function wraps create_robust_options_features with caching capabilities:
    - Automatically checks for cached features based on input data hash
    - Loads cached features if available and not expired
    - Generates new features if cache miss or expired
    - Saves new features to cache for future use
    
    Args:
        df_call (pd.DataFrame): Call options data
        df_put (pd.DataFrame): Put options data
        bank_nifty (pd.DataFrame): Bank Nifty spot data
        cache_dir (str, optional): Directory for cache storage. Defaults to ./data/cache
        max_cache_age_hours (int): Maximum age of cached features in hours. Default: 24
        force_regenerate (bool): Force regeneration even if cache exists. Default: False
        
    Returns:
        pd.DataFrame: Engineered features dataset
        
    Example:
        >>> features_df = create_features_with_cache(df_call, df_put, bank_nifty)
        >>> # First run: Generates and caches features
        >>> # Subsequent runs: Loads from cache if available
    """
    try:
        # Import the caching module
        from .feature_cache import FeatureCache
        
        # Initialize cache
        cache = FeatureCache(cache_dir, max_cache_age_hours)
        
        # Generate cache key from input data
        cache_key = cache.generate_data_hash(df_call, df_put, bank_nifty)
        
        # Try to load from cache first (unless forced regeneration)
        if not force_regenerate:
            features_df, cache_status = cache.load_features(cache_key, "xgboost_features")
            
            if features_df is not None:
                print(f"🚀 Using cached features: {cache_status}")
                return features_df
            else:
                print(f"🔄 Cache miss: {cache_status}")
        else:
            print("🔄 Forced regeneration: Skipping cache lookup")
        
        # Generate features using the main function
        print("⚙️ Generating features using create_robust_options_features...")
        features_df = create_robust_options_features(df_call, df_put, bank_nifty)
        
        if features_df is not None and not features_df.empty:
            # Prepare metadata for caching
            metadata = {
                'call_records': len(df_call),
                'put_records': len(df_put),
                'spot_records': len(bank_nifty),
                'date_range': f"{features_df['Date'].min():%Y-%m-%d} to {features_df['Date'].max():%Y-%m-%d}",
                'feature_count': len([col for col in features_df.columns if col not in ['Date', 'target_spot_price']]),
                'shape': str(features_df.shape),
                'generated_at': pd.Timestamp.now().isoformat()
            }
            
            # Save to cache
            cache_saved = cache.save_features(features_df, cache_key, metadata, "xgboost_features")
            
            if cache_saved:
                print("✅ Features saved to cache for future use")
            else:
                print("⚠️ Failed to save features to cache")
        
        return features_df
        
    except ImportError:
        print("⚠️ Feature caching not available - using direct generation")
        print("💡 To enable caching, ensure feature_cache.py is available")
        return create_robust_options_features(df_call, df_put, bank_nifty)
    
    except Exception as e:
        print(f"❌ Error in cached feature generation: {e}")
        print("🔄 Falling back to direct feature generation")
        return create_robust_options_features(df_call, df_put, bank_nifty)


def clear_feature_cache(cache_dir: Optional[str] = None, confirm: bool = False, older_than_hours: Optional[int] = None) -> int:
    """
    Clear cached feature files
    
    Args:
        cache_dir (str, optional): Cache directory. Defaults to ./data/cache
        confirm (bool): Must be True to actually delete files
        older_than_hours (int, optional): Only delete files older than this
        
    Returns:
        int: Number of files deleted
    """
    try:
        from .feature_cache import FeatureCache
        
        cache = FeatureCache(cache_dir)
        return cache.clear_cache("xgboost_features", confirm, older_than_hours)
        
    except ImportError:
        print("❌ Feature caching module not available")
        return 0


def list_cached_features(cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    List all cached feature files with details
    
    Args:
        cache_dir (str, optional): Cache directory. Defaults to ./data/cache
        
    Returns:
        pd.DataFrame: Information about cached files
    """
    try:
        from .feature_cache import FeatureCache
        
        cache = FeatureCache(cache_dir)
        return cache.list_cached_features("xgboost_features")
        
    except ImportError:
        print("❌ Feature caching module not available")
        return pd.DataFrame()


def validate_feature_cache(cache_dir: Optional[str] = None) -> Dict:
    """
    Validate cached feature files
    
    Args:
        cache_dir (str, optional): Cache directory. Defaults to ./data/cache
        
    Returns:
        Dict: Validation summary
    """
    try:
        from .feature_cache import FeatureCache
        
        cache = FeatureCache(cache_dir)
        return cache.validate_cache("xgboost_features")
        
    except ImportError:
        print("❌ Feature caching module not available")
    return {'error': 'Caching module not available'}
