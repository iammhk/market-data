"""
🤖 XGBOOST FEATURE ENGINEERING WITH CACHING
Advanced feature engineering utilities for XGBoost with intelligent caching support

This module provides comprehensive feature engineering functions specifically designed
for XGBoost modeling with integrated caching to improve performance and reduce
computation time for repeated executions.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
from typing import Dict, Any, Optional, Tuple, List


def generate_data_hash(df_call: pd.DataFrame, df_put: pd.DataFrame, bank_nifty: pd.DataFrame) -> str:
    """
    Generate a hash of the input data to check if features need regeneration
    
    Args:
        df_call: Call options DataFrame
        df_put: Put options DataFrame  
        bank_nifty: Bank Nifty spot price DataFrame
        
    Returns:
        str: Combined hash string for the datasets
    """
    try:
        # Create a combined hash of all input data
        call_hash = hashlib.md5(
            str(df_call.shape).encode() + 
            str(df_call['Date'].min()).encode() + 
            str(df_call['Date'].max()).encode()
        ).hexdigest()[:8]
        
        put_hash = hashlib.md5(
            str(df_put.shape).encode() + 
            str(df_put['Date'].min()).encode() + 
            str(df_put['Date'].max()).encode()
        ).hexdigest()[:8]
        
        spot_hash = hashlib.md5(
            str(bank_nifty.shape).encode() + 
            str(bank_nifty['Date'].min()).encode() + 
            str(bank_nifty['Date'].max()).encode()
        ).hexdigest()[:8]
        
        combined_hash = call_hash + put_hash + spot_hash
        return combined_hash
    except Exception as e:
        print(f"⚠️ Hash generation failed: {e}")
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_features_cache_path(data_hash: str, data_path: str) -> str:
    """
    Get the cache file path for features
    
    Args:
        data_hash: Hash string for the data
        data_path: Base data directory path
        
    Returns:
        str: Full path to cache file
    """
    cache_dir = os.path.join(data_path, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_filename = f"xgboost_features_{data_hash}.pkl"
    return os.path.join(cache_dir, cache_filename)


def save_features_to_cache(features_df: pd.DataFrame, cache_path: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save features DataFrame to cache with metadata
    
    Args:
        features_df: Features DataFrame to cache
        cache_path: Path to save the cache file
        metadata: Optional metadata dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cache_data = {
            'features_df': features_df,
            'created_at': datetime.now(),
            'metadata': metadata or {},
            'version': '1.0'
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"💾 Features saved to cache: {os.path.basename(cache_path)}")
        print(f"📊 Cached features shape: {features_df.shape}")
        return True
    except Exception as e:
        print(f"❌ Failed to save features to cache: {e}")
        return False


def load_features_from_cache(cache_path: str, max_age_hours: int = 24) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load features DataFrame from cache if valid
    
    Args:
        cache_path: Path to cache file
        max_age_hours: Maximum age in hours before cache expires
        
    Returns:
        Tuple[Optional[pd.DataFrame], str]: Features DataFrame and status message
    """
    try:
        if not os.path.exists(cache_path):
            return None, "Cache file not found"
        
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check cache age
        created_at = cache_data.get('created_at', datetime.now())
        age_hours = (datetime.now() - created_at).total_seconds() / 3600
        
        if age_hours > max_age_hours:
            return None, f"Cache expired (age: {age_hours:.1f}h, max: {max_age_hours}h)"
        
        features_df = cache_data.get('features_df')
        if features_df is None or features_df.empty:
            return None, "Invalid cached data"
        
        print(f"✅ Features loaded from cache: {os.path.basename(cache_path)}")
        print(f"📊 Cached features shape: {features_df.shape}")
        print(f"⏰ Cache age: {age_hours:.1f} hours")
        
        metadata = cache_data.get('metadata', {})
        if metadata:
            print(f"📋 Cache metadata: {metadata}")
        
        return features_df, "Success"
        
    except Exception as e:
        return None, f"Failed to load cache: {e}"


def execute_feature_engineering_with_cache(df_call: pd.DataFrame, df_put: pd.DataFrame, 
                                         bank_nifty: pd.DataFrame, data_path: str,
                                         create_features_func, get_feature_groups_func = None,
                                         max_cache_age_hours: int = 24,
                                         force_regenerate: bool = False) -> Optional[pd.DataFrame]:
    """
    Execute feature engineering with intelligent caching
    
    Args:
        df_call: Call options DataFrame
        df_put: Put options DataFrame
        bank_nifty: Bank Nifty spot price DataFrame
        data_path: Base data directory path
        create_features_func: Function to create features
        get_feature_groups_func: Optional function to get feature groups
        max_cache_age_hours: Maximum cache age in hours
        force_regenerate: Force regeneration even if cache exists
        
    Returns:
        Optional[pd.DataFrame]: Features DataFrame or None if failed
    """
    
    print("\n🔍 CHECKING FOR CACHED FEATURES...")
    print("-" * 35)
    
    # Generate data hash
    data_hash = generate_data_hash(df_call, df_put, bank_nifty)
    cache_path = get_features_cache_path(data_hash, data_path)
    
    print(f"🔐 Data hash: {data_hash}")
    print(f"📁 Cache path: {os.path.basename(cache_path)}")
    
    features_df = None
    
    # Try to load from cache first (unless force regenerate)
    if not force_regenerate:
        features_df, cache_status = load_features_from_cache(cache_path, max_cache_age_hours)
        
        if features_df is not None:
            print(f"\n🚀 USING CACHED FEATURES!")
            print("-" * 25)
            print(f"✅ {cache_status}")
            return features_df
    
    # Generate new features
    print(f"\n🛠️ GENERATING NEW FEATURES...")
    print("-" * 30)
    if not force_regenerate:
        print(f"📝 Reason: {cache_status if 'cache_status' in locals() else 'Force regeneration requested'}")
    else:
        print(f"📝 Reason: Force regeneration requested")
    
    # Generate features using the provided function
    print("⚙️ Running cached feature engineering pipeline...")
    
    try:
        # Try with caching parameters if supported
        try:
            features_df = create_features_func(
                df_call, 
                df_put, 
                bank_nifty,
                cache_dir=os.path.join(data_path, 'cache'),
                max_cache_age_hours=max_cache_age_hours,
                force_regenerate=force_regenerate
            )
        except TypeError:
            # Fallback for functions that don't support caching parameters
            features_df = create_features_func(df_call, df_put, bank_nifty)
        
        if features_df is not None and not features_df.empty:
            # Save to cache for future use
            metadata = {
                'call_records': len(df_call),
                'put_records': len(df_put),
                'spot_records': len(bank_nifty),
                'date_range': f"{features_df['Date'].min():%Y-%m-%d} to {features_df['Date'].max():%Y-%m-%d}",
                'feature_count': len([col for col in features_df.columns if col not in ['Date', 'target_spot_price']])
            }
            
            cache_saved = save_features_to_cache(features_df, cache_path, metadata)
            if cache_saved:
                print("✅ Features cached for future use")
            else:
                print("⚠️ Failed to cache features (will regenerate next time)")
        else:
            print("❌ Feature generation failed")
            return None
            
    except Exception as e:
        print(f"❌ Error during feature generation: {e}")
        return None
    
    return features_df


def display_feature_engineering_results(features_df: pd.DataFrame, 
                                       get_feature_groups_func = None) -> None:
    """
    Display comprehensive results of feature engineering
    
    Args:
        features_df: Generated features DataFrame
        get_feature_groups_func: Optional function to get feature groups
    """
    if features_df is None or features_df.empty:
        print("❌ No features to display")
        return
        
    print(f"\n✅ SUCCESS! Features ready for modeling")
    print("=" * 40)
    print(f"📊 Feature dimensions: {features_df.shape}")
    print(f"📅 Date range: {features_df['Date'].min():%d-%b-%Y} to {features_df['Date'].max():%d-%b-%Y}")
    
    # Display feature summary using feature groups
    try:
        if get_feature_groups_func:
            feature_groups = get_feature_groups_func()
            feature_cols = [col for col in features_df.columns if col not in ['Date', 'target_spot_price']]
            print(f"🎯 Total features created: {len(feature_cols)}")
            print(f"💰 Target range: ₹{features_df['target_spot_price'].min():,.0f} - ₹{features_df['target_spot_price'].max():,.0f}")
            
            # Show feature breakdown by category
            print(f"\n📋 FEATURE BREAKDOWN BY CATEGORY:")
            for category, feature_list in feature_groups.items():
                available_features = [f for f in feature_list if f in features_df.columns]
                print(f"   📊 {category.replace('_', ' ').title()}: {len(available_features)} features")
        else:
            feature_cols = [col for col in features_df.columns if col not in ['Date', 'target_spot_price']]
            print(f"🎯 Total features created: {len(feature_cols)}")
            print(f"💰 Target range: ₹{features_df['target_spot_price'].min():,.0f} - ₹{features_df['target_spot_price'].max():,.0f}")
    except Exception as e:
        print(f"⚠️ Could not load feature groups: {e}")
        feature_cols = [col for col in features_df.columns if col not in ['Date', 'target_spot_price']]
        print(f"🎯 Total features created: {len(feature_cols)}")
    
    # Show sample features
    print(f"\n📋 SAMPLE FEATURES (First 3 records):")
    display_cols = ['Date', 'target_spot_price', 'call_total_volume', 'put_total_volume', 
                   'pcr_volume', 'call_atm_ltp', 'put_atm_ltp', 'total_oi']
    available_display_cols = [col for col in display_cols if col in features_df.columns]
    
    # Return the DataFrame slice for display
    return features_df[available_display_cols].head(3)


def display_cache_management_info(cache_path: str, data_path: str) -> None:
    """
    Display cache management information
    
    Args:
        cache_path: Current cache file path
        data_path: Base data directory path
    """
    print(f"\n💾 CACHE MANAGEMENT:")
    print("-" * 20)
    cache_dir = os.path.dirname(cache_path)
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith('xgboost_features_') and f.endswith('.pkl')]
        print(f"📁 Cache directory: {cache_dir}")
        print(f"🗃️ Cached feature files: {len(cache_files)}")
        
        # Show cache file sizes
        total_cache_size = 0
        for cache_file in cache_files:
            cache_file_path = os.path.join(cache_dir, cache_file)
            if os.path.exists(cache_file_path):
                size_mb = os.path.getsize(cache_file_path) / (1024 * 1024)
                total_cache_size += size_mb
                if cache_file == os.path.basename(cache_path):
                    print(f"   📄 {cache_file} ({size_mb:.1f}MB) ← Current")
                else:
                    print(f"   📄 {cache_file} ({size_mb:.1f}MB)")
        
        print(f"💽 Total cache size: {total_cache_size:.1f}MB")


def clear_xgboost_feature_cache(data_path: str, confirm: bool = False) -> None:
    """
    Clear all cached XGBoost feature files
    
    Args:
        data_path: Base data directory path
        confirm: Must be True to actually clear cache
    """
    if not confirm:
        print("⚠️ Use clear_xgboost_feature_cache(data_path, confirm=True) to actually clear cache")
        return
    
    cache_dir = os.path.join(data_path, 'cache')
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith('xgboost_features_') and f.endswith('.pkl')]
        removed_count = 0
        for cache_file in cache_files:
            cache_file_path = os.path.join(cache_dir, cache_file)
            try:
                os.remove(cache_file_path)
                print(f"🗑️ Deleted: {cache_file}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to delete {cache_file}: {e}")
        print(f"✅ Cache cleared: {removed_count} files removed")
    else:
        print("📁 No cache directory found")


def list_xgboost_feature_cache(data_path: str) -> None:
    """
    List all cached XGBoost feature files with details
    
    Args:
        data_path: Base data directory path
    """
    cache_dir = os.path.join(data_path, 'cache')
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith('xgboost_features_') and f.endswith('.pkl')]
        
        if cache_files:
            print(f"📋 CACHED XGBOOST FEATURE FILES")
            print("-" * 35)
            for cache_file in cache_files:
                cache_file_path = os.path.join(cache_dir, cache_file)
                try:
                    size_mb = os.path.getsize(cache_file_path) / (1024 * 1024)
                    modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file_path))
                    age_hours = (datetime.now() - modified_time).total_seconds() / 3600
                    
                    # Try to get metadata
                    with open(cache_file_path, 'rb') as f:
                        cache_data = pickle.load(f)
                        shape = cache_data.get('features_df', pd.DataFrame()).shape
                        metadata = cache_data.get('metadata', {})
                    
                    print(f"📄 {cache_file}")
                    print(f"   💽 Size: {size_mb:.1f}MB")
                    print(f"   📊 Shape: {shape}")
                    print(f"   ⏰ Age: {age_hours:.1f}h")
                    if metadata:
                        print(f"   📋 Range: {metadata.get('date_range', 'Unknown')}")
                    print()
                    
                except Exception as e:
                    print(f"📄 {cache_file} (Error reading: {e})")
        else:
            print("📁 No cached XGBoost feature files found")
    else:
        print("📁 Cache directory does not exist")


def create_xgboost_cache_management_functions(data_path: str) -> Dict[str, callable]:
    """
    Create cache management functions bound to a specific data path
    
    Args:
        data_path: Base data directory path
        
    Returns:
        Dict[str, callable]: Dictionary of cache management functions
    """
    
    def clear_cache(confirm: bool = False):
        """Clear XGBoost feature cache"""
        return clear_xgboost_feature_cache(data_path, confirm)
    
    def list_cache():
        """List XGBoost feature cache"""
        return list_xgboost_feature_cache(data_path)
    
    def get_cache_info():
        """Get cache directory info"""
        cache_dir = os.path.join(data_path, 'cache')
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) if f.startswith('xgboost_features_') and f.endswith('.pkl')]
            total_size = sum(
                os.path.getsize(os.path.join(cache_dir, f)) 
                for f in cache_files 
                if os.path.exists(os.path.join(cache_dir, f))
            ) / (1024 * 1024)
            
            return {
                'cache_dir': cache_dir,
                'file_count': len(cache_files),
                'total_size_mb': total_size,
                'files': cache_files
            }
        else:
            return {
                'cache_dir': cache_dir,
                'file_count': 0,
                'total_size_mb': 0,
                'files': []
            }
    
    return {
        'clear_cache': clear_cache,
        'list_cache': list_cache,
        'get_cache_info': get_cache_info
    }


# Export main functions
__all__ = [
    'generate_data_hash',
    'get_features_cache_path',
    'save_features_to_cache',
    'load_features_from_cache',
    'execute_feature_engineering_with_cache',
    'display_feature_engineering_results',
    'display_cache_management_info',
    'clear_xgboost_feature_cache',
    'list_xgboost_feature_cache',
    'create_xgboost_cache_management_functions'
]
