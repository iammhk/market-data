"""
Feature Caching Utility for Market Data Analysis
Provides intelligent caching for engineered features to improve performance
"""

import os
import pickle
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any


class FeatureCache:
    """
    Intelligent caching system for engineered features
    
    Features:
    - Hash-based cache invalidation
    - Configurable cache expiration
    - Metadata storage
    - Cache management utilities
    """
    
    def __init__(self, cache_dir: Optional[str] = None, default_max_age_hours: int = 24):
        """
        Initialize feature cache
        
        Args:
            cache_dir: Directory to store cache files (defaults to ./data/cache)
            default_max_age_hours: Default maximum age for cached features
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'data', 'cache')
        self.default_max_age_hours = default_max_age_hours
        self.version = '1.0'
        
        # Ensure cache directory exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def generate_data_hash(self, *dataframes) -> str:
        """
        Generate a hash from input dataframes for cache key
        
        Args:
            *dataframes: DataFrames to include in hash
            
        Returns:
            Combined hash string
        """
        try:
            hash_components = []
            
            for i, df in enumerate(dataframes):
                if df is not None and not df.empty:
                    # Create hash from shape, date range, and sample data
                    shape_str = str(df.shape)
                    
                    # Include date range if Date column exists
                    if 'Date' in df.columns:
                        date_min = str(df['Date'].min())
                        date_max = str(df['Date'].max())
                    else:
                        date_min = date_max = "no_date"
                    
                    # Sample some data for more robust hashing
                    sample_str = str(df.head(2).to_dict()) if len(df) > 0 else "empty"
                    
                    component_str = f"{shape_str}_{date_min}_{date_max}_{sample_str}"
                    component_hash = hashlib.md5(component_str.encode()).hexdigest()[:8]
                    hash_components.append(f"df{i}_{component_hash}")
                else:
                    hash_components.append(f"df{i}_empty")
            
            combined_hash = "_".join(hash_components)
            return combined_hash[:32]  # Limit length for filesystem compatibility
            
        except Exception as e:
            print(f"⚠️ Hash generation failed: {e}")
            # Fallback to timestamp-based hash
            return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_cache_path(self, cache_key: str, prefix: str = "features") -> str:
        """
        Get full path for cache file
        
        Args:
            cache_key: Unique identifier for cached data
            prefix: Prefix for cache filename
            
        Returns:
            Full path to cache file
        """
        filename = f"{prefix}_{cache_key}.pkl"
        return os.path.join(self.cache_dir, filename)
    
    def save_features(self, features_df: pd.DataFrame, cache_key: str, 
                     metadata: Dict[str, Any] = None, prefix: str = "features") -> bool:
        """
        Save features to cache with metadata
        
        Args:
            features_df: DataFrame containing engineered features
            cache_key: Unique identifier for this feature set
            metadata: Additional metadata to store
            prefix: Prefix for cache filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_path = self.get_cache_path(cache_key, prefix)
            
            cache_data = {
                'features_df': features_df,
                'created_at': datetime.now(),
                'cache_key': cache_key,
                'version': self.version,
                'metadata': metadata or {}
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"💾 Features cached successfully!")
            print(f"   📁 File: {os.path.basename(cache_path)}")
            print(f"   📊 Shape: {features_df.shape}")
            print(f"   💽 Size: {file_size_mb:.1f}MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save features to cache: {e}")
            return False
    
    def load_features(self, cache_key: str, prefix: str = "features", 
                     max_age_hours: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Load features from cache if valid
        
        Args:
            cache_key: Unique identifier for cached features
            prefix: Prefix for cache filename
            max_age_hours: Maximum age in hours (uses default if None)
            
        Returns:
            Tuple of (DataFrame or None, status message)
        """
        max_age_hours = max_age_hours or self.default_max_age_hours
        cache_path = self.get_cache_path(cache_key, prefix)
        
        try:
            if not os.path.exists(cache_path):
                return None, "Cache file not found"
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache data structure
            if not isinstance(cache_data, dict) or 'features_df' not in cache_data:
                return None, "Invalid cache data structure"
            
            # Check cache age
            created_at = cache_data.get('created_at', datetime.now())
            age_hours = (datetime.now() - created_at).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                return None, f"Cache expired (age: {age_hours:.1f}h, max: {max_age_hours}h)"
            
            features_df = cache_data.get('features_df')
            if features_df is None or features_df.empty:
                return None, "Empty or invalid cached features"
            
            # Version compatibility check
            cache_version = cache_data.get('version', '0.0')
            if cache_version != self.version:
                print(f"⚠️ Cache version mismatch: {cache_version} vs {self.version}")
            
            file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"✅ Features loaded from cache!")
            print(f"   📁 File: {os.path.basename(cache_path)}")
            print(f"   📊 Shape: {features_df.shape}")
            print(f"   ⏰ Age: {age_hours:.1f}h")
            print(f"   💽 Size: {file_size_mb:.1f}MB")
            
            # Display metadata if available
            metadata = cache_data.get('metadata', {})
            if metadata:
                print(f"   📋 Metadata: {metadata}")
            
            return features_df, "Success"
            
        except Exception as e:
            return None, f"Failed to load cache: {str(e)}"
    
    def list_cached_features(self, prefix: str = "features") -> pd.DataFrame:
        """
        List all cached feature files with details
        
        Args:
            prefix: Prefix to filter cache files
            
        Returns:
            DataFrame with cache file information
        """
        cache_files = []
        
        if not os.path.exists(self.cache_dir):
            return pd.DataFrame()
        
        for filename in os.listdir(self.cache_dir):
            if filename.startswith(f"{prefix}_") and filename.endswith('.pkl'):
                cache_path = os.path.join(self.cache_dir, filename)
                
                try:
                    # Get file stats
                    stat = os.stat(cache_path)
                    size_mb = stat.st_size / (1024 * 1024)
                    modified_time = datetime.fromtimestamp(stat.st_mtime)
                    age_hours = (datetime.now() - modified_time).total_seconds() / 3600
                    
                    # Try to load metadata
                    try:
                        with open(cache_path, 'rb') as f:
                            cache_data = pickle.load(f)
                        
                        features_df = cache_data.get('features_df', pd.DataFrame())
                        metadata = cache_data.get('metadata', {})
                        created_at = cache_data.get('created_at', modified_time)
                        cache_key = cache_data.get('cache_key', 'unknown')
                        
                        cache_files.append({
                            'filename': filename,
                            'cache_key': cache_key,
                            'shape': str(features_df.shape),
                            'size_mb': round(size_mb, 2),
                            'age_hours': round(age_hours, 1),
                            'created_at': created_at,
                            'modified_at': modified_time,
                            'date_range': metadata.get('date_range', 'Unknown'),
                            'feature_count': metadata.get('feature_count', len(features_df.columns) - 2 if len(features_df.columns) > 2 else 0),
                            'metadata': str(metadata) if metadata else ''
                        })
                        
                    except Exception as e:
                        cache_files.append({
                            'filename': filename,
                            'cache_key': 'unknown',
                            'shape': 'Error',
                            'size_mb': round(size_mb, 2),
                            'age_hours': round(age_hours, 1),
                            'created_at': modified_time,
                            'modified_at': modified_time,
                            'date_range': 'Error',
                            'feature_count': 0,
                            'metadata': f'Error: {e}'
                        })
                        
                except Exception as e:
                    print(f"⚠️ Error reading cache file {filename}: {e}")
        
        return pd.DataFrame(cache_files)
    
    def clear_cache(self, prefix: str = "features", confirm: bool = False, 
                   older_than_hours: Optional[int] = None) -> int:
        """
        Clear cached feature files
        
        Args:
            prefix: Prefix to filter cache files
            confirm: Must be True to actually delete files
            older_than_hours: Only delete files older than this (None = all)
            
        Returns:
            Number of files deleted
        """
        if not confirm:
            print("⚠️ Use clear_cache(confirm=True) to actually delete files")
            return 0
        
        if not os.path.exists(self.cache_dir):
            print("📁 No cache directory found")
            return 0
        
        deleted_count = 0
        cutoff_time = None
        
        if older_than_hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        for filename in os.listdir(self.cache_dir):
            if filename.startswith(f"{prefix}_") and filename.endswith('.pkl'):
                cache_path = os.path.join(self.cache_dir, filename)
                
                try:
                    # Check age if specified
                    if cutoff_time is not None:
                        modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
                        if modified_time > cutoff_time:
                            continue  # Skip newer files
                    
                    os.remove(cache_path)
                    print(f"🗑️ Deleted: {filename}")
                    deleted_count += 1
                    
                except Exception as e:
                    print(f"❌ Failed to delete {filename}: {e}")
        
        age_str = f" older than {older_than_hours}h" if older_than_hours else ""
        print(f"✅ Cache cleared: {deleted_count} files{age_str} removed")
        return deleted_count
    
    def validate_cache(self, prefix: str = "features") -> Dict[str, Any]:
        """
        Validate all cache files and return summary
        
        Args:
            prefix: Prefix to filter cache files
            
        Returns:
            Dictionary with validation summary
        """
        summary = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'total_size_mb': 0,
            'errors': []
        }
        
        if not os.path.exists(self.cache_dir):
            return summary
        
        for filename in os.listdir(self.cache_dir):
            if filename.startswith(f"{prefix}_") and filename.endswith('.pkl'):
                cache_path = os.path.join(self.cache_dir, filename)
                summary['total_files'] += 1
                
                try:
                    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
                    summary['total_size_mb'] += size_mb
                    
                    # Try to load and validate
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    if (isinstance(cache_data, dict) and 
                        'features_df' in cache_data and 
                        isinstance(cache_data['features_df'], pd.DataFrame) and 
                        not cache_data['features_df'].empty):
                        summary['valid_files'] += 1
                    else:
                        summary['invalid_files'] += 1
                        summary['errors'].append(f"{filename}: Invalid data structure")
                        
                except Exception as e:
                    summary['invalid_files'] += 1
                    summary['errors'].append(f"{filename}: {str(e)}")
        
        summary['total_size_mb'] = round(summary['total_size_mb'], 2)
        return summary


def create_feature_cache(cache_dir: str = None) -> FeatureCache:
    """
    Factory function to create a FeatureCache instance
    
    Args:
        cache_dir: Directory for cache storage
        
    Returns:
        FeatureCache instance
    """
    return FeatureCache(cache_dir)


# Example usage
if __name__ == "__main__":
    # Create cache instance
    cache = FeatureCache()
    
    # Example: Create dummy features
    dummy_features = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100),
        'feature1': range(100),
        'feature2': range(100, 200),
        'target': range(200, 300)
    })
    
    # Save features
    cache_key = cache.generate_data_hash(dummy_features)
    metadata = {'description': 'Test features', 'feature_count': 2}
    cache.save_features(dummy_features, cache_key, metadata)
    
    # Load features
    loaded_features, status = cache.load_features(cache_key)
    print(f"Load status: {status}")
    print(f"Loaded shape: {loaded_features.shape if loaded_features is not None else 'None'}")
    
    # List cache
    cache_list = cache.list_cached_features()
    print(f"Cache files: {len(cache_list)}")
    
    # Validate cache
    validation = cache.validate_cache()
    print(f"Validation: {validation}")
