"""
🔗 MONITORING INTEGRATION UTILITIES
Functions to connect the monitoring system to prediction functions

This module provides utilities to integrate the monitoring system with
live prediction functions, including enhanced prediction wrappers and
integration setup functions.
"""

from typing import Dict, Any, Optional, Callable
import sys


def setup_monitoring_integration(monitor_instance, globals_dict: Dict[str, Any]) -> bool:
    """
    Setup monitoring integration with prediction functions
    
    Args:
        monitor_instance: The PredictionMonitor instance
        globals_dict: The global namespace dictionary (usually globals())
        
    Returns:
        bool: True if integration successful, False otherwise
    """
    
    # Check if required objects exist
    if monitor_instance is None:
        print("❌ Monitor not available. Please run the monitoring cell first.")
        return False
    
    if 'make_live_prediction' not in globals_dict:
        print("⚠️ make_live_prediction function not yet available.")
        print("💡 Run this cell again after the prediction engine is loaded.")
        return False
    
    # Get the prediction function
    make_live_prediction = globals_dict['make_live_prediction']
    
    # Create the enhanced prediction function with monitoring
    def enhanced_live_prediction():
        """Enhanced prediction function with monitoring"""
        try:
            # Use the imported enhanced_live_prediction from utils if available
            try:
                from utils.monitoring_system import enhanced_live_prediction as utils_enhanced_prediction
                result = utils_enhanced_prediction(make_live_prediction)
            except (ImportError, AttributeError):
                # Fallback to basic enhanced prediction
                result = make_live_prediction(use_previous_day_data=True)
                
                if result and result.get('success'):
                    # Additional analysis
                    features = result['features_used']
                    
                    print(f"\n🔍 ENHANCED ANALYSIS")
                    print("-" * 20)
                    
                    # Market sentiment analysis
                    pcr = features.get('pcr_volume', 1)
                    call_vol = features.get('call_total_volume', 0)
                    put_vol = features.get('put_total_volume', 0)
                    
                    if pcr > 1.2:
                        sentiment = "📉 Bearish (Fear dominant)"
                    elif pcr < 0.8:
                        sentiment = "📈 Bullish (Greed dominant)"
                    else:
                        sentiment = "⚖️ Neutral (Balanced)"
                    
                    print(f"🎭 Market Sentiment: {sentiment}")
                    print(f"📊 Call Volume: {call_vol:,.0f}")
                    print(f"📊 Put Volume: {put_vol:,.0f}")
                    
                    # Volume analysis
                    total_vol = features.get('total_volume', 0)
                    if total_vol > 1000000:
                        vol_signal = "🔥 High activity"
                    elif total_vol > 500000:
                        vol_signal = "📊 Moderate activity"
                    else:
                        vol_signal = "😴 Low activity"
                    
                    print(f"📈 Volume Signal: {vol_signal}")
                    
                    # Price momentum
                    price_change = features.get('price_change_pct', 0)
                    if abs(price_change) > 0.5:
                        momentum = f"🚀 Strong momentum ({price_change:+.2f}%)"
                    else:
                        momentum = f"🐌 Weak momentum ({price_change:+.2f}%)"
                    
                    print(f"⚡ Momentum: {momentum}")
            
            # Log the prediction if monitor is available
            if result and result.get('success') and monitor_instance:
                monitor_instance.log_prediction(result)
                
            return result
            
        except Exception as e:
            print(f"❌ Error in enhanced prediction: {e}")
            return None

    def quick_prediction():
        """Quick prediction with monitoring"""
        return enhanced_live_prediction()

    # Update global functions
    globals_dict['enhanced_live_prediction'] = enhanced_live_prediction
    globals_dict['quick_prediction'] = quick_prediction
    
    print("✅ MONITORING INTEGRATION COMPLETE!")
    print("-" * 40)
    print("🔗 Updated functions:")
    print("   • enhanced_live_prediction() - Full prediction with monitoring and analysis")
    print("   • quick_prediction() - Same as enhanced_live_prediction()")
    print("   • start_monitoring() - Show dashboard")
    print("   • show_trends() - Plot trends")
    print("   • export_data() - Export predictions to CSV")
    
    return True


def create_enhanced_prediction_wrapper(make_live_prediction_func: Callable, 
                                     monitor_instance: Optional[Any] = None) -> Callable:
    """
    Create an enhanced prediction wrapper function
    
    Args:
        make_live_prediction_func: The base prediction function to wrap
        monitor_instance: Optional PredictionMonitor instance for logging
        
    Returns:
        Callable: Enhanced prediction function with monitoring
    """
    
    def enhanced_live_prediction():
        """Enhanced prediction function with monitoring"""
        try:
            # Use the imported enhanced_live_prediction from utils if available
            try:
                from utils.monitoring_system import enhanced_live_prediction as utils_enhanced_prediction
                result = utils_enhanced_prediction(make_live_prediction_func)
            except (ImportError, AttributeError):
                # Fallback to basic enhanced prediction
                result = make_live_prediction_func(use_previous_day_data=True)
                
                if result and result.get('success'):
                    # Additional analysis
                    features = result['features_used']
                    
                    print(f"\n🔍 ENHANCED ANALYSIS")
                    print("-" * 20)
                    
                    # Market sentiment analysis
                    pcr = features.get('pcr_volume', 1)
                    call_vol = features.get('call_total_volume', 0)
                    put_vol = features.get('put_total_volume', 0)
                    
                    if pcr > 1.2:
                        sentiment = "📉 Bearish (Fear dominant)"
                    elif pcr < 0.8:
                        sentiment = "📈 Bullish (Greed dominant)"
                    else:
                        sentiment = "⚖️ Neutral (Balanced)"
                    
                    print(f"🎭 Market Sentiment: {sentiment}")
                    print(f"📊 Call Volume: {call_vol:,.0f}")
                    print(f"📊 Put Volume: {put_vol:,.0f}")
                    
                    # Volume analysis
                    total_vol = features.get('total_volume', 0)
                    if total_vol > 1000000:
                        vol_signal = "🔥 High activity"
                    elif total_vol > 500000:
                        vol_signal = "📊 Moderate activity"
                    else:
                        vol_signal = "😴 Low activity"
                    
                    print(f"📈 Volume Signal: {vol_signal}")
                    
                    # Price momentum
                    price_change = features.get('price_change_pct', 0)
                    if abs(price_change) > 0.5:
                        momentum = f"🚀 Strong momentum ({price_change:+.2f}%)"
                    else:
                        momentum = f"🐌 Weak momentum ({price_change:+.2f}%)"
                    
                    print(f"⚡ Momentum: {momentum}")
            
            # Log the prediction if monitor is available
            if result and result.get('success') and monitor_instance:
                monitor_instance.log_prediction(result)
                
            return result
            
        except Exception as e:
            print(f"❌ Error in enhanced prediction: {e}")
            return None
    
    return enhanced_live_prediction


def create_quick_prediction_wrapper(enhanced_prediction_func: Callable) -> Callable:
    """
    Create a quick prediction wrapper function
    
    Args:
        enhanced_prediction_func: The enhanced prediction function
        
    Returns:
        Callable: Quick prediction function
    """
    def quick_prediction():
        """Quick prediction with monitoring"""
        return enhanced_prediction_func()
    
    return quick_prediction


def initialize_integration_system(monitor_instance, globals_dict: Dict[str, Any]) -> Dict[str, Callable]:
    """
    Initialize the complete integration system
    
    Args:
        monitor_instance: The PredictionMonitor instance
        globals_dict: The global namespace dictionary
        
    Returns:
        Dict[str, Callable]: Dictionary of integration functions
    """
    
    def setup_integration():
        """Setup monitoring integration wrapper"""
        return setup_monitoring_integration(monitor_instance, globals_dict)
    
    def get_integration_status():
        """Get integration status"""
        has_monitor = monitor_instance is not None
        has_prediction = 'make_live_prediction' in globals_dict
        
        print(f"\n🔗 INTEGRATION STATUS")
        print("-" * 25)
        print(f"📊 Monitor Available: {'✅ Yes' if has_monitor else '❌ No'}")
        print(f"🤖 Prediction Function: {'✅ Available' if has_prediction else '❌ Not Found'}")
        print(f"🔗 Integration Ready: {'✅ Yes' if has_monitor and has_prediction else '❌ No'}")
        
        if has_monitor and has_prediction:
            print(f"\n💡 Ready to use:")
            print("   • enhanced_live_prediction()")
            print("   • quick_prediction()")
        else:
            print(f"\n💡 Next steps:")
            if not has_monitor:
                print("   • Load monitoring system")
            if not has_prediction:
                print("   • Load prediction engine")
    
    def reset_integration():
        """Reset integration functions"""
        functions_to_remove = [
            'enhanced_live_prediction',
            'quick_prediction'
        ]
        
        removed_count = 0
        for func_name in functions_to_remove:
            if func_name in globals_dict:
                del globals_dict[func_name]
                removed_count += 1
        
        print(f"🔄 Integration reset complete! Removed {removed_count} functions.")
    
    return {
        'setup_integration': setup_integration,
        'get_integration_status': get_integration_status,
        'reset_integration': reset_integration
    }


# Export main functions
__all__ = [
    'setup_monitoring_integration',
    'create_enhanced_prediction_wrapper', 
    'create_quick_prediction_wrapper',
    'initialize_integration_system'
]
