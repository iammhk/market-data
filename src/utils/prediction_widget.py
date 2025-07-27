# 🎛️ INTERACTIVE PREDICTION WIDGET
# Interactive widget-based manual prediction interface for Bank Nifty

import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np

def create_interactive_prediction_widget(model_results):
    """
    Create interactive widget for manual Bank Nifty predictions
    
    Args:
        model_results (dict): Dictionary containing trained model, scaler, and feature columns
        
    Returns:
        widgets.VBox: Complete interactive widget interface
    """
    
    # Check if model_results is valid
    if not model_results or 'model' not in model_results:
        print("❌ XGBoost model not available. Please train the model first.")
        return None
    
    # Style for widgets
    style = {'description_width': 'initial'}
    layout = widgets.Layout(width='300px')
    
    print("🎯 Creating Interactive Prediction Interface...")
    
    # Create input widgets
    spot_price_widget = widgets.FloatText(
        value=51500.0,
        description='💰 Spot Price (₹):',
        style=style,
        layout=layout
    )
    
    # Call options widgets
    call_volume_widget = widgets.FloatText(
        value=1000000.0,
        description='📈 Call Volume:',
        style=style,
        layout=layout
    )
    
    call_oi_widget = widgets.FloatText(
        value=2000000.0,
        description='📈 Call OI:',
        style=style,
        layout=layout
    )
    
    call_ltp_widget = widgets.FloatText(
        value=150.0,
        description='📈 Call Avg LTP (₹):',
        style=style,
        layout=layout
    )
    
    call_atm_widget = widgets.FloatText(
        value=200.0,
        description='📈 Call ATM LTP (₹):',
        style=style,
        layout=layout
    )
    
    # Put options widgets
    put_volume_widget = widgets.FloatText(
        value=1200000.0,
        description='📉 Put Volume:',
        style=style,
        layout=layout
    )
    
    put_oi_widget = widgets.FloatText(
        value=2200000.0,
        description='📉 Put OI:',
        style=style,
        layout=layout
    )
    
    put_ltp_widget = widgets.FloatText(
        value=140.0,
        description='📉 Put Avg LTP (₹):',
        style=style,
        layout=layout
    )
    
    put_atm_widget = widgets.FloatText(
        value=180.0,
        description='📉 Put ATM LTP (₹):',
        style=style,
        layout=layout
    )
    
    # Control buttons
    predict_button = widgets.Button(
        description='🔮 Make Prediction',
        button_style='success',
        layout=widgets.Layout(width='200px', height='40px')
    )
    
    # Preset buttons
    bullish_button = widgets.Button(
        description='🔥 Bullish Preset',
        button_style='info',
        layout=widgets.Layout(width='150px')
    )
    
    bearish_button = widgets.Button(
        description='📉 Bearish Preset',
        button_style='warning',
        layout=widgets.Layout(width='150px')
    )
    
    neutral_button = widgets.Button(
        description='⚖️ Neutral Preset',
        button_style='',
        layout=widgets.Layout(width='150px')
    )
    
    # Output widget
    output_widget = widgets.Output()
    
    # Preset functions
    def set_bullish_preset(b):
        """Set bullish market preset values"""
        spot_price_widget.value = 51500.0
        call_volume_widget.value = 2000000.0
        put_volume_widget.value = 800000.0
        call_oi_widget.value = 2500000.0
        put_oi_widget.value = 1500000.0
        call_ltp_widget.value = 180.0
        put_ltp_widget.value = 100.0
        call_atm_widget.value = 250.0
        put_atm_widget.value = 120.0
        
        with output_widget:
            clear_output()
            print("🔥 Bullish preset loaded!")
            print("💡 High call volume, low put volume")
            print("📈 Expensive calls, cheap puts")
    
    def set_bearish_preset(b):
        """Set bearish market preset values"""
        spot_price_widget.value = 51500.0
        call_volume_widget.value = 800000.0
        put_volume_widget.value = 2200000.0
        call_oi_widget.value = 1500000.0
        put_oi_widget.value = 2800000.0
        call_ltp_widget.value = 100.0
        put_ltp_widget.value = 200.0
        call_atm_widget.value = 120.0
        put_atm_widget.value = 280.0
        
        with output_widget:
            clear_output()
            print("📉 Bearish preset loaded!")
            print("💡 High put volume, low call volume")
            print("📉 Expensive puts, cheap calls")
    
    def set_neutral_preset(b):
        """Set neutral market preset values"""
        spot_price_widget.value = 51500.0
        call_volume_widget.value = 1200000.0
        put_volume_widget.value = 1300000.0
        call_oi_widget.value = 2000000.0
        put_oi_widget.value = 2100000.0
        call_ltp_widget.value = 150.0
        put_ltp_widget.value = 145.0
        call_atm_widget.value = 180.0
        put_atm_widget.value = 170.0
        
        with output_widget:
            clear_output()
            print("⚖️ Neutral preset loaded!")
            print("💡 Balanced call/put activity")
            print("🔄 Similar option prices")
    
    def make_widget_prediction(b):
        """Make prediction using widget input values"""
        with output_widget:
            clear_output()
            
            try:
                # Get values from widgets
                spot_price = spot_price_widget.value
                call_volume = call_volume_widget.value
                call_oi = call_oi_widget.value
                call_ltp = call_ltp_widget.value
                call_atm = call_atm_widget.value
                put_volume = put_volume_widget.value
                put_oi = put_oi_widget.value
                put_ltp = put_ltp_widget.value
                put_atm = put_atm_widget.value
                
                # Calculate derived features
                total_volume = call_volume + put_volume
                total_oi = call_oi + put_oi
                pcr_volume = put_volume / max(call_volume, 1)
                pcr_oi = put_oi / max(call_oi, 1)
                
                # Get model components
                model = model_results['model']
                scaler = model_results['scaler']
                feature_columns = model_results['feature_columns']
                
                # Create feature dictionary
                features = {}
                for feature in feature_columns:
                    if feature == 'spot_price':
                        features[feature] = spot_price
                    elif feature == 'call_total_volume':
                        features[feature] = call_volume
                    elif feature == 'call_total_oi':
                        features[feature] = call_oi
                    elif feature == 'call_avg_ltp':
                        features[feature] = call_ltp
                    elif feature == 'call_max_ltp':
                        features[feature] = call_ltp * 1.2
                    elif feature == 'call_count':
                        features[feature] = 50
                    elif feature == 'put_total_volume':
                        features[feature] = put_volume
                    elif feature == 'put_total_oi':
                        features[feature] = put_oi
                    elif feature == 'put_avg_ltp':
                        features[feature] = put_ltp
                    elif feature == 'put_max_ltp':
                        features[feature] = put_ltp * 1.2
                    elif feature == 'put_count':
                        features[feature] = 50
                    elif feature == 'total_volume':
                        features[feature] = total_volume
                    elif feature == 'total_oi':
                        features[feature] = total_oi
                    elif feature == 'pcr_volume':
                        features[feature] = pcr_volume
                    elif feature == 'pcr_oi':
                        features[feature] = pcr_oi
                    elif feature == 'call_atm_ltp':
                        features[feature] = call_atm
                    elif feature == 'put_atm_ltp':
                        features[feature] = put_atm
                    else:
                        features[feature] = 0.0
                
                # Make prediction
                feature_df = pd.DataFrame([features])
                feature_scaled = scaler.transform(feature_df)
                predicted_price = model.predict(feature_scaled)[0]
                
                # Calculate results
                difference = predicted_price - spot_price
                difference_pct = (difference / spot_price) * 100
                
                # Display results with formatting
                print("🔮 PREDICTION RESULTS")
                print("=" * 35)
                print(f"💰 Current Price:      ₹{spot_price:,.2f}")
                print(f"🎯 Predicted Price:    ₹{predicted_price:,.2f}")
                print(f"📊 Difference:         ₹{difference:+,.2f}")
                print(f"📈 Change:             {difference_pct:+.2f}%")
                
                # Market signal with colors
                if difference > 0:
                    signal = "📈 BULLISH"
                    recommendation = "💡 Consider CALL options"
                    signal_color = "🟢"
                elif difference < 0:
                    signal = "📉 BEARISH"
                    recommendation = "💡 Consider PUT options"
                    signal_color = "🔴"
                else:
                    signal = "⚖️ NEUTRAL"
                    recommendation = "💡 Consider range strategies"
                    signal_color = "🟡"
                
                print(f"\n🎯 MARKET SIGNAL")
                print("-" * 20)
                print(f"{signal_color} Direction: {signal}")
                print(f"💡 Strategy: {recommendation}")
                
                # Market indicators
                print(f"\n📋 MARKET INDICATORS")
                print("-" * 25)
                print(f"🔄 PCR (Volume):   {pcr_volume:.3f}")
                print(f"🔄 PCR (OI):       {pcr_oi:.3f}")
                print(f"📊 Total Volume:   {total_volume:,.0f}")
                print(f"📊 Total OI:       {total_oi:,.0f}")
                
                # PCR interpretation
                if pcr_volume > 1.2:
                    pcr_signal = "📉 Bearish (High PCR)"
                elif pcr_volume < 0.8:
                    pcr_signal = "📈 Bullish (Low PCR)"
                else:
                    pcr_signal = "⚖️ Neutral PCR"
                print(f"🎯 PCR Signal:     {pcr_signal}")
                
                # Confidence indicator
                confidence = min(abs(difference_pct) * 10, 100)
                confidence_bars = "█" * int(confidence / 10)
                print(f"\n📊 CONFIDENCE")
                print("-" * 15)
                print(f"{confidence_bars} {confidence:.0f}%")
                
                if confidence > 70:
                    print("🔥 High confidence")
                elif confidence > 40:
                    print("📊 Moderate confidence")
                else:
                    print("⚠️ Low confidence")
                
                print(f"\n⚠️ DISCLAIMER: Predictive model, not financial advice")
                
                # Return result for potential further processing
                return {
                    'current_price': spot_price,
                    'predicted_price': predicted_price,
                    'difference': difference,
                    'difference_pct': difference_pct,
                    'signal': signal.replace('📈 ', '').replace('📉 ', '').replace('⚖️ ', ''),
                    'pcr_volume': pcr_volume,
                    'confidence': confidence
                }
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("💡 Please check your model is trained first")
                return None
    
    # Connect button events
    predict_button.on_click(make_widget_prediction)
    bullish_button.on_click(set_bullish_preset)
    bearish_button.on_click(set_bearish_preset)
    neutral_button.on_click(set_neutral_preset)
    
    # Create layout sections
    input_widgets = widgets.VBox([
        widgets.HTML("<h3>💰 Market Data Inputs</h3>"),
        spot_price_widget,
        widgets.HTML("<h4>📈 Call Options</h4>"),
        call_volume_widget,
        call_oi_widget,
        call_ltp_widget,
        call_atm_widget,
        widgets.HTML("<h4>📉 Put Options</h4>"),
        put_volume_widget,
        put_oi_widget,
        put_ltp_widget,
        put_atm_widget,
    ])
    
    control_widgets = widgets.VBox([
        widgets.HTML("<h3>🎛️ Controls</h3>"),
        predict_button,
        widgets.HTML("<h4>📋 Quick Presets</h4>"),
        widgets.HBox([bullish_button, bearish_button, neutral_button]),
        widgets.HTML("<p><b>Instructions:</b></p><ul><li>🔧 Adjust values above</li><li>📋 Use presets for quick testing</li><li>🔮 Click 'Make Prediction' to analyze</li></ul>")
    ])
    
    main_interface = widgets.HBox([
        input_widgets,
        control_widgets
    ])
    
    full_widget = widgets.VBox([
        widgets.HTML("<h2>🎛️ Interactive Bank Nifty Prediction Widget</h2>"),
        main_interface,
        output_widget
    ])
    
    return full_widget

def display_prediction_widget(model_results):
    """
    Create and display the interactive prediction widget
    
    Args:
        model_results (dict): Dictionary containing trained model, scaler, and feature columns
    """
    print("🎛️ INTERACTIVE BANK NIFTY PREDICTION WIDGET")
    print("=" * 50)
    
    try:
        widget = create_interactive_prediction_widget(model_results)
        if widget:
            display(widget)
            print("✅ Interactive prediction widget loaded successfully!")
            print("🎯 Use the interface above to make predictions")
            return widget
        else:
            print("⚠️ Widget not created - model may not be available")
            return None
            
    except ImportError:
        print("❌ ipywidgets not available!")
        print("📦 Install with: pip install ipywidgets")
        print("🔧 Then restart the kernel and try again")
        return None
    except Exception as e:
        print(f"❌ Error creating widget: {str(e)}")
        print("💡 Make sure the XGBoost model is trained first")
        return None

def create_simple_prediction_form(model_results):
    """
    Create a simpler form-based prediction interface (fallback for when widgets aren't available)
    
    Args:
        model_results (dict): Dictionary containing trained model, scaler, and feature columns
    """
    print("📝 SIMPLE PREDICTION FORM (Text-based)")
    print("=" * 45)
    print("💡 Enter values when prompted (or press Enter for defaults)")
    
    def make_simple_prediction():
        """Simple text-based prediction interface"""
        try:
            # Get inputs
            spot_price_input = input("💰 Current Bank Nifty Price (₹) [51500]: ").strip()
            spot_price = float(spot_price_input) if spot_price_input else 51500.0
            
            call_volume_input = input("📈 Call Volume [1000000]: ").strip()
            call_volume = float(call_volume_input) if call_volume_input else 1000000.0
            
            put_volume_input = input("📉 Put Volume [1200000]: ").strip()
            put_volume = float(put_volume_input) if put_volume_input else 1200000.0
            
            call_atm_input = input("🎯 Call ATM LTP (₹) [200]: ").strip()
            call_atm = float(call_atm_input) if call_atm_input else 200.0
            
            put_atm_input = input("🎯 Put ATM LTP (₹) [180]: ").strip() 
            put_atm = float(put_atm_input) if put_atm_input else 180.0
            
            # Create simplified features
            pcr_volume = put_volume / max(call_volume, 1)
            
            # Simple prediction logic here
            print(f"\n🔮 PREDICTION RESULTS")
            print(f"💰 Current Price: ₹{spot_price:,.2f}")
            print(f"🔄 PCR (Volume): {pcr_volume:.3f}")
            
            if pcr_volume > 1.2:
                print("📉 Signal: BEARISH (High PCR)")
            elif pcr_volume < 0.8:
                print("📈 Signal: BULLISH (Low PCR)")
            else:
                print("⚖️ Signal: NEUTRAL")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    return make_simple_prediction
