import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils import get_marketstack_data, calculate_technical_indicators
from models import prepare_data, train_model, predict_next_day
import os
from dotenv import load_dotenv


load_dotenv()


st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# Add disclaimer
st.sidebar.markdown("""
**Disclaimer:**  
This tool is for educational purposes only.  
Stock market predictions are inherently uncertain.  
Never make investment decisions based solely on algorithmic predictions.
""")

# Title and description
st.title("🧠 Stock Price Prediction Tool")
st.write("""
This tool uses machine learning to predict stock prices based on historical data.
Select a stock symbol and date range to train the model and get predictions.
""")

# Sidebar controls
st.sidebar.header("Settings")

# Get available symbols
DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX']
symbol = st.sidebar.selectbox("Stock Symbol", DEFAULT_SYMBOLS)

# Date range selection
end_date = datetime.today()
start_date = end_date - timedelta(days=365*2)  # Default 2 years of data

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", start_date)
with col2:
    end_date = st.date_input("End Date", end_date)

# Model selection
model_type = st.sidebar.radio(
    "Select Model",
    ('xgb', 'rf'),
    index=0,
    format_func=lambda x: 'XGBoost' if x == 'xgb' else 'Random Forest'
)

# Fetch data button
fetch_button = st.sidebar.button("Fetch Data and Train Model")

# Main content area
if fetch_button:
    with st.spinner('Fetching data and training model...'):
        try:
            # Fetch data
            df = get_marketstack_data(symbol, start_date, end_date)
            
            if df.empty:
                st.error("No data returned for the selected symbol and date range.")
            else:
                # Calculate technical indicators
                df = calculate_technical_indicators(df)
                
                # Show raw data
                st.subheader(f"Historical Data for {symbol}")
                st.dataframe(df.tail(10))
                
                # Plot historical prices
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='royalblue')
                ))
                fig1.update_layout(
                    title=f"{symbol} Closing Prices",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Prepare data for ML
                features, target = prepare_data(df)
                
                # Train model
                model_result = train_model(features, target, model_type)
                
                # Show model results
                st.subheader("Model Evaluation")
                st.write(f"Model Type: {'XGBoost' if model_type == 'xgb' else 'Random Forest'}")
                st.write(f"Mean Absolute Error (MAE): ${model_result['mae']:.2f}")
                st.write(f"Root Mean Squared Error (RMSE): ${model_result['rmse']:.2f}")
                st.write(f"95% Confidence Interval: ±${model_result['confidence_interval']:.2f}")
                st.write("Best Hyperparameters:")
                st.json(model_result['best_params'])
                
                # Make prediction for next day
                prediction = predict_next_day(
                    model_result['model'],
                    df,
                    model_result['confidence_interval']
                )
                
                # Display prediction
                st.subheader("Next Day Prediction")
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Predicted Close",
                    f"${prediction['predicted_price']:.2f}",
                    help="Model's prediction for the next trading day's closing price"
                )
                col2.metric(
                    "Upper Bound (95% CI)",
                    f"${prediction['upper_bound']:.2f}",
                    delta=f"+${model_result['confidence_interval']:.2f}",
                    delta_color="off"
                )
                col3.metric(
                    "Lower Bound (95% CI)",
                    f"${prediction['lower_bound']:.2f}",
                    delta=f"-${model_result['confidence_interval']:.2f}",
                    delta_color="off"
                )
                
                # Plot prediction with confidence interval
                last_date = df.index[-1]
                next_date = last_date + pd.Timedelta(days=1)
                
                fig2 = go.Figure()
                
                # Historical data
                fig2.add_trace(go.Scatter(
                    x=df.index[-30:],  # Last 30 days
                    y=df['close'][-30:],
                    mode='lines',
                    name='Historical Close',
                    line=dict(color='royalblue')
                ))
                
                # Prediction point
                fig2.add_trace(go.Scatter(
                    x=[next_date],
                    y=[prediction['predicted_price']],
                    mode='markers',
                    name='Prediction',
                    marker=dict(color='green', size=10)
                ))
                
                # Confidence interval
                fig2.add_trace(go.Scatter(
                    x=[next_date, next_date],
                    y=[prediction['lower_bound'], prediction['upper_bound']],
                    mode='lines',
                    name='Confidence Interval',
                    line=dict(color='gray', width=2, dash='dash')
                ))
                
                fig2.update_layout(
                    title=f"{symbol} Price Prediction for {next_date.strftime('%Y-%m-%d')}",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode="x unified",
                    showlegend=True
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Feature importance plot
                if 'feature_names' in model_result:
                    try:
                        if hasattr(model_result['model'].named_steps[model_type], 'feature_importances_'):
                            importances = model_result['model'].named_steps[model_type].feature_importances_
                            feature_names = model_result['feature_names']
                            
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False)
                            
                            fig3 = go.Figure()
                            fig3.add_trace(go.Bar(
                                x=importance_df['Importance'],
                                y=importance_df['Feature'],
                                orientation='h',
                                marker=dict(color='teal')
                            ))
                            fig3.update_layout(
                                title="Feature Importance",
                                xaxis_title="Importance Score",
                                yaxis_title="Feature",
                                height=600
                            )
                            st.plotly_chart(fig3, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display feature importance: {str(e)}")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add some instructions when the app first loads
if not fetch_button:
    st.info("""
    **Instructions:**
    1. Select a stock symbol from the sidebar
    2. Choose a date range for historical data
    3. Select a machine learning model
    4. Click "Fetch Data and Train Model" to see predictions
    """)
    
    st.write("""
    **Note:**  
    The models are trained on the fly with your selected parameters.  
    For more accurate predictions, consider using a longer historical period.
    """)