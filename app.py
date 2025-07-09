import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils import get_marketstack_data, calculate_technical_indicators
from models import prepare_data, train_model, predict_next_day
import os
from dotenv import load_dotenv
import requests
from textblob import TextBlob
from dateutil import parser


load_dotenv()


st.set_page_config(
    page_title="Stock Prediction Pro",
    page_icon="📊",
    layout="wide"
)


def analyze_sentiment(text):
    """Returns sentiment polarity (-1 to 1) and classification"""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return polarity, "positive"
    elif polarity < -0.1:
        return polarity, "negative"
    else:
        return polarity, "neutral"
    
def fetch_marketaux_news(ticker):
    """Fetch financial news from Marketaux API"""
    api_key = os.getenv("MARKETAUX_API_KEY")
    url = f"https://api.marketaux.com/v1/news/all?symbols={ticker}&filter_entities=true&language=en&api_token={api_key}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        st.error(f"News API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Failed to fetch news: {str(e)}")
    return []

def display_news(news_data):
    """Render news cards with sentiment indicators"""
    for item in news_data:
        with st.container():
            try:
                
                polarity, sentiment = analyze_sentiment(item["title"] + " " + item["description"])
                
               
                sentiment_color = {
                    "positive": "#4CAF50",  
                    "negative": "#F44336",   
                    "neutral": "#FFC107"   
                }.get(sentiment, "#000000")
                
                
                col1, col2 = st.columns([0.85, 0.15])
                
                with col1:
                   
                    st.markdown(f"""
                    <div style="display: flex; align-items: center;">
                        <div style="
                            height: 12px;
                            width: 12px;
                            background-color: {sentiment_color};
                            border-radius: 50%;
                            margin-right: 8px;
                        "></div>
                        <h4 style="margin: 0;">{item['title']}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"{item['source']} · {parser.parse(item['published_at']).strftime('%b %d, %Y %I:%M %p')}")
                    st.write(item["description"])
                    st.markdown(f"[Read more]({item['url']})")
                
                with col2:
                    
                    st.markdown(f"""
                    <div style="
                        background: rgba(0, 0, 0, 0) !important;  /* Fully transparent */
                        border-radius: 10px;
                        padding: 5px;
                        margin-top: 10px;
                        border: 0px solid {sentiment_color};  /* Optional: Add border for visibility */
                    ">
                        <div style="
                            width: {abs(polarity)*100}%;
                            height: 20px;
                            background: {sentiment_color};
                            border-radius: 8px;
                            margin-left: {50-(abs(polarity)*50)}%;
                        "></div>
                        <p style="
                            text-align: center; 
                            margin: 5px 0; 
                            font-weight: bold; 
                            color: {sentiment_color};
                            text-shadow: 0 0 2px #000;  /* Improves text readability */
                        ">
                            {sentiment.upper()}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
            
            except Exception as e:
                st.warning(f"Couldn't process news item: {str(e)}")
                continue


st.sidebar.markdown("""
**Disclaimer:**  
This tool is for educational purposes only.  
Stock market predictions are inherently uncertain.  
Never make investment decisions based solely on algorithmic predictions.
""")

st.title("🧠 Stock Prediction 2.0")
st.write("""
Advanced stock price prediction with market news analysis.
Select a stock symbol and date range to get started.
""")


st.sidebar.header("Settings")

DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX']
symbol = st.sidebar.selectbox("Stock Symbol", DEFAULT_SYMBOLS)

end_date = datetime.today()
start_date = end_date - timedelta(days=365*2)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", start_date)
with col2:
    end_date = st.date_input("End Date", end_date)

model_type = st.sidebar.radio(
    "Select Model",
    ('xgb', 'rf'),
    index=0,
    format_func=lambda x: 'XGBoost' if x == 'xgb' else 'Random Forest'
)

fetch_button = st.sidebar.button("Fetch Data and Train Model")


if fetch_button:
    with st.spinner('Fetching data and training model...'):
        try:
           
            df = get_marketstack_data(symbol, start_date, end_date)
            
            if df.empty:
                st.error("No data returned for the selected symbol and date range.")
            else:
                df = calculate_technical_indicators(df)
                
                
                st.subheader(f"Historical Data for {symbol}")
                st.dataframe(df.tail(10))
                
                
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
                
               
                features, target = prepare_data(df)
                model_result = train_model(features, target, model_type)
                
               
                st.subheader("Model Evaluation")
                st.write(f"Model Type: {'XGBoost' if model_type == 'xgb' else 'Random Forest'}")
                st.write(f"Mean Absolute Error (MAE): ${model_result['mae']:.2f}")
                st.write(f"Root Mean Squared Error (RMSE): ${model_result['rmse']:.2f}")
                st.write(f"95% Confidence Interval: ±${model_result['confidence_interval']:.2f}")
                st.write("Best Hyperparameters:")
                st.json(model_result['best_params'])
                
                
                prediction = predict_next_day(
                    model_result['model'],
                    df,
                    model_result['confidence_interval']
                )
                
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
                
                
                last_date = df.index[-1]
                next_date = last_date + pd.Timedelta(days=1)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df.index[-30:],  
                    y=df['close'][-30:],
                    mode='lines',
                    name='Historical Close',
                    line=dict(color='green')
                ))
                fig2.add_trace(go.Scatter(
                    x=[next_date],
                    y=[prediction['predicted_price']],
                    mode='markers',
                    name='Prediction',
                    marker=dict(color='green', size=10)
                ))
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
            
            
            st.markdown("---")
            st.subheader(f"Latest {symbol} News")
            news_data = fetch_marketaux_news(symbol)
            if news_data:
                display_news(news_data[:5])  
            if news_data:
                st.subheader("Predicted News Impact")
    
               
                total_impact = sum(analyze_sentiment(n["title"] + " " + n["description"])[0] for n in news_data[:5])
                impact_percent = min(max(total_impact * 2, -5), 5) 
    
                if abs(impact_percent) > 0.5: 
                    direction = "increase" if impact_percent > 0 else "decrease"
                    col1, col2 = st.columns(2)
        
                    with col1:
                        st.metric(
                            label="Predicted Price Impact",
                            value=f"{abs(impact_percent):.1f}% {direction}",
                            delta_color="inverse" if impact_percent < 0 else "normal"
            )
        
                    with col2:
                        st.write("""
                            *Based on sentiment analysis of recent news.
                            Actual market movement may vary.*
                            """)
            else:
                st.warning("No recent news found for this stock")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

else:
    
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

   
    st.subheader("Try these popular stocks:")
    cols = st.columns(4)
    for i, sym in enumerate(DEFAULT_SYMBOLS[:4]):
        with cols[i]:
            if st.button(sym):
                st.session_state.symbol = sym
                st.rerun()