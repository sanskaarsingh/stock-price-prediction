import requests
import streamlit as st
from datetime import datetime, timedelta

def fetch_marketaux_news(api_key, ticker="AAPL", limit=3):
    
    url = f"https://api.marketaux.com/v1/news/all?symbols={ticker}&filter_entities=true&language=en&api_token={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["data"][:limit]  
        else:
            st.error(f"API Error: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return []

def display_news(news_data):
    
    for item in news_data:
        published_at = datetime.strptime(item["published_at"], "%Y-%m-%dT%H:%M:%SZ").strftime("%b %d, %Y %I:%M %p")
        
        st.subheader(item["title"])
        st.caption(f"**Source:** {item['source']} | **Published:** {published_at}")
        st.write(item["description"])
        st.markdown(f"[Read more]({item['url']})")
        st.divider()