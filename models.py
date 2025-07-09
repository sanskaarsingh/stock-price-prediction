import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
from datetime import datetime

def prepare_data(df, forecast_days=1):
    """Prepare features and target variable for training"""
    df = df.copy()
    df['target'] = df['close'].shift(-forecast_days) 
    df.dropna(subset=['target'], inplace=True)
    features = df.select_dtypes(include=[np.number]).drop(columns=['target'], errors='ignore')
    return features, df['target']

def train_model(features, target, model_type='xgb'):
    """Train and evaluate the model"""
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )
    
    if model_type == 'xgb':
        model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
        ])
        param_grid = {
            'xgb__max_depth': [3, 5],
            'xgb__learning_rate': [0.01, 0.1],
        }
    elif model_type == 'rf':
        model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('rf', RandomForestRegressor(random_state=42))
        ])
        param_grid = {
            'rf__max_depth': [5, 10], 
            'rf__min_samples_split': [2, 5],
        }
    
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    
    y_pred = grid_search.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    ci = 1.96 * np.std(y_test - y_pred)
    
    return {
        'model': grid_search.best_estimator_,
        'mae': mae,
        'rmse': rmse,
        'confidence_interval': ci,
        'best_params': grid_search.best_params_,
    }

def predict_next_day(model, last_available_data, confidence_interval):
    """Predict today's closing price using the latest available data"""
    try:
       
        features = last_available_data.select_dtypes(include=[np.number])
        if 'target' in features.columns:
            features = features.drop(columns=['target'])
        features = features.iloc[-1:] 
        
       
        predicted_price = model.predict(features)[0]
        
        return {
            'prediction_date': datetime.today().strftime('%Y-%m-%d'),
            'predicted_price': predicted_price,
            'upper_bound': predicted_price + confidence_interval,
            'lower_bound': predicted_price - confidence_interval,
        }
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")