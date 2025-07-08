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

def prepare_data(df, forecast_days=1):
    """
    Prepare data for machine learning by creating features and target
    """
    try:
        if df.empty:
            raise ValueError("Empty DataFrame provided")
            
        
        df['target'] = df['close'].shift(-forecast_days)
        
       
        df.dropna(subset=['target'], inplace=True)
        
       
        features = df.select_dtypes(include=[np.number]).drop(columns=['target'], errors='ignore')
        target = df['target']
        
        if features.empty or len(target) == 0:
            raise ValueError("No valid features or target after preprocessing")
            
        return features, target
        
    except Exception as e:
        raise ValueError(f"Error preparing data: {str(e)}")

def train_model(features, target, model_type='xgb'):
    """
    Train and evaluate a machine learning model
    """
    try:
       
        features = features.select_dtypes(include=[np.number])
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, shuffle=False)
        
        if model_type == 'xgb':
            
            model = Pipeline([
                ('scaler', MinMaxScaler()),
                ('xgb', XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=100,
                    random_state=42
                ))
            ])
            
            param_grid = {
                'xgb__max_depth': [3, 5],
                'xgb__learning_rate': [0.01, 0.1],
                'xgb__subsample': [0.8, 1.0]
            }
            
        elif model_type == 'rf':
           
            model = Pipeline([
                ('scaler', MinMaxScaler()),
                ('rf', RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                ))
            ])
            
            param_grid = {
                'rf__max_depth': [None, 5],
                'rf__min_samples_split': [2, 5]
            }
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                verbose=0,
                error_score='raise'
            )
            
            grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
       
        residuals = y_test - y_pred
        std_residuals = np.std(residuals)
        confidence_interval = 1.96 * std_residuals  
        
        evaluation = {
            'model': best_model,
            'mae': mae,
            'rmse': rmse,
            'confidence_interval': confidence_interval,
            'residuals_std': std_residuals,
            'best_params': grid_search.best_params_,
            'feature_names': features.columns.tolist()
        }
        
        return evaluation
        
    except Exception as e:
        raise ValueError(f"Model training failed: {str(e)}")

def predict_next_day(model, last_available_data, confidence_interval):
    """
    Predict the next day's closing price with confidence interval
    """
    try:
        
        last_data = last_available_data.select_dtypes(include=[np.number])
        if 'target' in last_data.columns:
            last_data = last_data.drop(columns=['target'])
        last_data = last_data.iloc[-1:].copy()
        
        if last_data.empty:
            raise ValueError("No valid data for prediction")
            
        # Make prediction
        predicted_price = model.predict(last_data)[0]
        
        return {
            'predicted_price': predicted_price,
            'upper_bound': predicted_price + confidence_interval,
            'lower_bound': predicted_price - confidence_interval
        }
        
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")