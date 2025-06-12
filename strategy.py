# ML Strategy
# Strategy Type: custom
# Description: Machine learning model using Random Forest for price prediction with feature engineering
# Created: 2025-06-12T23:33:21.103Z

# Machine Learning Strategy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ML Strategy Parameters
FEATURE_WINDOW = 20
PREDICTION_HORIZON = 5
CONFIDENCE_THRESHOLD = 0.6

def create_technical_features(prices, volume=None):
    """Create technical analysis features"""
    df = pd.DataFrame({'price': prices})
    
    # Price-based features
    df['sma_5'] = df['price'].rolling(5).mean()
    df['sma_20'] = df['price'].rolling(20).mean()
    df['ema_12'] = df['price'].ewm(span=12).mean()
    
    # Momentum indicators
    df['rsi'] = calculate_rsi(df['price'], 14)
    df['price_change_5d'] = df['price'].pct_change(5)
    df['price_change_20d'] = df['price'].pct_change(20)
    
    # Volatility features
    df['volatility_10d'] = df['price'].pct_change().rolling(10).std()
    df['volatility_20d'] = df['price'].pct_change().rolling(20).std()
    
    # Mean reversion features
    df['price_vs_sma20'] = (df['price'] - df['sma_20']) / df['sma_20']
    df['sma_ratio'] = df['sma_5'] / df['sma_20']
    
    return df

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def simple_ml_prediction(features):
    """Simplified ML prediction (simulated Random Forest)"""
    # Simulate feature importance weights
    feature_weights = {
        'rsi': -0.15,           # Contrarian signal
        'price_vs_sma20': -0.20, # Mean reversion
        'sma_ratio': 0.25,      # Trend following
        'price_change_5d': 0.10, # Short-term momentum
        'price_change_20d': 0.15, # Long-term momentum
        'volatility_10d': -0.10,  # High vol = negative
        'volatility_20d': -0.05   # Long-term vol
    }
    
    # Calculate prediction score
    prediction_score = 0
    for feature, weight in feature_weights.items():
        if feature in features and not pd.isna(features[feature]):
            normalized_value = np.tanh(features[feature])  # Normalize to [-1, 1]
            prediction_score += weight * normalized_value
    
    # Convert to probability
    probability = 1 / (1 + np.exp(-prediction_score * 3))  # Sigmoid
    
    return probability, prediction_score

def ml_strategy():
    """Main ML strategy implementation"""
    print("=== Machine Learning Strategy ===")
    
    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Create realistic price series with trend and noise
    trend = np.linspace(100, 120, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    cycle = 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 50)
    prices = pd.Series(trend + noise + cycle, index=dates)
    
    # Create features
    df = create_technical_features(prices)
    
    print("Feature Engineering Complete")
    print(f"Data points: {len(df)}")
    print(f"Features: {list(df.columns)}")
    
    # Make predictions for recent data
    predictions = []
    signals = []
    
    print("
ML Predictions (Last 10 Days):")
    print("-" * 60)
    print(f"{'Date':<12} {'Price':<8} {'Pred':<6} {'Conf':<6} {'Signal':<8} {'Features'}")
    print("-" * 60)
    
    for i in range(max(0, len(df)-10), len(df)):
        if i < FEATURE_WINDOW:
            continue
            
        row = df.iloc[i]
        features = {
            'rsi': row['rsi'],
            'price_vs_sma20': row['price_vs_sma20'],
            'sma_ratio': row['sma_ratio'],
            'price_change_5d': row['price_change_5d'],
            'price_change_20d': row['price_change_20d'],
            'volatility_10d': row['volatility_10d'],
            'volatility_20d': row['volatility_20d']
        }
        
        # Make prediction
        prob, score = simple_ml_prediction(features)
        
        # Generate signal
        if prob > 0.5 + CONFIDENCE_THRESHOLD/2:
            signal = "BUY"
        elif prob < 0.5 - CONFIDENCE_THRESHOLD/2:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        predictions.append(prob)
        signals.append(signal)
        
        # Display key features
        key_features = f"RSI:{row['rsi']:.1f} SMA:{row['sma_ratio']:.3f}"
        
        print(f"{dates[i].strftime('%Y-%m-%d'):<12} {row['price']:<8.2f} {prob:<6.3f} {abs(prob-0.5)*2:<6.3f} {signal:<8} {key_features}")
    
    # Strategy performance summary
    buy_signals = signals.count('BUY')
    sell_signals = signals.count('SELL')
    hold_signals = signals.count('HOLD')
    
    print(f"
Strategy Summary:")
    print(f"Buy Signals: {buy_signals}")
    print(f"Sell Signals: {sell_signals}")
    print(f"Hold Signals: {hold_signals}")
    print(f"Average Confidence: {np.mean([abs(p-0.5)*2 for p in predictions]):.3f}")
    
    # Simulate strategy return
    strategy_return = (buy_signals - sell_signals) * 2.5 + np.random.normal(0, 1)
    print(f"Simulated Strategy Return: {strategy_return:.2f}%")
    
    return strategy_return

# Execute strategy
if __name__ == "__main__":
    ml_strategy()

# Strategy Analysis and Performance
# Add your backtesting results and analysis here

# Risk Management
# Document your risk parameters and constraints

# Performance Metrics
# Track your strategy's key performance indicators:
# - Sharpe Ratio
# - Maximum Drawdown
# - Win Rate
# - Average Return
