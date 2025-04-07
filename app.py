import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Load stock symbols
@st.cache_data
def load_stock_symbols():
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", 
        "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "LT.NS", "SBIN.NS", "BAJFINANCE.NS",
        "ASIANPAINT.NS", "HDFC.NS", "MARUTI.NS", "TITAN.NS",
        "NESTLEIND.NS", "ONGC.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS"
    ]

def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

def calculate_technical_indicators(df):
    # Trend Indicators
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
    df['EMA_10'] = EMAIndicator(df['Close'], window=10).ema_indicator()
    
    indicator_macd = MACD(df['Close'])
    df['MACD'] = indicator_macd.macd()
    df['MACD_signal'] = indicator_macd.macd_signal()
    df['MACD_hist'] = indicator_macd.macd_diff()
    
    # Momentum Indicators
    indicator_rsi = RSIIndicator(df['Close'], window=14)
    df['RSI'] = indicator_rsi.rsi()
    
    indicator_so = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
    df['SO_%K'] = indicator_so.stoch()
    df['SO_%D'] = indicator_so.stoch_signal()
    
    # Volatility Indicators
    indicator_bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = indicator_bb.bollinger_hband()
    df['BB_middle'] = indicator_bb.bollinger_mavg()
    df['BB_lower'] = indicator_bb.bollinger_lband()
    
    indicator_atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    df['ATR'] = indicator_atr.average_true_range()
    
    # Volume Indicators
    indicator_vwap = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'])
    df['VWAP'] = indicator_vwap.volume_weighted_average_price()
    
    return df

def generate_trading_signals(df):
    signals = []
    last_row = df.iloc[-1]
    
    # MACD Signal
    if last_row['MACD'] > last_row['MACD_signal']:
        signals.append(("MACD", "Bullish (MACD above signal line)", 1))
    else:
        signals.append(("MACD", "Bearish (MACD below signal line)", -1))
    
    # RSI Signal
    if last_row['RSI'] > 70:
        signals.append(("RSI", "Overbought (>70)", -1))
    elif last_row['RSI'] < 30:
        signals.append(("RSI", "Oversold (<30)", 1))
    else:
        signals.append(("RSI", "Neutral (30-70)", 0))
    
    # Bollinger Bands Signal
    if last_row['Close'] < last_row['BB_lower']:
        signals.append(("Bollinger Bands", "Potential Buy (Price below lower band)", 1))
    elif last_row['Close'] > last_row['BB_upper']:
        signals.append(("Bollinger Bands", "Potential Sell (Price above upper band)", -1))
    else:
        signals.append(("Bollinger Bands", "Neutral (Price within bands)", 0))
    
    # Moving Averages Signal
    if last_row['SMA_50'] > last_row['SMA_200']:
        signals.append(("Moving Averages", "Golden Cross (50-day above 200-day)", 1))
    else:
        signals.append(("Moving Averages", "Death Cross (50-day below 200-day)", -1))
    
    # Stochastic Oscillator
    if last_row['SO_%K'] > 80 and last_row['SO_%D'] > 80:
        signals.append(("Stochastic", "Overbought (>80)", -1))
    elif last_row['SO_%K'] < 20 and last_row['SO_%D'] < 20:
        signals.append(("Stochastic", "Oversold (<20)", 1))
    else:
        signals.append(("Stochastic", "Neutral (20-80)", 0))
    
    # Price vs VWAP
    if last_row['Close'] > last_row['VWAP']:
        signals.append(("VWAP", "Price above VWAP (Bullish)", 1))
    else:
        signals.append(("VWAP", "Price below VWAP (Bearish)", -1))
    
    return signals

def predict_next_week_ohlc(df):
    try:
        # Prepare data for prediction
        df = df.copy()
        df['Date'] = df.index
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days
        df = df.dropna()
        
        # Get ATR for volatility measure
        atr = df['ATR'].iloc[-1]
        
        # Predict Close prices using multiple methods
        # Method 1: Linear Regression
        X = df[['Days']].values[-30:]  # Use last 30 days
        y_close = df['Close'].values[-30:]
        
        lr_model = LinearRegression()
        lr_model.fit(X, y_close)
        next_days = np.array([X[-1] + i + 1 for i in range(7)]).reshape(-1, 1)
        lr_predictions = lr_model.predict(next_days)
        
        # Method 2: ARIMA
        arima_model = ARIMA(y_close, order=(5,1,0))
        arima_model_fit = arima_model.fit()
        arima_predictions = arima_model_fit.forecast(steps=7)
        
        # Combine predictions (weighted average)
        combined_close = (lr_predictions * 0.6 + arima_predictions * 0.4)
        
        # Generate OHLC data based on predicted close and volatility
        predicted_ohlc = []
        prev_close = df['Close'].iloc[-1]
        
        for i in range(7):
            # Base the prediction on the combined close
            close = combined_close[i]
            
            # Calculate open (previous close with small random variation)
            open_price = prev_close * (1 + np.random.uniform(-0.005, 0.005))
            
            # Calculate high and low based on ATR (volatility)
            high = close * (1 + np.random.uniform(0, atr/close))
            low = close * (1 - np.random.uniform(0, atr/close))
            
            # Ensure high > low and proper ordering
            high = max(open_price, close, high)
            low = min(open_price, close, low)
            
            predicted_ohlc.append({
                'Date': datetime.today() + timedelta(days=i+1),
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close
            })
            
            prev_close = close
        
        predicted_df = pd.DataFrame(predicted_ohlc)
        predicted_df.set_index('Date', inplace=True)
        
        # Calculate trend
        current_price = df['Close'].iloc[-1]
        predicted_end_price = predicted_df['Close'].iloc[-1]
        percent_change = ((predicted_end_price - current_price) / current_price) * 100
        
        if percent_change > 3:
            trend = "Strong Bullish"
            confidence = "High"
        elif percent_change > 1:
            trend = "Bullish"
            confidence = "Medium"
        elif percent_change < -3:
            trend = "Strong Bearish"
            confidence = "High"
        elif percent_change < -1:
            trend = "Bearish"
            confidence = "Medium"
        else:
            trend = "Neutral"
            confidence = "Low"
        
        return {
            "current_price": current_price,
            "predicted_ohlc": predicted_df,
            "predicted_end_price": predicted_end_price,
            "percent_change": percent_change,
            "trend": trend,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def plot_stock_data(df, ticker):
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_upper'],
        line=dict(color='rgba(255, 0, 0, 0.5)'),
        name='Upper Band'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_middle'],
        line=dict(color='rgba(0, 0, 255, 0.5)'),
        name='Middle Band'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_lower'],
        line=dict(color='rgba(0, 255, 0, 0.5)'),
        name='Lower Band'
    ))
    
    # Moving Averages
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_50'],
        line=dict(color='orange', width=1.5),
        name='50-day SMA'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_200'],
        line=dict(color='purple', width=1.5),
        name='200-day SMA'
    ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price with Technical Indicators',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_macd(df):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        line=dict(color='blue', width=2),
        name='MACD'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD_signal'],
        line=dict(color='red', width=2),
        name='Signal Line'
    ))
    
    # Histogram
    colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD_hist'],
        marker_color=colors,
        name='Histogram'
    ))
    
    fig.update_layout(
        title='MACD Indicator',
        xaxis_title='Date',
        yaxis_title='Value',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi(df):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        line=dict(color='purple', width=2),
        name='RSI'
    ))
    
    # Add overbought and oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI Value',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_ohlc_prediction(current_data, predicted_data, ticker):
    # Combine last 5 days of actual data with predicted data
    last_actual = current_data.iloc[-5:]
    combined = pd.concat([last_actual, predicted_data])
    
    fig = go.Figure()
    
    # Actual OHLC (last 5 days)
    fig.add_trace(go.Candlestick(
        x=last_actual.index,
        open=last_actual['Open'],
        high=last_actual['High'],
        low=last_actual['Low'],
        close=last_actual['Close'],
        name='Actual',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Predicted OHLC (next 7 days)
    fig.add_trace(go.Candlestick(
        x=predicted_data.index,
        open=predicted_data['Open'],
        high=predicted_data['High'],
        low=predicted_data['Low'],
        close=predicted_data['Close'],
        name='Predicted',
        increasing_line_color='lightgreen',
        decreasing_line_color='lightcoral',
        opacity=0.7
    ))
    
    # Add divider line between actual and predicted
    fig.add_vline(
        x=last_actual.index[-1],
        line_dash="dash",
        line_color="gray"
    )
    
    fig.update_layout(
        title=f'{ticker} - Actual vs Predicted OHLC (Next 7 Days)',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def analyze_all_stocks(stock_symbols, period="3mo"):
    analysis_results = []
    
    for ticker in stock_symbols:
        try:
            df = get_stock_data(ticker, period)
            if df.empty:
                continue
                
            df = calculate_technical_indicators(df)
            signals = generate_trading_signals(df)
            
            # Calculate overall score
            score = sum([s[2] for s in signals])
            
            # Get prediction
            prediction = predict_next_week_ohlc(df)
            if prediction:
                pred_score = 2 if "Bullish" in prediction['trend'] else (-2 if "Bearish" in prediction['trend'] else 0)
                score += pred_score
            
            analysis_results.append({
                "Stock": ticker,
                "Current Price": df['Close'].iloc[-1],
                "Score": score,
                "Signals": signals,
                "Prediction": prediction
            })
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
    
    return sorted(analysis_results, key=lambda x: x['Score'], reverse=True)

def main():
    st.set_page_config(page_title="Stock Analysis Pro", layout="wide")
    
    st.title("📊 Advanced Stock Analysis with OHLC Prediction")
    st.write("Technical analysis dashboard with 7-day OHLC price prediction")
    
    # Load stock symbols
    stock_symbols = load_stock_symbols()
    
    # Sidebar controls
    st.sidebar.header("Settings")
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks", 
        stock_symbols,
        default=["RELIANCE.NS", "TCS.NS"]
    )
    
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    selected_period = st.sidebar.selectbox(
        "Select Time Period",
        list(period_options.keys()),
        index=1
    )
    
    # Add button to analyze all stocks
    if st.sidebar.button("Analyze All Stocks (Top Performers)"):
        with st.spinner("Analyzing all stocks..."):
            all_stocks_analysis = analyze_all_stocks(stock_symbols, period_options[selected_period])
            
            st.subheader("Top Stocks Based on Technical Analysis")
            st.write("The following stocks show the strongest bullish signals according to technical indicators:")
            
            top_stocks = all_stocks_analysis[:5]  # Show top 5
            
            for stock in top_stocks:
                with st.expander(f"{stock['Stock']} - Score: {stock['Score']}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Current Price", f"₹{stock['Current Price']:,.2f}")
                        if stock['Prediction']:
                            pred = stock['Prediction']
                            st.metric("Next Week Trend", 
                                     f"{pred['trend']} ({pred['percent_change']:.2f}%)",
                                     delta_color="off")
                    
                    with col2:
                        st.write("**Key Signals:**")
                        for signal in stock['Signals'][:3]:  # Show top 3 signals
                            if signal[2] > 0:
                                st.success(f"{signal[0]}: {signal[1]}")
                            elif signal[2] < 0:
                                st.error(f"{signal[0]}: {signal[1]}")
                            else:
                                st.info(f"{signal[0]}: {signal[1]}")
    
    # Main content
    if not selected_stocks:
        st.warning("Please select at least one stock from the sidebar")
        return
    
    for ticker in selected_stocks:
        st.subheader(f"Analysis for {ticker}")
        
        # Get stock data
        try:
            df = get_stock_data(ticker, period_options[selected_period])
            if df.empty:
                st.error(f"No data available for {ticker}")
                continue
                
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            # Display current price
            current_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
            price_change = current_price - prev_close
            percent_change = (price_change / prev_close) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"₹{current_price:,.2f}")
            col2.metric("Daily Change", f"₹{price_change:,.2f}", f"{percent_change:.2f}%")
            
            # Generate trading signals
            signals = generate_trading_signals(df)
            
            # Display signals
            with st.expander("Trading Signals", expanded=True):
                for signal in signals:
                    if signal[2] > 0:
                        st.success(f"{signal[0]}: {signal[1]}")
                    elif signal[2] < 0:
                        st.error(f"{signal[0]}: {signal[1]}")
                    else:
                        st.info(f"{signal[0]}: {signal[1]}")
            
            # Predict next week trend with OHLC
            prediction = predict_next_week_ohlc(df)
            if prediction:
                with st.expander("Next Week OHLC Prediction", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Predicted Trend", prediction['trend'])
                    col2.metric("Expected Change", f"{prediction['percent_change']:.2f}%")
                    col3.metric("Confidence", prediction['confidence'])
                    
                    # Show predicted OHLC data
                    st.write("Predicted OHLC Data for Next 7 Days:")
                    st.dataframe(prediction['predicted_ohlc'].style.format("{:.2f}"))
                    
                    # Plot OHLC prediction chart
                    plot_ohlc_prediction(df, prediction['predicted_ohlc'], ticker)
            
            # Plot technical charts
            plot_stock_data(df, ticker)
            
            col1, col2 = st.columns(2)
            with col1:
                plot_macd(df)
            with col2:
                plot_rsi(df)
                
            st.divider()
            
        except Exception as e:
            st.error(f"Error processing {ticker}: {str(e)}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides advanced technical analysis with OHLC price prediction. "
        "Predictions are based on Linear Regression and ARIMA models combined with volatility measures. "
        "The 'Analyze All Stocks' button identifies the top performing stocks based on technical signals."
    )

if __name__ == "__main__":
    main()
