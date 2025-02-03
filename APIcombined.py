import streamlit as st
import plotly.graph_objects as go
import requests
import pandas as pd
import plotly.express as px

# Polygon API Key (Replace with your own API key)
API_KEY = "7F2V73lH7AM6upHipIOtRi8WoA4FaJ7G"

# Function to fetch stock data from Polygon API
def fetch_stock_data(ticker, start_date="2023-01-01", end_date="2023-12-31"):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            df = pd.DataFrame(data["results"])
            df["t"] = pd.to_datetime(df["t"], unit="ms")  # Convert timestamp
            return df
    return None

# Streamlit App
st.set_page_config(page_title="Stock Market MetaData Visualization", layout="wide")
st.title("📈 Stock Market MetaData Visualization")

# Sidebar options
st.sidebar.header("Stock Selection")
st.sidebar.markdown("Select stocks to compare:")
stock_options = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA", "NFLX"]
tickers = st.sidebar.multiselect("Select Stocks", stock_options, default=["AAPL", "MSFT"])

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

if tickers:
    st.sidebar.success(f"Selected Stocks: {', '.join(tickers)}")
else:
    st.sidebar.warning("Please select at least one stock.")

st.sidebar.header("Chart Customization")
chart_type = st.sidebar.selectbox("Choose Chart Type",
                                  ["Candlestick", "Line", "OHLC", "Bar", "Moving Average", "Heatmap", "Scatter", "Area"])

# Define a color palette for better visualization
color_palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]
stock_colors = {ticker: color_palette[i % len(color_palette)] for i, ticker in enumerate(stock_options)}

# Visualization
st.subheader("Stock Market Trends")
fig = go.Figure()
stock_data = {}

for ticker in tickers:
    df = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if df is not None:
        stock_data[ticker] = df  # Store data for later use
        color = stock_colors[ticker]

        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(x=df["t"], open=df["o"], high=df["h"], low=df["l"], close=df["c"],
                                         name=ticker,
                                         increasing=dict(line=dict(color=color)),
                                         decreasing=dict(line=dict(color=color))))
        elif chart_type == "Line":
            fig.add_trace(go.Scatter(x=df["t"], y=df["c"], mode='lines', name=ticker, line=dict(color=color)))
        elif chart_type == "OHLC":
            fig.add_trace(go.Ohlc(x=df["t"], open=df["o"], high=df["h"], low=df["l"], close=df["c"], name=ticker,
                                  increasing=dict(line=dict(color=color)),
                                  decreasing=dict(line=dict(color=color))))
        elif chart_type == "Bar":
            fig.add_trace(go.Bar(x=df["t"], y=df["c"], name=ticker, marker=dict(color=color)))
        elif chart_type == "Moving Average":
            df['MA'] = df['c'].rolling(window=10, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=df["t"], y=df["MA"], mode='lines', name=f"{ticker} MA", line=dict(color=color)))
        elif chart_type == "Scatter":
            fig.add_trace(go.Scatter(x=df["t"], y=df["c"], mode='markers', name=ticker, marker=dict(color=color)))
        elif chart_type == "Area":
            fig.add_trace(go.Scatter(x=df["t"], y=df["c"], fill='tozeroy', mode='lines', name=ticker, line=dict(color=color)))
    else:
        st.warning(f"No data available for {ticker}.")

# Special handling for heatmap
if chart_type == "Heatmap" and stock_data:
    combined_df = pd.concat(stock_data.values(), ignore_index=True)
    fig = px.density_heatmap(combined_df, x='t', y='c', title="Stock Price Heatmap")

fig.update_layout(title="Stock Price Movement", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# Volume Bar Chart
st.subheader("Trading Volume")
fig_volume = go.Figure()
for ticker, df in stock_data.items():
    fig_volume.add_trace(go.Bar(x=df["t"], y=df["v"], name=ticker, marker=dict(color=stock_colors[ticker])))
fig_volume.update_layout(title="Trading Volume", xaxis_title="Date", yaxis_title="Volume", template="plotly_dark")
st.plotly_chart(fig_volume, use_container_width=True)

st.sidebar.header("Additional Features")
if st.sidebar.button("Show Data Table"):
    if stock_data:
        df_combined = pd.concat(stock_data.values(), ignore_index=True)
        st.dataframe(df_combined)
    else:
        st.warning("Select at least one stock to display data.")
