# Quantum Market Terminal

Streamlit dashboard for NIFTY 50 and BANKNIFTY market data, India VIX, selectable daily/15-minute/1-hour candles, a configurable Yahoo Finance ticker scanner, global macro snapshot, RSS headline search, and optional evidence-grounded AI analysis.

## Data and limitations

Index, VIX, watchlist, and global market quotes are fetched through Yahoo Finance and may be delayed or unavailable; a direct licensed NSE feed is not configured. The page reruns every 60 seconds and market history/options chains use a 60-second cache; this is not an exchange-level tick feed. Technical indicators (EMA, RSI, ATR, relative volume, and 20-bar range checks) describe historical underlying prices. The directional evidence tally counts rule conditions; it is not a calibrated probability and does not predict or guarantee profitable results. The options risk calculator uses manually supplied premium, stop premium, capital, risk percentage, and lot size; it excludes fees, taxes, slippage, stop gaps, implied-volatility changes, and broker margin, and is not a contract or entry recommendation.

The global panel covers selected Yahoo Finance India/US/Europe/Asia indices, FX pairs, commodity futures, US yield indices, and crypto pairs; it is not every exchange or a direct NSE feed. Yahoo option chains are fetched only for tickers/expiries that Yahoo exposes (the default example is SPY); NIFTY/BANKNIFTY Yahoo option expiries are unavailable here, so those panels show reference strikes only. Chain PCR and peak open-interest strikes describe only the selected returned Yahoo chain; they are not trading signals. A direct authorized NSE options-chain feed is not configured. Headlines are fetched on demand from Google News RSS for a selected/custom query; this does not cover every article, official filing, government scheme, or geopolitical event. Headlines are not full-text verified.

The optional analyst uses only the displayed snapshot and fetched headline metadata. To enable it, set `OPENAI_API_KEY` in the server environment before starting Streamlit. Optionally set `OPENAI_MODEL` to a model enabled for your API account. Never commit keys or paste them into the app/chat. Without a key, news links and the local market snapshot still work. Model output may be incomplete or wrong; it is not investment advice and does not promise returns.

## Run locally

1. Install the requirements:

   ```
   pip install -r requirements.txt
   ```

2. Start the app:

   ```
   streamlit run streamlit_app.py
   ```

To enable the optional analyst in a shell before starting the app:

```
export OPENAI_API_KEY="your-key"
streamlit run streamlit_app.py
```
