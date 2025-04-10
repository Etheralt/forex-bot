
# === Equities Trading Bot ===

import os
import time
import json
import random
import asyncio
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
import MetaTrader5 as mt5
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# removed yfinance
import talib as ta
import gym
from gym_anytrading.envs import TradingEnv

# === MetaTrader5 Initialization ===
MT5_LOGIN = 12345678
MT5_PASSWORD = 'your_password'
MT5_SERVER = 'MetaQuotes-Demo'

mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
    exit()

# === CONFIGURATION ===

# Capital Allocation Across Tickers
TOTAL_CAPITAL = 100000  # Forex paper capital
capital_per_ticker = TOTAL_CAPITAL / len(TICKERS)

TICKERS = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCHF']
STRATEGY_COUNT = 5
MAIN_STRATEGY_IDX = 0
SWITCH_THRESHOLD = 0.15
RISK_PER_TRADE = 0.01  # 1% of capital
CAPITAL = 100000  # Forex paper capital
TIMEFRAME = 'M5'
INTERVAL_MINUTES = 5  # 5-minute interval

# === Position State Tracking ===
positions = {ticker: {"side": None, "entry_price": 0, "trail_stop": 0} for ticker in TICKERS}

WEBHOOK_URL = 'YOUR_DISCORD_WEBHOOK_URL'

NEWS_API_KEY = 'YOUR_NEWS_API_KEY'

import backoff

@backoff.on_exception(backoff.expo, Exception, max_tries=5)

# Logging
def log_to_discord(msg):
    try:
        data = {"content": msg}
        requests.post(WEBHOOK_URL, json=data)
    except Exception as e:
        print(f"Discord log failed: {e}")

# Sentiment
async def fetch_sentiment():
    url = 'https://newsapi.org/v2/top-headlines'
    params = {'apiKey': NEWS_API_KEY, 'country': 'us'}
    try:
        r = await asyncio.to_thread(requests.get, url, params=params)
        headlines = [a['title'] for a in r.json().get('articles', [])]
        sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
        return np.mean(sentiments) if sentiments else 0
    except Exception:
        return 0

# Data
async def fetch_data(ticker):
    df = await asyncio.to_thread(yf.download, ticker, period='60d', interval='5m')
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
    df['SMA'] = ta.SMA(df['Close'], timeperiod=30)
    df['EMA'] = ta.EMA(df['Close'], timeperiod=30)
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df.dropna(inplace=True)
    return df

# Custom Env
class CustomTradingEnv(TradingEnv):
    def __init__(self, df):
        self.df = df
        super().__init__(df=df)

# Models
strategy_models = [None] * STRATEGY_COUNT
strategy_pnls = [0] * STRATEGY_COUNT
strategy_paths = [f"model_strategy_{i}.zip" for i in range(STRATEGY_COUNT)]

def load_models()
first_train = [True] * STRATEGY_COUNT
:
    for i in range(STRATEGY_COUNT):
        try:
            strategy_models[i] = PPO.load(strategy_paths[i])
        except Exception:
            strategy_models[i] = None

def save_model(i):
    path = strategy_paths[i]
    strategy_models[i].save(path)
    log_to_discord(f"Model {i} saved!  [file: {path}]")

# Risk

# Trailing Stop Loss Logic
def calculate_trailing_stop(price, atr, side):
    trail_amount = 1.5 * atr
    if side == 'buy':
        return price - trail_amount
    else:
        return price + trail_amount

def get_risk_bounds(price, atr):
    return price - 1.5 * atr, price + 3 * atr

# Execute Trade
async def execute_trade(action, ticker, price, atr, capital):

    pos = positions[ticker]
    side = 'buy' if action == 0 else 'sell'

    # Only trade if not in position
    if pos["side"] is None:
        try:

    side = 'buy' if action == 0 else 'sell'
    stop_loss = get_risk_bounds(price, atr)[0 if side == 'buy' else 1]
    trailing_stop = calculate_trailing_stop(price, atr, side)
    qty = calculate_position_size(capital, atr)

    log_to_discord(f"Trade Executed: {side.upper()} {ticker} | Price: ${price:.2f} | Trailing SL: ${trailing_stop:.2f} | QTY: {qty}")

            positions[ticker] = {
                "side": side,
                "entry_price": price,
                "trail_stop": price - atr * 1.5 if side == "buy" else price + atr * 1.5
            }
            info = f"{side.upper()} {ticker} at ${price:.2f} | TRAIL: {positions[ticker]['trail_stop']:.2f}"
            log_to_discord(f"Trade Executed: {info} ✅")
        except Exception as e:
            log_to_discord(f"Trade Failed: {e} ❌")

def check_switch():
    global MAIN_STRATEGY_IDX
    best_idx = max(range(STRATEGY_COUNT), key=lambda i: strategy_pnls[i])
    if strategy_pnls[best_idx] > strategy_pnls[MAIN_STRATEGY_IDX] * (1 + SWITCH_THRESHOLD):
        MAIN_STRATEGY_IDX = best_idx
        log_to_discord(f"Switched to strategy {best_idx} due to superior PnL! 🔁")

# Main Loop

from pytz import timezone

def is_market_open():
    now = datetime.now(timezone("US/Eastern"))
    return now.weekday() < 5 and now.hour >= 9 and now.hour < 16

def simulate_data(df):
    # Simple simulation logic (can be replaced with GAN-generated or replayed sequences)
    simulated = df.copy()
    simulated['Close'] = simulated['Close'] * (1 + np.random.normal(0, 0.01, len(simulated)))
    simulated['High'] = simulated['Close'] + np.random.uniform(0.1, 0.3, len(simulated))
    simulated['Low'] = simulated['Close'] - np.random.uniform(0.1, 0.3, len(simulated))
    simulated['Open'] = simulated['Close'] * (1 + np.random.normal(0, 0.005, len(simulated)))
    simulated['Volume'] = simulated['Volume'] * np.random.uniform(0.9, 1.1, len(simulated))
    return simulated

# Overwrite the training loop to account for context-aware training

async def trading_loop():
    load_models()
first_train = [True] * STRATEGY_COUNT

    while True:
        sentiment = await fetch_sentiment()
        for ticker in TICKERS:
            df = await fetch_data(ticker)
            if df is None or df.empty:
                continue
            env = DummyVecEnv([lambda df=df: CustomTradingEnv(df)])
            obs = env.reset()

            for i in range(STRATEGY_COUNT):
                if strategy_models[i] is None:
                    strategy_models[i] = PPO("MlpPolicy", env, verbose=0)

if first_train[i]:
    strategy_models[i].learn(total_timesteps=50000)
    first_train[i] = False
elif is_market_open():
    strategy_models[i].learn(total_timesteps=10000)
else:
    sim_df = simulate_data(df)
    sim_env = DummyVecEnv([lambda df=sim_df: CustomTradingEnv(df)])
    strategy_models[i].set_env(sim_env)
    strategy_models[i].learn(total_timesteps=20000)

                    save_model(i)
                else:

if first_train[i]:
    strategy_models[i].learn(total_timesteps=50000)
    first_train[i] = False
elif is_market_open():
    strategy_models[i].learn(total_timesteps=10000)
else:
    sim_df = simulate_data(df)
    sim_env = DummyVecEnv([lambda df=sim_df: CustomTradingEnv(df)])
    strategy_models[i].set_env(sim_env)
    strategy_models[i].learn(total_timesteps=20000)

                action, _ = strategy_models[i].predict(obs)
                price_series = df['Close'].values
                pnl = (price_series[-1] - price_series[-2]) / price_series[-2] * 100
                strategy_pnls[i] += pnl

            main_action, _ = strategy_models[MAIN_STRATEGY_IDX].predict(obs)
            bias = 0 if sentiment > 0 else 1
            if main_action == bias:
                price = df['Close'].iloc[-1]
                atr = df['ATR'].iloc[-1]
                await execute_trade(main_action, ticker, price, atr)

        check_switch()
        time.sleep(60 * INTERVAL_MINUTES)

# Entry
if __name__ == '__main__':
    asyncio.run(trading_loop())

def calculate_reward(ticker, current_price):
    pos = positions[ticker]
    if pos["side"] is None:
        return 0
    delta = current_price - pos["entry_price"]
    reward = delta if pos["side"] == "buy" else -delta
    return reward

def update_trailing_stop(ticker, current_price, atr):
    pos = positions[ticker]
    if pos["side"] == "buy":
        new_trail = max(pos["trail_stop"], current_price - atr * 1.5)
    elif pos["side"] == "sell":
        new_trail = min(pos["trail_stop"], current_price + atr * 1.5)
    else:
        return
    positions[ticker]["trail_stop"] = new_trail

def check_exit(ticker, current_price):
    pos = positions[ticker]
    if pos["side"] == "buy" and current_price < pos["trail_stop"]:
        return True
    if pos["side"] == "sell" and current_price > pos["trail_stop"]:
        return True
    return False

# --- Multi-Timeframe Analysis ---
def get_multi_timeframe_data(symbol, timeframes):
    data = {}
    for tf in timeframes:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, 500)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        data[tf] = df
    return data

# --- News Sentiment Analysis ---
def fetch_forex_news_sentiment():
    # Placeholder for news API integration
    # Parse sentiment from top headlines about forex market
    sentiment_score = 0.1  # Dummy positive signal
    return sentiment_score

# --- Backtesting Functionality ---
def backtest_strategy(symbol, time_frame, strategy_func, start, end):
    rates = mt5.copy_rates_range(symbol, time_frame, start, end)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    returns = []
    for i in range(len(df)):
        signal = strategy_func(df.iloc[:i+1])
        if signal == "buy":
            returns.append(df['close'].iloc[i+1] - df['close'].iloc[i])
        elif signal == "sell":
            returns.append(df['close'].iloc[i] - df['close'].iloc[i+1])
    return sum(returns)

def upload_model_to_discord(model_path, webhook_url):
    import requests
    with open(model_path, 'rb') as f:
        file_data = {'file': (model_path, f)}
        response = requests.post(webhook_url, files=file_data)
        if response.status_code == 204:
            print("Model uploaded to Discord successfully.")
        else:
            print(f"Failed to upload model to Discord: {response.status_code} - {response.text}")

# Example usage after saving model
def save_model(i):
    model_path = f"model_checkpoint_{i}.pt"
    torch.save(agent.model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    send_discord_message(f"Model checkpoint saved at iteration {i}.")
    upload_model_to_discord(model_path, DISCORD_WEBHOOK_URL)
