import pandas as pd
import numpy as np
import requests
import io
import os
import ta
from scipy.stats import norm

def get_data_tiingo(ticker: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """
    Fetches historical price data for a given ticker from Tiingo.
    """
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {'startDate': start_date, 'endDate': end_date, 'format': 'csv', 'token': api_key}
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        print(f"Error fetching {ticker}: {resp.status_code}")
        return None
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=['date']).set_index('date')
    return df



def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['returns'] = df['adjClose'].pct_change()
    return df.dropna(subset=['returns'])


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / 252
    return np.sqrt(252) * excess.mean() / excess.std()


def calculate_max_drawdown(cum_returns: pd.Series) -> float:
    roll_max = cum_returns.cummax()
    drawdown = cum_returns / roll_max - 1.0
    return drawdown.min()


def calculate_cagr(cum_returns: pd.Series, periods_per_year: int = 252) -> float:
    years = len(cum_returns) / periods_per_year
    return cum_returns.iloc[-1] ** (1 / years) - 1

def run_ema_crossover(df: pd.DataFrame, ema_fast: int, ema_slow: int) -> pd.DataFrame:
    df = df.copy()
    df['ema_fast'] = df['adjClose'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['adjClose'].ewm(span=ema_slow, adjust=False).mean()
    df['signal'] = np.where(df['ema_fast'] > df['ema_slow'], 1, np.where(df['ema_fast'] < df['ema_slow'], -1, 0))
    df['position'] = df['signal'].shift(1).fillna(0)
    df['strategy_returns'] = df['returns'] * df['position']
    df['cum_returns'] = (1 + df['strategy_returns']).cumprod()
    return df

def run_rsi_sma(df: pd.DataFrame, rsi_period: int, sma_period: int) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['adjClose'], window=rsi_period)
    df['sma'] = df['adjClose'].rolling(window=sma_period).mean()
    df['signal'] = np.where((df['rsi'] > 50) & (df['adjClose'] > df['sma']), 1, np.where((df['rsi'] < 50) & (df['adjClose'] < df['sma']), -1, 0))
    df['position'] = df['signal'].shift(1).fillna(0)
    df['strategy_returns'] = df['returns'] * df['position']
    df['cum_returns'] = (1 + df['strategy_returns']).cumprod()
    return df

def run_donchian_breakout(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df = df.copy()
    df['donchian_high'] = df['adjClose'].rolling(window=lookback).max()
    df['donchian_low'] = df['adjClose'].rolling(window=lookback).min()
    df['signal'] = np.where(df['adjClose'] > df['donchian_high'].shift(1), 1, np.where(df['adjClose'] < df['donchian_low'].shift(1), -1, np.nan))
    df['signal'] = df['signal'].ffill()
    df['position'] = df['signal'].shift(1).fillna(0)
    df['strategy_returns'] = df['returns'] * df['position']
    df['cum_returns'] = (1 + df['strategy_returns']).cumprod()
    return df

def run_bollinger_mean_reversion(df: pd.DataFrame, window: int, n_std: float) -> pd.DataFrame:
    df = df.copy()
    df['rolling_mean'] = df['adjClose'].rolling(window=window).mean()
    df['rolling_std'] = df['adjClose'].rolling(window=window).std()
    df['upper_band'] = df['rolling_mean'] + n_std * df['rolling_std']
    df['lower_band'] = df['rolling_mean'] - n_std * df['rolling_std']
    df['signal'] = np.where(df['adjClose'] < df['lower_band'], 1, np.where(df['adjClose'] > df['upper_band'], -1, np.nan))
    df['signal'] = df['signal'].ffill()
    df['position'] = df['signal'].shift(1).fillna(0)
    df['strategy_returns'] = df['returns'] * df['position']
    df['cum_returns'] = (1 + df['strategy_returns']).cumprod()
    return df

def run_rsi_reversal_atr(df: pd.DataFrame, rsi_period: int, atr_period: int, rsi_threshold: int = 30) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = ta.momentum.rsi(df['adjClose'], window=rsi_period)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['adjClose'], window=atr_period)
    df['signal'] = np.where(df['rsi'] < rsi_threshold, 1, np.where(df['rsi'] > (100 - rsi_threshold), -1, np.nan))
    df['signal'] = df['signal'].ffill()
    df['position'] = df['signal'].shift(1).fillna(0)
    df['strategy_returns'] = df['returns'] * df['position']
    df['cum_returns'] = (1 + df['strategy_returns']).cumprod()
    return df

def run_macd_filter(df: pd.DataFrame, ema_fast: int, ema_slow: int, signal_period: int) -> pd.DataFrame:
    df = df.copy()
    fast_ema = df['adjClose'].ewm(span=ema_fast, adjust=False).mean()
    slow_ema = df['adjClose'].ewm(span=ema_slow, adjust=False).mean()
    df['macd_line'] = fast_ema - slow_ema
    df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
    df['signal'] = np.where(df['macd_line'] > df['macd_signal'], 1, np.where(df['macd_line'] < df['macd_signal'], -1, 0))
    df['position'] = df['signal'].shift(1).fillna(0)
    df['strategy_returns'] = df['returns'] * df['position']
    df['cum_returns'] = (1 + df['strategy_returns']).cumprod()
    return df

def run_buy_and_hold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estrategia Buy & Hold: posición siempre larga desde el inicio hasta el final.
    """
    df = df.copy()
    df['position'] = 1
    df['strategy_returns'] = df['returns'] * df['position']
    df['cum_returns'] = (1 + df['strategy_returns']).cumprod()
    return df

def main():
    api_key = os.getenv('TIINGO_API_KEY')
    tickers = ['AAPL', 'MSFT', 'TSLA', 'SPY']
    start_date = "2015-01-01"
    end_date = "2025-06-01"
    results = []

    for ticker in tickers:
        df_raw = get_data_tiingo(ticker, start_date, end_date, api_key)
        if df_raw is None:
            continue
        df = calculate_returns(df_raw)

            # --- Buy & Hold ---
        strat = run_buy_and_hold(df)
        sharpe = calculate_sharpe_ratio(strat['strategy_returns'].dropna())
        max_dd = calculate_max_drawdown(strat['cum_returns'].dropna()) * 100
        cagr = calculate_cagr(strat['cum_returns'].dropna()) * 100
        score = sharpe / abs(max_dd) * cagr

        results.append({
            'ticker': ticker,
            'strategy_name': 'Buy & Hold',
            'param1': None,
            'param2': None,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'cagr_pct': cagr,
            'score_compuesto': score
        })


        # EMA crossover
        for fast, slow in [(5,20), (9,30), (12,50)]:
            strat = run_ema_crossover(df, fast, slow)
            sharpe = calculate_sharpe_ratio(strat['strategy_returns'].dropna())
            max_dd = calculate_max_drawdown(strat['cum_returns'].dropna()) * 100
            cagr = calculate_cagr(strat['cum_returns'].dropna()) * 100
            score = sharpe / abs(max_dd) * cagr
            results.append({
                'ticker': ticker,
                'strategy_name': 'EMA Crossover',
                'param1': fast,
                'param2': slow,
                'sharpe_ratio': sharpe,
                'max_drawdown_pct': max_dd,
                'cagr_pct': cagr,
                'score_compuesto': score
            })

        # RSI + SMA
        for rsi_p in [14, 20]:
            for sma_p in [50, 100]:
                strat = run_rsi_sma(df, rsi_p, sma_p)
                sharpe = calculate_sharpe_ratio(strat['strategy_returns'].dropna())
                max_dd = calculate_max_drawdown(strat['cum_returns'].dropna()) * 100
                cagr = calculate_cagr(strat['cum_returns'].dropna()) * 100
                score = sharpe / abs(max_dd) * cagr
                results.append({
                    'ticker': ticker,
                    'strategy_name': 'RSI + SMA',
                    'param1': rsi_p,
                    'param2': sma_p,
                    'sharpe_ratio': sharpe,
                    'max_drawdown_pct': max_dd,
                    'cagr_pct': cagr,
                    'score_compuesto': score
                })

        # Donchian Breakout
        for lb in [20, 30, 50]:
            strat = run_donchian_breakout(df, lb)
            sharpe = calculate_sharpe_ratio(strat['strategy_returns'].dropna())
            max_dd = calculate_max_drawdown(strat['cum_returns'].dropna()) * 100
            cagr = calculate_cagr(strat['cum_returns'].dropna()) * 100
            score = sharpe / abs(max_dd) * cagr
            results.append({
                'ticker': ticker,
                'strategy_name': 'Donchian Breakout',
                'param1': lb,
                'param2': None,
                'sharpe_ratio': sharpe,
                'max_drawdown_pct': max_dd,
                'cagr_pct': cagr,
                'score_compuesto': score
            })

        # Bollinger Mean Reversion
        for window in [20, 30, 50]:
            for std in [2]:
                strat = run_bollinger_mean_reversion(df, window, std)
                sharpe = calculate_sharpe_ratio(strat['strategy_returns'].dropna())
                max_dd = calculate_max_drawdown(strat['cum_returns'].dropna()) * 100
                cagr = calculate_cagr(strat['cum_returns'].dropna()) * 100
                score = sharpe / abs(max_dd) * cagr
                results.append({
                    'ticker': ticker,
                    'strategy_name': 'Bollinger Mean Reversion',
                    'param1': window,
                    'param2': std,
                    'sharpe_ratio': sharpe,
                    'max_drawdown_pct': max_dd,
                    'cagr_pct': cagr,
                    'score_compuesto': score
                })

        # RSI Reversal + ATR
        for rsi_p in [14, 20]:
            for atr_p in [14, 20]:
                strat = run_rsi_reversal_atr(df, rsi_p, atr_p)
                sharpe = calculate_sharpe_ratio(strat['strategy_returns'].dropna())
                max_dd = calculate_max_drawdown(strat['cum_returns'].dropna()) * 100
                cagr = calculate_cagr(strat['cum_returns'].dropna()) * 100
                score = sharpe / abs(max_dd) * cagr
                results.append({
                    'ticker': ticker,
                    'strategy_name': 'RSI Reversal + ATR',
                    'param1': rsi_p,
                    'param2': atr_p,
                    'sharpe_ratio': sharpe,
                    'max_drawdown_pct': max_dd,
                    'cagr_pct': cagr,
                    'score_compuesto': score
                })

        # MACD Filter
        for fast in [12, 19]:
            for slow in [26, 39]:
                signal = 9
                strat = run_macd_filter(df, fast, slow, signal)
                sharpe = calculate_sharpe_ratio(strat['strategy_returns'].dropna())
                max_dd = calculate_max_drawdown(strat['cum_returns'].dropna()) * 100
                cagr = calculate_cagr(strat['cum_returns'].dropna()) * 100
                score = sharpe / abs(max_dd) * cagr
                results.append({
                    'ticker': ticker,
                    'strategy_name': 'MACD Filter',
                    'param1': fast,
                    'param2': slow,
                    'sharpe_ratio': sharpe,
                    'max_drawdown_pct': max_dd,
                    'cagr_pct': cagr,
                    'score_compuesto': score
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv("quant_results_v3.csv", index=False)
    print("Results saved to quant_results_v3.csv")

if __name__ == "__main__":
    main()
