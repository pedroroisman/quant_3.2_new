import pandas as pd
import numpy as np
from ta.volatility import average_true_range
from typing import List, Dict, Any, Tuple
import datetime
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv("tiingo.env")

class StrategyResult(BaseModel):
    strategy_name: str
    params: Dict[str, Any]
    plazo: int
    score: float
    sharpe: float
    cagr: float
    max_drawdown: float
    current_price: float
    position: str
    price_target: float
    avg_change: float
    num_changes: int
    last_change_date: str
    last_change_type: str
    avg_position_duration: float | None
    

def calculate_price_target(precio_actual: float, df_strat: pd.DataFrame, strategy: str, plazo: int, atr: float) -> tuple[float, float, int, str, str, float | None]:
    changes_pct = []
    changes_dates = []
    position_changes = []
    position_durations = []
    position = df_strat['position'].iloc[-1]
    position_start_idx = None
    position_start_date = None
    last_change_price = df_strat['adjClose'].loc[last_change_date_index] if last_change_date_index in df_strat.index else precio_actual  # Fallback si no encuentra

    for i in range(1, len(df_strat)):
        if df_strat['position'].iloc[i] != df_strat['position'].iloc[i-1]:
            change_type = "to_long" if df_strat['position'].iloc[i] == 1 else "to_short" if df_strat['position'].iloc[i] == -1 else "to_neutral"
            position_changes.append(change_type)
            changes_dates.append(df_strat.index[i].strftime('%Y-%m-%d'))
            future_idx = min(i + plazo, len(df_strat) - 1)
            change = (df_strat['adjClose'].iloc[future_idx] - df_strat['adjClose'].iloc[i]) / df_strat['adjClose'].iloc[i]
            if df_strat['position'].iloc[i] == 1 and change_type == "to_long":
                changes_pct.append(change)
            elif df_strat['position'].iloc[i] == -1 and change_type == "to_short":
                changes_pct.append(change)
            
            # Calcular duración de la posición anterior si era activa
            if position_start_idx is not None and df_strat['position'].iloc[i-1] != 0:
                duration = (df_strat.index[i] - df_strat.index[position_start_idx]).days
                if duration > 0:
                    position_durations.append(duration)
                    print(f"Strategy: {strategy}, Position: {df_strat['position'].iloc[i-1]}, Duration: {duration}, Start: {df_strat.index[position_start_idx].strftime('%Y-%m-%d')}, End: {df_strat.index[i].strftime('%Y-%m-%d')}")  # Depuración

            # Actualizar el inicio de la nueva posición
            if df_strat['position'].iloc[i] != 0:
                position_start_idx = i
                position_start_date = df_strat.index[i].strftime('%Y-%m-%d')
            else:
                position_start_idx = None
                position_start_date = None

    # Calcular duración de la última posición activa si existe
    if position_start_idx is not None and position != 0:
        duration = (df_strat.index[-1] - df_strat.index[position_start_idx]).days
        if duration > 0:
            position_durations.append(duration)
            print(f"Strategy: {strategy}, Position: {position}, Duration: {duration}, Start: {position_start_date}, End: {df_strat.index[-1].strftime('%Y-%m-%d')}")  # Depuración

    print(f"Strategy: {strategy}, Position Durations: {position_durations}, Mean: {np.mean(position_durations) if position_durations else None}")  # Depuración
    atr_today = average_true_range(df_strat['high'], df_strat['low'], df_strat['close'], window=plazo).iloc[-1]
    volatility_factor = 1.0 if atr < 2 else atr / 2
    if not changes_pct or len(changes_pct) < 3:
        adjusted_change = atr_today / precio_actual if position == 1 else - (atr_today / precio_actual) if position == -1 else 0
    else:
        adjusted_change = max(np.mean(changes_pct) * volatility_factor, 0) if position == 1 else min(np.mean(changes_pct) * volatility_factor, 0) if position == -1 else 0
    price_target = precio_actual * (1 + adjusted_change)
    avg_change = adjusted_change
    num_changes = len(changes_pct)
    last_change_date = changes_dates[-1] if changes_dates else df_strat.index[0].strftime('%Y-%m-%d')
    last_change_type = position_changes[-1] if position_changes else "to_neutral"
    avg_position_duration = np.mean(position_durations) if position_durations else None
    return price_target, avg_change, num_changes, last_change_date, last_change_type, avg_position_duration

def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['returns'] = df['adjClose'].pct_change()
    df['cum_returns'] = (1 + df['returns']).cumprod() - 1
    return df

def run_quant_for_ticker(ticker: str, start_date: str = "2020-01-01", end_date: str = datetime.datetime.now().strftime('%Y-%m-%d')) -> List[Dict]:
    from MVP_ENGINE_V4 import get_data_tiingo, run_buy_and_hold, run_ema_crossover, run_rsi_sma, run_donchian_breakout, run_bollinger_mean_reversion, run_rsi_reversal_atr, run_macd_filter
    df = get_data_tiingo(ticker, start_date, end_date, os.getenv("TIINGO_API_KEY"))
    df = calculate_returns(df)
    strategies = [
        (run_buy_and_hold(df), "Buy & Hold", {}),
        (run_ema_crossover(df, 5, 20), "EMA Crossover", {"fast": 5, "slow": 20}),
        (run_ema_crossover(df, 9, 30), "EMA Crossover", {"fast": 9, "slow": 30}),
        (run_ema_crossover(df, 12, 50), "EMA Crossover", {"fast": 12, "slow": 50}),
        (run_rsi_sma(df, 14, 50), "RSI + SMA", {"rsi": 14, "sma": 50}),
        (run_rsi_sma(df, 14, 100), "RSI + SMA", {"rsi": 14, "sma": 100}),
        (run_rsi_sma(df, 20, 50), "RSI + SMA", {"rsi": 20, "sma": 50}),
        (run_rsi_sma(df, 20, 100), "RSI + SMA", {"rsi": 20, "sma": 100}),
        (run_donchian_breakout(df, 20), "Donchian Breakout", {"lookback": 20}),
        (run_donchian_breakout(df, 30), "Donchian Breakout", {"lookback": 30}),
        (run_donchian_breakout(df, 50), "Donchian Breakout", {"lookback": 50}),
        (run_bollinger_mean_reversion(df, 20, 2), "Bollinger Mean Reversion", {"window": 20, "std": 2}),
        (run_bollinger_mean_reversion(df, 30, 2), "Bollinger Mean Reversion", {"window": 30, "std": 2}),
        (run_bollinger_mean_reversion(df, 50, 2), "Bollinger Mean Reversion", {"window": 50, "std": 2}),
        (run_rsi_reversal_atr(df, 14, 14), "RSI Reversal + ATR", {"rsi": 14, "atr": 14}),
        (run_rsi_reversal_atr(df, 14, 20), "RSI Reversal + ATR", {"rsi": 14, "atr": 20}),
        (run_rsi_reversal_atr(df, 20, 14), "RSI Reversal + ATR", {"rsi": 20, "atr": 14}),
        (run_rsi_reversal_atr(df, 20, 20), "RSI Reversal + ATR", {"rsi": 20, "atr": 20}),
        (run_macd_filter(df, 12, 26, 9), "MACD Filter", {"fast": 12, "slow": 26, "signal": 9}),
        (run_macd_filter(df, 12, 39, 9), "MACD Filter", {"fast": 12, "slow": 39, "signal": 9}),
        (run_macd_filter(df, 19, 26, 9), "MACD Filter", {"fast": 19, "slow": 26, "signal": 9}),
        (run_macd_filter(df, 19, 39, 9), "MACD Filter", {"fast": 19, "slow": 39, "signal": 9}),
    ]
    results = []
    for df_strat, name, params in strategies:
        if df_strat is None or df_strat.empty:
            continue
        ret = df_strat['strategy_returns']
        sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() != 0 else 0
        cagr = (1 + df_strat['cum_returns'].iloc[-1]) ** (252 / len(df_strat)) - 1 if df_strat['cum_returns'].iloc[-1] != 0 else 0
        max_drawdown = (df_strat['cum_returns'].cummax() - df_strat['cum_returns']).max() * -1 if len(df_strat) > 0 else 0
        score = abs(sharpe / max_drawdown * cagr) if max_drawdown != 0 else 0
        price_target, avg_change, num_changes, last_change_date, last_change_type, avg_position_duration = calculate_price_target(df_strat['adjClose'].iloc[-1], df_strat, name, params.get('plazo', 30), params.get('atr', 2))
        position_map = {1: "long", -1: "short", 0: "neutral"}
        position_str = position_map.get(df_strat['position'].iloc[-1], "neutral")
        
        # Nuevo ajuste: Si es short y avg_change == 0, cambiamos a neutral
        if position_str == "short" and avg_change == 0:
            position_str = "neutral"
        
        results.append(
            StrategyResult(
                strategy_name=name,
                params=params,
                plazo=params.get('plazo', 30),
                score=score,
                sharpe=sharpe,
                cagr=cagr,
                max_drawdown=max_drawdown,
                current_price=df_strat['adjClose'].iloc[-1],
                position=position_str,
                price_target=price_target,
                avg_change=avg_change,
                num_changes=num_changes,
                last_change_date=last_change_date,
                last_change_type=last_change_type,
                avg_position_duration=avg_position_duration
            ).dict()
        )
    results.sort(key=lambda x: x['score'], reverse=True)
    return results