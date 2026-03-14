import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

# ---------------------------
# DATA
# ---------------------------
@st.cache_data(ttl=120, show_spinner=False)
def download_data(symbol: str, period="30d", interval="5m") -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.dropna().copy()
        return add_indicators(df)
    except Exception:
        return pd.DataFrame()

# ---------------------------
# INDICATORS
# ---------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_UPPER"] = mid + (2 * std)
    df["BB_LOWER"] = mid - (2 * std)

    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["STOCH_K"] = ((df["Close"] - low14) / (high14 - low14).replace(0, np.nan)) * 100
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()
    df["STOCH_K"] = df["STOCH_K"].fillna(50)
    df["STOCH_D"] = df["STOCH_D"].fillna(50)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean().bfill()

    vwap_den = df["Volume"].replace(0, np.nan).cumsum()
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / vwap_den
    df["RET"] = df["Close"].pct_change().fillna(0)

    return df.dropna()

# ---------------------------
# STRATEGIES
# ---------------------------
def strategy_signal(df: pd.DataFrame, strategy_name: str, p1=None, p2=None):
    row = df.iloc[-1]
    prev = df.iloc[-2]

    close_ = row["Close"]
    signal = "HOLD"
    reason = "No strong edge detected."

    if strategy_name == "EMA_CROSS":
        fast = row.get(f"EMA_{p1}", row["EMA_5"])
        slow = row.get(f"EMA_{p2}", row["EMA_20"])
        prev_fast = prev.get(f"EMA_{p1}", prev["EMA_5"])
        prev_slow = prev.get(f"EMA_{p2}", prev["EMA_20"])
        if prev_fast <= prev_slow and fast > slow:
            signal = "BUY"
            reason = f"Fast EMA ({p1}) crossed above slow EMA ({p2})."
        elif prev_fast >= prev_slow and fast < slow:
            signal = "SELL"
            reason = f"Fast EMA ({p1}) crossed below slow EMA ({p2})."

    elif strategy_name == "RSI_REVERSAL":
        rsi = row["RSI"]
        if rsi < p1:
            signal = "BUY"
            reason = f"RSI is oversold at {rsi:.1f}."
        elif rsi > p2:
            signal = "SELL"
            reason = f"RSI is overbought at {rsi:.1f}."

    elif strategy_name == "MACD_CROSS":
        if prev["MACD"] <= prev["MACD_SIGNAL"] and row["MACD"] > row["MACD_SIGNAL"]:
            signal = "BUY"
            reason = "MACD crossed above signal line."
        elif prev["MACD"] >= prev["MACD_SIGNAL"] and row["MACD"] < row["MACD_SIGNAL"]:
            signal = "SELL"
            reason = "MACD crossed below signal line."

    elif strategy_name == "BB_MEAN_REVERT":
        if close_ < row["BB_LOWER"]:
            signal = "BUY"
            reason = "Price is below the lower Bollinger Band."
        elif close_ > row["BB_UPPER"]:
            signal = "SELL"
            reason = "Price is above the upper Bollinger Band."

    elif strategy_name == "VWAP_RECLAIM":
        if prev["Close"] <= prev["VWAP"] and row["Close"] > row["VWAP"]:
            signal = "BUY"
            reason = "Price reclaimed VWAP."
        elif prev["Close"] >= prev["VWAP"] and row["Close"] < row["VWAP"]:
            signal = "SELL"
            reason = "Price lost VWAP."

    elif strategy_name == "STOCH_TURN":
        if row["STOCH_K"] < p1 and row["STOCH_K"] > row["STOCH_D"]:
            signal = "BUY"
            reason = f"Stochastic turned up from oversold ({row['STOCH_K']:.1f})."
        elif row["STOCH_K"] > p2 and row["STOCH_K"] < row["STOCH_D"]:
            signal = "SELL"
            reason = f"Stochastic turned down from overbought ({row['STOCH_K']:.1f})."

    elif strategy_name == "SMA_TREND":
        if row["SMA_20"] > row["SMA_50"] and close_ > row["SMA_20"]:
            signal = "BUY"
            reason = "Short trend is above long trend."
        elif row["SMA_20"] < row["SMA_50"] and close_ < row["SMA_20"]:
            signal = "SELL"
            reason = "Short trend is below long trend."

    elif strategy_name == "MOMENTUM_3BAR":
        last3 = df["RET"].tail(3)
        if (last3 > 0).all():
            signal = "BUY"
            reason = "Three straight positive bars."
        elif (last3 < 0).all():
            signal = "SELL"
            reason = "Three straight negative bars."

    elif strategy_name == "ATR_BREAKOUT":
        recent_range = row["High"] - row["Low"]
        if recent_range > row["ATR"] * p1 and row["Close"] > prev["Close"]:
            signal = "BUY"
            reason = f"Range expanded above {p1}x ATR with upward follow-through."
        elif recent_range > row["ATR"] * p1 and row["Close"] < prev["Close"]:
            signal = "SELL"
            reason = f"Range expanded above {p1}x ATR with downward follow-through."

    elif strategy_name == "PRICE_EMA20_DISTANCE":
        dist = (row["Close"] - row["EMA_20"]) / row["EMA_20"] * 100
        if dist < -p1:
            signal = "BUY"
            reason = f"Price is stretched {abs(dist):.2f}% below EMA20."
        elif dist > p1:
            signal = "SELL"
            reason = f"Price is stretched {dist:.2f}% above EMA20."

    return signal, reason

def simple_backtest(df: pd.DataFrame, strategy_name: str, p1=None, p2=None, allow_short=True):
    wins = 0
    trades = 0
    pnl = 0.0

    if len(df) < 80:
        return {"win_rate": 0, "avg_return": 0, "trades": 0, "score": 0}

    for i in range(60, len(df) - 3):
        subset = df.iloc[:i+1]
        signal, _ = strategy_signal(subset, strategy_name, p1, p2)
        entry = df["Close"].iloc[i]
        exit_ = df["Close"].iloc[i+3]

        if signal == "BUY":
            ret = (exit_ - entry) / entry
            pnl += ret
            wins += int(ret > 0)
            trades += 1
        elif signal == "SELL" and allow_short:
            ret = (entry - exit_) / entry
            pnl += ret
            wins += int(ret > 0)
            trades += 1

    if trades == 0:
        return {"win_rate": 0, "avg_return": 0, "trades": 0, "score": 0}

    win_rate = wins / trades * 100
    avg_return = pnl / trades * 100
    score = (win_rate * 0.7) + (avg_return * 15) + min(trades, 50) * 0.1

    return {
        "win_rate": win_rate,
        "avg_return": avg_return,
        "trades": trades,
        "score": score
    }

@st.cache_data(show_spinner=False)
def generate_strategy_variants():
    variants = []

    ema_pairs = [(5, 9), (5, 20), (9, 20), (9, 50), (20, 50)]
    rsi_levels = [(25, 75), (30, 70), (35, 65)]
    stoch_levels = [(20, 80), (25, 75), (30, 70)]
    atr_mults = [1.2, 1.5, 1.8, 2.0]
    ema_dist = [0.8, 1.0, 1.5, 2.0]

    for a, b in ema_pairs:
        for _ in range(10):
            variants.append(("EMA_CROSS", a, b))

    for low, high in rsi_levels:
        for _ in range(10):
            variants.append(("RSI_REVERSAL", low, high))

    for _ in range(10):
        variants.append(("MACD_CROSS", None, None))

    for _ in range(10):
        variants.append(("BB_MEAN_REVERT", None, None))

    for _ in range(10):
        variants.append(("VWAP_RECLAIM", None, None))

    for low, high in stoch_levels:
        for _ in range(8):
            variants.append(("STOCH_TURN", low, high))

    for _ in range(8):
        variants.append(("SMA_TREND", None, None))

    for _ in range(8):
        variants.append(("MOMENTUM_3BAR", None, None))

    for x in atr_mults:
        for _ in range(5):
            variants.append(("ATR_BREAKOUT", x, None))

    for x in ema_dist:
        for _ in range(4):
            variants.append(("PRICE_EMA20_DISTANCE", x, None))

    return variants[:100]

def get_risk_level(df: pd.DataFrame):
    atr_pct = (df["ATR"].iloc[-1] / df["Close"].iloc[-1]) * 100
    if atr_pct < 0.8:
        return "Low"
    elif atr_pct < 1.8:
        return "Medium"
    return "High"

def run_all_strategies(df: pd.DataFrame, allow_short=True):
    variants = generate_strategy_variants()
    results = []
    overall_risk = get_risk_level(df)

    for name, p1, p2 in variants:
        bt = simple_backtest(df, name, p1, p2, allow_short=allow_short)
        signal, reason = strategy_signal(df, name, p1, p2)

        if name == "EMA_CROSS":
            display_name = f"EMA Cross {p1}/{p2}"
        elif name == "RSI_REVERSAL":
            display_name = f"RSI Reversal {p1}/{p2}"
        elif name == "STOCH_TURN":
            display_name = f"Stochastic Turn {p1}/{p2}"
        elif name == "ATR_BREAKOUT":
            display_name = f"ATR Breakout x{p1}"
        elif name == "PRICE_EMA20_DISTANCE":
            display_name = f"EMA20 Stretch {p1}%"
        else:
            display_name = name.replace("_", " ").title()

        results.append({
            "name": display_name,
            "base_name": name,
            "signal": signal,
            "reason": reason,
            "win_rate": round(bt["win_rate"], 2),
            "avg_return": round(bt["avg_return"], 3),
            "trades": bt["trades"],
            "score": round(bt["score"], 2),
            "risk_level": overall_risk
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

def get_top_strategies(results, top_n=5):
    top = results[:top_n]
    total_score = sum(max(x["score"], 0.001) for x in top)

    for row in top:
        row["confidence_contribution"] = (max(row["score"], 0.001) / total_score) * 100

    return top

def aggregate_signal(top_results):
    buy_weight = sum(r["confidence_contribution"] for r in top_results if r["signal"] == "BUY")
    sell_weight = sum(r["confidence_contribution"] for r in top_results if r["signal"] == "SELL")
    hold_weight = sum(r["confidence_contribution"] for r in top_results if r["signal"] == "HOLD")

    total = buy_weight + sell_weight + hold_weight
    if total == 0:
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "buy_pct": 0.0,
            "hold_pct": 100.0,
            "sell_pct": 0.0,
            "risk_level": "Medium"
        }

    buy_pct = buy_weight / total * 100
    sell_pct = sell_weight / total * 100
    hold_pct = hold_weight / total * 100

    max_pct = max(buy_pct, sell_pct, hold_pct)
    if buy_pct == max_pct:
        signal = "BUY"
    elif sell_pct == max_pct:
        signal = "SELL"
    else:
        signal = "HOLD"

    directional_gap = abs(buy_pct - sell_pct)
    confidence = min(95, max_pct * 0.75 + directional_gap * 0.25)

    risk_values = [r["risk_level"] for r in top_results]
    if risk_values.count("High") >= 3:
        risk = "High"
    elif risk_values.count("Low") >= 3:
        risk = "Low"
    else:
        risk = "Medium"

    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "buy_pct": round(buy_pct, 2),
        "hold_pct": round(hold_pct, 2),
        "sell_pct": round(sell_pct, 2),
        "risk_level": risk
    }

def scan_watchlist(symbols, interval="5m", allow_short=True):
    rows = []
    period_map = {
        "1m": "7d",
        "2m": "30d",
        "5m": "30d",
        "15m": "60d",
        "30m": "60d",
        "60m": "730d"
    }

    for sym in symbols:
        df = download_data(sym, period=period_map.get(interval, "30d"), interval=interval)
        if df.empty:
            continue
        results = run_all_strategies(df, allow_short=allow_short)
        top = get_top_strategies(results, top_n=5)
        summary = aggregate_signal(top)
        rows.append({
            "Ticker": sym,
            "Price": round(float(df["Close"].iloc[-1]), 2),
            "Signal": summary["signal"],
            "Confidence %": summary["confidence"],
            "BUY %": summary["buy_pct"],
            "HOLD %": summary["hold_pct"],
            "SELL %": summary["sell_pct"],
            "Risk": summary["risk_level"]
        })

    if not rows:
        return pd.DataFrame(columns=["Ticker", "Price", "Signal", "Confidence %", "BUY %", "HOLD %", "SELL %", "Risk"])

    return pd.DataFrame(rows).sort_values(by="Confidence %", ascending=False).reset_index(drop=True)
