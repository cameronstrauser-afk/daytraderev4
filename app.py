import streamlit as st
import pandas as pd
from strategy_engine import (
    download_data,
    run_all_strategies,
    get_top_strategies,
    aggregate_signal,
    scan_watchlist,
)
from paper_trading import init_paper_trading, render_paper_trading
from alerts import init_alerts, render_alerts

st.set_page_config(
    page_title="Day Trading Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- STYLES ----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0e1117;
    color: #e6edf3;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
.hero-card, .section-card, .signal-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 34px;
    font-weight: 800;
}
.subtle {
    color: #9aa4b2;
    font-size: 14px;
}
.buy { color: #2ecc71; font-weight: 800; }
.sell { color: #ff5c5c; font-weight: 800; }
.hold { color: #f1c40f; font-weight: 800; }
.kpi {
    background: #11161d;
    border: 1px solid #222a33;
    border-radius: 14px;
    padding: 14px;
    text-align: center;
}
.kpi-label {
    color: #9aa4b2;
    font-size: 12px;
}
.kpi-value {
    font-size: 24px;
    font-weight: 800;
}
.small-note {
    font-size: 12px;
    color: #9aa4b2;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
init_paper_trading()
init_alerts()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "MSFT", "AMD", "META"]

if "latest_run" not in st.session_state:
    st.session_state.latest_run = None

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">Day Trading Stock Predictor</div>
    <div class="subtle">
        Read this in order: 1) Final Signal 2) Confidence 3) Vote Breakdown 4) Strategy Reasons
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Controls")
    symbol = st.text_input("Ticker", value="AAPL").upper().strip()

    interval = st.selectbox("Interval", ["1m", "2m", "5m", "15m", "30m", "60m"], index=2)

    period_map = {
        "1m": "7d",
        "2m": "30d",
        "5m": "30d",
        "15m": "60d",
        "30m": "60d",
        "60m": "730d"
    }
    period = period_map[interval]

    use_top_n = st.slider("Top strategies used in final vote", 3, 10, 5)
    allow_short = st.toggle("Enable short selling", value=True)
    starting_cash = st.number_input("Paper trading starting cash", min_value=100.0, value=10000.0, step=100.0)

    st.markdown("---")
    run_main = st.button("Run Predictor", use_container_width=True)
    run_watchlist = st.button("Run Watchlist Scanner", use_container_width=True)

    st.markdown("---")
    st.subheader("Watchlist")
    watchlist_text = st.text_area(
        "Comma-separated tickers",
        value=", ".join(st.session_state.watchlist),
        height=90
    )
    if st.button("Update Watchlist", use_container_width=True):
        st.session_state.watchlist = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]
        st.success("Watchlist updated.")

# ---------------- RUN MODEL ONLY WHEN ASKED ----------------
def run_predictor():
    df = download_data(symbol, period=period, interval=interval)
    if df.empty:
        return None

    results = run_all_strategies(df, allow_short=allow_short)
    top_results = get_top_strategies(results, top_n=use_top_n)
    summary = aggregate_signal(top_results)

    return {
        "df": df,
        "all_results": results,
        "top_results": top_results,
        "summary": summary,
        "symbol": symbol,
        "interval": interval,
    }

if run_main or st.session_state.latest_run is None:
    st.session_state.latest_run = run_predictor()

if st.session_state.latest_run is None:
    st.error("No data returned for that ticker/interval.")
    st.stop()

data = st.session_state.latest_run
df = data["df"]
results = data["all_results"]
top_results = data["top_results"]
summary = data["summary"]
latest_price = float(df["Close"].iloc[-1])

# ---------------- MAIN TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Strategy Details", "Watchlist Scanner", "Paper Trading", "Alerts"]
)

# ================= OVERVIEW =================
with tab1:
    st.markdown("## Overview")

    left, right = st.columns([2, 1])

    with left:
        st.markdown("### Price Chart")
        tv_symbol = f"NASDAQ:{symbol}" if symbol else "NASDAQ:AAPL"
        tv_interval = interval.replace("m", "") if "m" in interval else interval

        tradingview_widget = f"""
        <iframe
            src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_1&symbol={tv_symbol}&interval={tv_interval}&hidesidetoolbar=0&symboledit=1&saveimage=1&toolbarbg=0e1117&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1&hideideas=1&studies=[]"
            width="100%"
            height="520"
            frameborder="0"
            allowtransparency="true"
            scrolling="no">
        </iframe>
        """
        st.components.v1.html(tradingview_widget, height=540)

    with right:
        signal_class = summary["signal"].lower()
        st.markdown(f"""
        <div class="signal-card">
            <div class="kpi-label">Final Signal</div>
            <div class="{signal_class}" style="font-size:32px;">{summary["signal"]}</div>
            <br>
            <div class="kpi-label">Confidence</div>
            <div class="kpi-value">{summary["confidence"]:.1f}%</div>
            <br>
            <div class="kpi-label">Risk Level</div>
            <div class="kpi-value">{summary["risk_level"]}</div>
            <br>
            <div class="kpi-label">Current Price</div>
            <div class="kpi-value">${latest_price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.info(
            "How to read this:\n"
            "- Final Signal = model’s main call\n"
            "- Confidence = how strong that call is\n"
            "- Risk = volatility level"
        )

    st.markdown("### Vote Breakdown")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi"><div class="kpi-label">BUY %</div><div class="kpi-value">{summary["buy_pct"]:.1f}%</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi"><div class="kpi-label">HOLD %</div><div class="kpi-value">{summary["hold_pct"]:.1f}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi"><div class="kpi-label">SELL %</div><div class="kpi-value">{summary["sell_pct"]:.1f}%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Top Strategies Used</div><div class="kpi-value">{len(top_results)}</div></div>', unsafe_allow_html=True)

    st.markdown("### Quick Explanation")
    st.markdown(f"""
    <div class="section-card">
        <b>What to look at first:</b><br>
        1. <b>Final Signal:</b> {summary["signal"]}<br>
        2. <b>Confidence:</b> {summary["confidence"]:.1f}%<br>
        3. <b>Vote split:</b> BUY {summary["buy_pct"]:.1f}% / HOLD {summary["hold_pct"]:.1f}% / SELL {summary["sell_pct"]:.1f}%<br>
        4. <b>Risk:</b> {summary["risk_level"]}
    </div>
    """, unsafe_allow_html=True)

# ================= STRATEGY DETAILS =================
with tab2:
    st.markdown("## Strategy Details")

    st.markdown("### Top Strategy Reasons")
    for i, strat in enumerate(top_results, start=1):
        sig_class = strat["signal"].lower()
        st.markdown(f"""
        <div class="section-card">
            <div style="font-size:18px;font-weight:800;">#{i} {strat["name"]}</div>
            <div class="small-note">
                Signal: <span class="{sig_class}">{strat["signal"]}</span> |
                Win Rate: {strat["win_rate"]:.2f}% |
                Score: {strat["score"]:.2f} |
                Risk: {strat["risk_level"]} |
                Confidence Contribution: {strat["confidence_contribution"]:.2f}%
            </div>
            <br>
            <div><b>Reason:</b> {strat["reason"]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Full Top Strategy Table")
    top_df = pd.DataFrame(top_results)[[
        "name", "signal", "win_rate", "score", "risk_level",
        "confidence_contribution", "reason"
    ]]
    st.dataframe(top_df, use_container_width=True)

# ================= WATCHLIST =================
with tab3:
    st.markdown("## Watchlist Scanner")
    st.caption("This is slower because it downloads data for multiple tickers. It only runs when you press the scanner button.")

    if run_watchlist:
        with st.spinner("Scanning watchlist..."):
            scanner_df = scan_watchlist(st.session_state.watchlist, interval=interval, allow_short=allow_short)
        st.dataframe(scanner_df, use_container_width=True)
    else:
        st.info("Press 'Run Watchlist Scanner' in the sidebar when you want to scan multiple stocks.")

# ================= PAPER TRADING =================
with tab4:
    st.markdown("## Paper Trading")
    render_paper_trading(
        symbol=symbol,
        latest_price=latest_price,
        capital_default=starting_cash,
        summary=summary
    )

# ================= ALERTS =================
with tab5:
    st.markdown("## Alerts")
    render_alerts(symbol=symbol, latest_price=latest_price, summary=summary)

st.markdown("---")
st.caption("Educational use only. Not financial advice.")
