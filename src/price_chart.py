"""Ticker price chart: candlesticks, volume, moving averages, RSI and signal markers.

Pure Plotly/pandas (no BigQuery or Gemini), so it can be tested with sample data.
StockAnalyzer.price_movement_plot() loads the data and calls build_price_figure().
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG = "#232B32"
GRID = "#33404A"
TEXT = "#E6EDF3"
UP = "#2ecc71"
DOWN = "#e74c3c"
MA7_COLOR = "#f5b942"
MA60_COLOR = "#5dade2"
RSI_COLOR = "#b388ff"

MA60_WARMUP = 59   # the SQL averages partial windows at the start of the history; hide those rows
RSI_WARMUP = 14

# Only these signals get a marker; trend labels (UPTREND, PULLBACK...) would cover every day.
BULLISH_SIGNALS = ("CAPITULATION BUY", "TECH BREAKOUT", "HIGH CONVICTION BUY")
BEARISH_SIGNALS = ("EXHAUSTED SELL", "BREAKDOWN")  # also matches FLASH BREAKDOWN

# (button label, months back). The 6M window is the default view.
RANGES = (("1M", 1), ("3M", 3), ("6M", 6), ("1Y", 12))
DEFAULT_RANGE = "6M"


def _prepare(prices: pd.DataFrame, signals: pd.DataFrame | None) -> pd.DataFrame:
    df = prices.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("Date").reset_index(drop=True)

    # SQL values keep the chart identical to the signal table; fall back to pandas if SQL gave nothing
    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_60"] = df["Close"].rolling(60).mean()
    df["RSI_14"] = np.nan
    df["Signal"] = ""
    if signals is not None and not signals.empty:
        sig = signals.copy()
        sig["Date"] = pd.to_datetime(sig["Date"]).dt.tz_localize(None).dt.normalize()
        sig = sig.drop_duplicates("Date").set_index("Date")
        for col in ("MA_7", "MA_60", "RSI_14"):
            if col in sig:
                df[col] = pd.to_numeric(df["Date"].map(sig[col]), errors="coerce").fillna(df[col])
        if "Signal" in sig:
            df["Signal"] = df["Date"].map(sig["Signal"]).fillna("")

    pos = np.arange(len(df))
    df.loc[pos < MA60_WARMUP, "MA_60"] = np.nan
    df.loc[pos < RSI_WARMUP, "RSI_14"] = np.nan
    df.loc[pos < MA60_WARMUP, "Signal"] = ""
    return df


def _signal_markers(df: pd.DataFrame, names: tuple) -> pd.DataFrame:
    """First day of each run of the same signal, for the given signal names."""
    hit = df["Signal"].apply(lambda s: any(n in s for n in names))
    first = df["Signal"] != df["Signal"].shift()
    return df[hit & first]


def _window(df: pd.DataFrame, months: int | None) -> pd.DataFrame:
    if months is None:
        return df
    return df[df["Date"] >= df["Date"].max() - pd.DateOffset(months=months)]


def _ranges(win: pd.DataFrame) -> dict:
    """Axis ranges for one window. Set explicitly so the y axes fit the window after a button click."""
    lo, hi = float(win["Low"].min()), float(win["High"].max())
    for c in ("MA_7", "MA_60"):
        if win[c].notna().any():
            lo, hi = min(lo, win[c].min()), max(hi, win[c].max())
    pad = (hi - lo) * 0.06 or hi * 0.02
    vmax = float(np.nanmax(win["Volume"].values)) if win["Volume"].notna().any() else 1.0
    return {
        "x": [win["Date"].min() - pd.Timedelta(days=1), win["Date"].max() + pd.Timedelta(days=2)],
        "price": [lo - pad, hi + pad],
        "volume": [0, vmax * 1.15],
    }


def build_price_figure(prices: pd.DataFrame, signals: pd.DataFrame | None = None):
    """Returns (figure, metrics) or (None, None) when there is no data.

    prices:  Date, Open, High, Low, Close, Volume (one row per trading day, oldest history available)
    signals: optional, one row per day with Date, Signal, MA_7, MA_60, RSI_14 (StockAnalyzer SQL output)
    metrics: perf_pct and current_price over the last six months, as the dashboard heading expects
    """
    if prices is None or prices.empty:
        return None, None
    df = _prepare(prices, signals)
    if df.empty:
        return None, None

    six = _window(df, 6)
    start_val, end_val = six["Close"].iloc[0], six["Close"].iloc[-1]
    metrics = {"perf_pct": (end_val - start_val) / start_val * 100, "current_price": end_val}

    has_rsi = df["RSI_14"].notna().any()
    rows = 3 if has_rsi else 2
    heights = [0.62, 0.16, 0.22] if has_rsi else [0.80, 0.20]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)

    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price",
        increasing=dict(line=dict(color=UP), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_7"], name="7D MA", mode="lines",
                             line=dict(color=MA7_COLOR, width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_60"], name="60D MA", mode="lines",
                             line=dict(color=MA60_COLOR, width=1.6)), row=1, col=1)

    for names, symbol, color, label, anchor in (
            (BULLISH_SIGNALS, "triangle-up", UP, "Bullish signal", "Low"),
            (BEARISH_SIGNALS, "triangle-down", DOWN, "Bearish signal", "High"),
    ):
        m = _signal_markers(df, names)
        if m.empty:
            continue
        offset = 0.985 if anchor == "Low" else 1.015
        fig.add_trace(go.Scatter(
            x=m["Date"], y=m[anchor] * offset, mode="markers", name=label,
            marker=dict(symbol=symbol, size=11, color=color, line=dict(color=TEXT, width=1)),
            text=m["Signal"], hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    bar_colors = np.where(df["Close"] >= df["Open"], UP, DOWN)
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color=bar_colors,
                         marker_opacity=0.55, showlegend=False), row=2, col=1)

    if has_rsi:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI_14"], name="RSI 14", mode="lines",
                                 line=dict(color=RSI_COLOR, width=1.4), showlegend=False), row=3, col=1)
        for level in (30, 70):
            fig.add_hline(y=level, line=dict(color="#8b98a5", width=1, dash="dot"), row=3, col=1)
        fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], title_text="RSI", row=3, col=1)

    # Range buttons: only offer windows the data covers, plus "All". Plotly's own range selector does not
    # refit the y axes to the chosen window, so each button sets the x and y ranges itself.
    span_days = (df["Date"].max() - df["Date"].min()).days
    options = [(label, months) for label, months in RANGES if span_days >= months * 28]
    options.append(("All", None))
    labels = [label for label, _ in options]
    default = DEFAULT_RANGE if DEFAULT_RANGE in labels else "All"

    buttons, initial = [], None
    for label, months in options:
        r = _ranges(_window(df, months))
        args = {"xaxis.range": r["x"], "yaxis.range": r["price"], "yaxis2.range": r["volume"]}
        buttons.append(dict(label=label, method="relayout", args=[args]))
        if label == default:
            initial = r

    fig.update_xaxes(range=initial["x"], rangebreaks=[dict(bounds=["sat", "mon"])],
                     rangeslider_visible=False, gridcolor=GRID, linecolor=GRID, showspikes=True,
                     spikecolor="#8b98a5", spikethickness=1, spikedash="dot")
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(range=initial["price"], title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(range=initial["volume"], title_text="Vol", showticklabels=False, row=2, col=1)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT),
        hovermode="x unified", xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.08, yanchor="bottom"),
        # Plotly paints the selected button near-white whatever bgcolor is, so use dark text on light buttons
        updatemenus=[dict(type="buttons", direction="right", active=labels.index(default), buttons=buttons,
                          x=0, xanchor="left", y=1.08, yanchor="bottom", pad=dict(r=4), showactive=True,
                          bgcolor="#9fb3c4", bordercolor=GRID, font=dict(color="#10171c"))],
    )
    return fig, metrics
