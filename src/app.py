import os
import pandas as pd
from google.cloud import bigquery
import streamlit as st

st.set_page_config(
    page_title="NitroStox Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from upgraded_nitrostox import SECDataUnavailable, StockAnalyzer, run_gemini_agent
except Exception as e:
    st.error(f"Backend Startup Failure: {e}")
    st.warning("Please check your database connectivity parameters.")
    st.stop()


# ------------------------------------------------------------------------------
# STREAMLIT CACHING DECORATORS (LLM & DATA FETCH)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_ai(ticker, _analyzer, insider_text, div_yield, payout_ratio):
    """Macro sentiment + 5-pillar deep dive, run in parallel and cached together for 1 hour per ticker."""
    return _analyzer.run_ai_analysis_parallel(insider_text, div_yield, payout_ratio)


@st.fragment
def rag_search_panel(analyzer):
    """Own rerun scope: searching filings doesn't re-execute the rest of the dashboard."""
    rag_query = st.text_input(
        "Search 10-Q filing chunks:",
        placeholder="e.g., supply chain costs, debt refinancing, capex...",
        key="rag_search_box",
    )
    if rag_query:
        with st.spinner("Running NumPy in-memory cosine similarity search..."):
            search_results = analyzer.search_sec_filings(rag_query, top_k=3)
        if not search_results.empty:
            for _, row in search_results.iterrows():
                st.info(f"**Similarity Score:** `{row['similarity_score']}`\n\n{row['chunk_text']}")
        else:
            st.warning("No semantic matches found in the vector database for this query.")


@st.fragment
def agent_panel(active_ticker):
    """Chat history survives reruns; the agent also sees the last few turns."""
    history = st.session_state.setdefault("agent_history", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("e.g. What are NVDA's cost risks? If high, add to watchlist.")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🤖 Agent planning & executing tool calls..."):
                reply = run_gemini_agent(prompt, active_ticker=active_ticker, history=history)
            st.markdown(reply)
        history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ])
        # The agent may have changed the watchlist: refresh the sidebar
        st.session_state.watchlist_refresh += 1
        st.rerun()


# BIGQUERY DATABASE CONFIGURATION & OPTIMIZED CLIENT
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "cloud-etl-500119")
DATASET_ID = os.getenv("BQ_DATASET", "nitrostox_db")


@st.cache_resource(show_spinner=False)
def get_bq_client():
    """Cache the BigQuery client to prevent reconnecting on every UI rerun."""
    try:
        return bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        st.sidebar.error(f"BigQuery Client Initialization Error: {e}")
        return None


bq_client = get_bq_client()

# Session State Initialization
if "analyzer_data" not in st.session_state:
    st.session_state.analyzer_data = None
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = None
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "watchlist_refresh" not in st.session_state:
    st.session_state.watchlist_refresh = 0
if "db_saved" not in st.session_state:
    st.session_state.db_saved = False

# Custom CSS Styling
st.markdown(
    """
<style>
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    font-size: 1.1rem !important;  
    font-weight: 700 !important;   
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

.stCheckbox {
    padding: 0.5rem 0;
}

.stTextInput > div > div > input {
    border-radius: 8px;
    border: 2px solid #667eea;
    transition: all 0.2s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #764ba2;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
}

div[data-testid="metric-container"], .stMetric {
    background: rgba(102, 126, 234, 0.1);
    padding: 0.75rem 0.5rem !important;
    border-radius: 8px;
    border-left: 3px solid #667eea;
}

div[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    padding-bottom: 4px;
}

.stTabs [data-baseweb="tab"] {
    padding: 6px 14px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    border-radius: 6px;
    white-space: nowrap !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 NitroStox Financial Intelligence")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_cached_watchlist(refresh_key):
    if not bq_client:
        return []
    try:
        query = f"SELECT symbol FROM `{PROJECT_ID}.{DATASET_ID}.watchlist` ORDER BY updated_at DESC;"
        return [row["symbol"] for row in bq_client.query(query).result()]
    except Exception as e:
        print(f"Watchlist fetch failed: {e}")
        return []


def add_to_watchlist(ticker):
    if not bq_client:
        return False
    try:
        ticker = ticker.upper().strip()
        merge_query = f"""
            MERGE `{PROJECT_ID}.{DATASET_ID}.watchlist` T
            USING (SELECT @ticker AS symbol) S
            ON T.symbol = S.symbol
            WHEN NOT MATCHED THEN
                INSERT (symbol, updated_at) VALUES (S.symbol, CURRENT_TIMESTAMP())
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
        )
        bq_client.query(merge_query, job_config=job_config).result()

        st.session_state.watchlist_refresh += 1
        return True
    except Exception as e:
        st.sidebar.error(f"Add to watchlist error: {e}")
        return False


def remove_from_watchlist(ticker):
    if not bq_client:
        return False
    try:
        ticker = ticker.upper().strip()
        delete_query = f"DELETE FROM `{PROJECT_ID}.{DATASET_ID}.watchlist` WHERE symbol = @ticker;"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
        )
        bq_client.query(delete_query, job_config=job_config).result()

        st.session_state.watchlist_refresh += 1
        return True
    except Exception as e:
        st.sidebar.error(f"Remove from watchlist error: {e}")
        return False


def trigger_analysis(ticker_to_run):
    with st.spinner(
            f"⚡ Fetching market data for {ticker_to_run.upper()}..."
    ):
        temp_analyzer = StockAnalyzer(ticker_to_run)
        success, _, _ = temp_analyzer.fetch_pipeline_data_parallel()

        if success:
            # We removed temp_analyzer.build_sec_knowledge_base() here.
            # Vectors will now lazy-load only when the user requests AI features.

            st.session_state.analyzer_data = temp_analyzer
            st.session_state.current_ticker = ticker_to_run.upper()
            st.session_state.db_saved = False
            return True
        else:
            st.session_state.analyzer_data = None
            st.error(
                "Execution failed. Unable to map or verify asset ticker:"
                f" {ticker_to_run.upper()}"
            )
            return False

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    ticker_input = st.text_input(
        "🎯 Ticker Symbol", value="AAPL", placeholder="e.g., AAPL, TSLA, MSFT"
    )

    use_ai = st.toggle(
        "🧠 Enable AI Intelligence",
        value=True,
        help="Powered by Gemini Flash • Parallel RAG Search",
    )

    enable_agent = st.toggle(
        "🤖 Enable Autonomous Agent Mode",
        value=False,
        help="Unlocks interactive ReAct tool calling with Gemini 3.7 Flash",
    )

    if enable_agent:
        st.caption("⚡ AI Inference Engine: Gemini 3.7 Flash (ReAct Agent)")
    elif use_ai:
        st.caption("⚡ AI Inference Engine: Gemini 3.7 Flash")

    st.markdown("---")

    run_button = st.button(
        "Execute Analysis",
        type="primary",
        use_container_width=True,
        help="Run comprehensive market analysis",
    )

    if bq_client:
        st.markdown("---")
        st.subheader("📊 Your Watchlist")

        st.session_state.watchlist = fetch_cached_watchlist(st.session_state.watchlist_refresh)

        if ticker_input:
            ticker_upper = ticker_input.upper().strip()
            is_in_watchlist = ticker_upper in st.session_state.watchlist

            if is_in_watchlist:
                if st.button("❌ Remove from Watchlist", use_container_width=True):
                    if remove_from_watchlist(ticker_upper):
                        st.success(f"Removed {ticker_upper}!")
                        st.rerun()
            else:
                if st.button("⭐ Add to Watchlist", use_container_width=True):
                    if add_to_watchlist(ticker_upper):
                        st.success(f"Added {ticker_upper}!")
                        st.rerun()

        if st.session_state.watchlist:
            st.caption(f"**{len(st.session_state.watchlist)} tickers tracked:**")

            cols_per_row = 3
            for i in range(0, len(st.session_state.watchlist), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(st.session_state.watchlist):
                        ticker = st.session_state.watchlist[idx]
                        if col.button(
                                ticker,
                                key=f"watchlist_{ticker}",
                                use_container_width=True,
                        ):
                            if trigger_analysis(ticker):
                                st.rerun()
        else:
            st.caption("ℹ️ No tickers in watchlist. Add one above!")

if run_button and ticker_input:
    if trigger_analysis(ticker_input):
        st.rerun()

# MAIN DISPLAY CONTROLLER

if st.session_state.analyzer_data:
    analyzer = st.session_state.analyzer_data
    active_ticker = st.session_state.current_ticker

    st.markdown(f"## {analyzer.company_name} (`{active_ticker}`)")
    st.markdown(
        f"**Sector:** {analyzer.sector} &nbsp;|&nbsp; **Industry:** {analyzer.industry}"
    )
    st.divider()

    # Computed once in trigger_analysis(); reruns just read it
    fig, metrics = analyzer.view["plot"]
    signals_df = analyzer.view["signals"]
    div_rate, y_pct, p_pct, div_history_df = analyzer.view["dividends"]
    insider_text = analyzer.view["insider"]

    col_viz, col_ai = st.columns([4, 3])

    with col_viz:
        if fig and metrics:
            pct = metrics["perf_pct"]
            sign = "+" if pct >= 0 else ""
            direction = "Up" if pct >= 0 else "Down"
            st.subheader(
                f"📊 {active_ticker} 6-Month Trend ({direction} {sign}{pct:.1f}%)"
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader(f"📊 6-Month {active_ticker} Price Action")
            st.info("Trend visualization window calculations are currently unavailable.")

    with col_ai:
        st.subheader("🧠 Financial Intelligence")
        ai_container = st.container()

    st.divider()

    col_signals, col_fundamentals = st.columns([5, 3])

    with col_signals:
        st.subheader("📈 Technical Trading Ledger")
        if not signals_df.empty:
            st.dataframe(
                signals_df,
                use_container_width=True,
                hide_index=True,
                height=350,
                column_order=[
                    "Date",
                    "Close_Price",
                    "Signal",
                    "Volume_Deviation",
                    "MA_7",
                    "MA_60",
                    "RSI_14",
                ],
                column_config={
                    "Date": st.column_config.DateColumn(
                        "Date", format="MM-DD", width="small"
                    ),
                    "Close_Price": st.column_config.NumberColumn(
                        "Price", format="$%.2f", width="small"
                    ),
                    "Signal": st.column_config.TextColumn(
                        "Market Signal", width="medium"
                    ),
                    "Volume_Deviation": st.column_config.TextColumn(
                        "Vol Delta",
                        width="small",
                        help=(
                            "Volume variance relative to 20-day average in millions"
                        ),
                    ),
                    "MA_7": st.column_config.NumberColumn(
                        "7D MA", format="$%.2f", width="small"
                    ),
                    "MA_60": st.column_config.NumberColumn(
                        "60D MA", format="$%.2f", width="small"
                    ),
                    "RSI_14": st.column_config.NumberColumn(
                        "RSI", format="%.1f", width="small"
                    ),
                },
            )

            with st.popover("ℹ️ Signal Guide"):
                st.markdown("""
                  **Signal Definitions:**
                  * 🟢 **Uptrend:** 7D MA > 60D MA & price holding support
                  * 📉 **Pullback:** Dip below 7D MA holding 60D macro support
                  * ⚠️ **Flash Breakdown:** Sharp dip below 60D MA while 7D MA holds
                  * ⚠️ **Breakdown:** Full structural breach (Price & 7D MA < 60D)
                  * ⚠️ **Weak Breakout:** Price > prior 20D high without volume surge
                  * 🟡 **Relief Rally:** Bounce above 7D MA during macro downtrend
                  * 🔴 **Downtrend:** Macro baseline down & price pinned below fast MA
                  * 🚀 **Breakout:** High-volume expansion at local highs
                  * ⚡ **Capitulation:** Oversold Z-Score + volume spike
                  """)
        else:
            st.info("No mathematical trading patterns detected.")

    with col_fundamentals:
        st.subheader("💰 Dividends & Insider Transactions")
        tab_div, tab_insider = st.tabs(
            ["💵 Dividend Profile", "👔 SEC Corporate Insiders"]
        )

        with tab_div:
            m1, m2, m3 = st.columns(3)
            m1.metric("Rate", f"${div_rate:.2f}")
            m2.metric("Yield", f"{y_pct:.2f}%")
            m3.metric("Payout Ratio", f"{p_pct:.2f}%")

            if p_pct > 100:
                st.error(
                    "High Risk: Dividend payout structural distribution exceeds 100%.",
                    icon="⚠️",
                )

            if not div_history_df.empty:
                col_map = {col.lower(): col for col in div_history_df.columns}
                required_cols = []
                for target in ["Date", "Dividends", "Daily_Percentage_Difference"]:
                    if target.lower() in col_map:
                        required_cols.append(col_map[target.lower()])

                st.dataframe(
                    div_history_df[required_cols],
                    hide_index=True,
                    use_container_width=True,
                    height=180,
                    column_config={
                        "Date": st.column_config.DateColumn(
                            "📅 Ex-Dividend Date", format="MM-DD-YYYY"
                        ),
                        "Dividends": st.column_config.NumberColumn("Payout", format="$%.2f"),
                        "Daily_Percentage_Difference": st.column_config.NumberColumn(
                            "📉 Stock Price Chg",
                            format="%.2f%%",
                            help="The daily percentage change of the stock price on the ex-dividend date.",
                        ),
                    },
                )

        with tab_insider:
            st.caption("Recent SEC Form 4 Filings Activity Summary:")
            st.code(insider_text, language="text")

    with ai_container:
        if not use_ai and not enable_agent:
            st.warning(
                "AI processing skipped. Enable via the sidebar configuration panel."
            )

        if use_ai:
            tab_macro, tab_micro = st.tabs([
                "🌍 Macro Sentiment (6-Month)",
                "📋 Earnings Deep-Dive (RAG 5-Pillar)",
            ])

            ai_error = None
            with tab_macro:
                with st.spinner("Running macro sentiment and RAG earnings audit in parallel..."):
                    try:
                        # An exception inside a st.cache_data function is not cached,
                        # so a failed SEC build is retried on the next run.
                        ai_sentiment_result, earnings_deep_dive = get_cached_ai(
                            active_ticker, analyzer, insider_text, y_pct, p_pct
                        )
                    except SECDataUnavailable as e:
                        ai_error = str(e)
                        earnings_deep_dive = ""
                        # Sentiment does not need the filing; still show it.
                        ai_sentiment_result = analyzer.analyze_sentiment(insider_text, y_pct, p_pct)
                if ai_sentiment_result:
                    st.info(ai_sentiment_result)

            with tab_micro:
                if ai_error:
                    st.error(f"Earnings deep-dive unavailable. {ai_error}")
                    st.caption("Fix the cause (see the terminal), then press Execute Analysis to retry.")
                elif earnings_deep_dive:
                    st.markdown(earnings_deep_dive)

            if ai_error:
                pass  # do not save an incomplete analysis to BigQuery
            elif not st.session_state.get("db_saved", False):
                with st.spinner("💾 Committing analysis payload to BigQuery..."):
                    try:
                        analyzer.export_to_bigquery(
                            df=signals_df,
                            sentiment=ai_sentiment_result if ai_sentiment_result else "",
                            earnings=earnings_deep_dive if earnings_deep_dive else "",
                        )
                        st.session_state.db_saved = True
                        st.toast(
                            "✅ Analysis successfully saved to BigQuery!", icon="💾"
                        )
                    except Exception as e:
                        st.error(f"🔴 Database Commit Error: {e}")

            st.divider()
            st.subheader("🔍 Interactive SEC 10-Q Semantic Search")
            st.caption("Query the in-memory vector knowledge base directly:")

            rag_search_panel(analyzer)

        if enable_agent:
            if use_ai:
                st.divider()

            st.subheader("🤖 Autonomous Gemini 3.7 Flash Agent")
            st.caption(
                "Ask questions, query SEC 10-Qs via in-memory search, or update your"
                " watchlist in real-time:"
            )

            agent_panel(active_ticker)
else:
    st.markdown("---")

    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        if bq_client:
            st.caption("🟢 **Google BigQuery Engine:** Online")
        else:
            st.caption("🔴 **Google BigQuery Engine:** Offline")
    with status_col2:
        st.caption("⚡ **AI Inference Engine:** Gemini 3.7 Flash")
    with status_col3:
        st.caption("🏛️ **SEC EDGAR Pipeline:** Active (NumPy RAM Search)")

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("⚡ Quick Launch Analysis")
    st.write("Choose a key asset chip below or enter a ticker in the sidebar to begin:")

    chip_col1, chip_col2, chip_col3, chip_col4, chip_col5 = st.columns(5)

    if chip_col1.button("📌 AAPL", use_container_width=True):
        if trigger_analysis("AAPL"):
            st.rerun()

    if chip_col2.button("🚀 NVDA", use_container_width=True):
        if trigger_analysis("NVDA"):
            st.rerun()

    if chip_col3.button("⚡ PLTR", use_container_width=True):
        if trigger_analysis("PLTR"):
            st.rerun()

    if chip_col4.button("🏦 NU", use_container_width=True):
        if trigger_analysis("NU"):
            st.rerun()

    if chip_col5.button("⚛️ CEG", use_container_width=True):
        if trigger_analysis("CEG"):
            st.rerun()

    st.markdown("---")

    st.subheader("🛠️ Platform Capabilities")

    feat_col1, feat_col2, feat_col3 = st.columns(3)

    with feat_col1:
        st.info(
            "**In-Database SQL Analytics**\n\n"
            "Calculates 7D/60D MA crossovers, RSI 14, and volume momentum "
            "directly using BigQuery Standard SQL window functions."
        )

    with feat_col2:
        st.success(
            "**In-Memory SEC Semantic RAG**\n\n"
            "Embeds SEC 10-Q filings via Gemini to audit margin pressures "
            "and operating risks using NumPy dot product math."
        )

    with feat_col3:
        st.warning(
            "**Real-Time Insider Intelligence**\n\n"
            "Streams SEC Form 4 filings via EDGAR to track executive "
            "buying and selling signals."
        )