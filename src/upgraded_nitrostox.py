import os
import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from edgar import Company, set_identity
from google import genai
from google.cloud import bigquery
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from sec_rag import SECVectorRAG
import zoneinfo

load_dotenv()

sec_id = os.getenv("SEC_IDENTITY")

if sec_id:
    os.environ["SEC_IDENTITY"] = sec_id
    try:
        set_identity(sec_id)
    except Exception as e:
        print(f"⚠️ Warning initializing SEC identity: {e}")
else:
    print("⚠️ Warning: SEC_IDENTITY environment variable not found in current environment.")

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "cloud-etl-500119")
DATASET_ID = os.getenv("BQ_DATASET", "nitrostox_db")

bq_client = bigquery.Client(project=PROJECT_ID)

# Shared across sessions and agent tool calls: a filing is downloaded and embedded once per TTL.
_KB_CACHE: dict = {}
_KB_CACHE_TTL_SECONDS = 24 * 3600
# Last analyzer loaded per ticker, so agent tools see the same sector/data as the dashboard.
_ANALYZERS: dict = {}


def _fetch_filing_text_direct(cik, identity: str, forms=("10-Q", "10-K", "20-F", "6-K")) -> str:
    """Fallback when edgar's Filing.text() fails: download the latest filing straight from SEC.

    Uses the public submissions API (data.sec.gov) to find the newest filing of the first form
    type that exists, then downloads its primary document and strips the HTML to plain text.
    The SEC requires a User-Agent that identifies you (SEC_IDENTITY: name and email).
    """
    import re
    import requests
    from bs4 import BeautifulSoup

    headers = {"User-Agent": identity, "Accept-Encoding": "gzip, deflate"}
    cik_int = int(cik)
    resp = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik_int:010d}.json", headers=headers, timeout=30
    )
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    for form in forms:
        for i, f in enumerate(recent["form"]):
            primary = recent["primaryDocument"][i]
            if f != form or not primary:
                continue
            acc = recent["accessionNumber"][i].replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{primary}"
            doc = requests.get(url, headers=headers, timeout=60)
            doc.raise_for_status()

            soup = BeautifulSoup(doc.text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            for tag in soup.find_all("ix:header"):  # hidden XBRL metadata block
                tag.decompose()
            text = soup.get_text("\n")
            text = re.sub(r"[ \t\xa0]+", " ", text)
            text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
            print(f"📄 Direct SEC download: {form} filed {recent['filingDate'][i]} "
                  f"({len(text):,} characters) from {url}")
            return text
    return ""


class SECDataUnavailable(RuntimeError):
    """Raised when no SEC filing text could be retrieved, so the AI audit must not run on an empty context."""


class StockAnalyzer:

    def __init__(self, ticker):
        self.ticker = ticker.upper().strip()
        self.table_name = f"`{PROJECT_ID}.{DATASET_ID}.core_financials`"
        self.company_name = "Loading..."
        self.sector = "Loading..."
        self.industry = "Loading..."
        self.ticker_info = {}
        # Local RAM states for vector search
        self.sec_chunks = []
        self.sec_embeddings = []
        self.sec_error = ""  # last reason the SEC knowledge base failed to build (shown in the UI)
        self._kb_lock = threading.Lock()
        self._news = None
        self._news_lock = threading.Lock()
        # Filled by fetch_pipeline_data_parallel(); the UI reads from here instead of recomputing
        self.view = {}

    def _get_news(self) -> list:
        """Yahoo headlines, fetched once per analyzer (sentiment and the deep-dive both need them)."""
        with self._news_lock:
            if self._news is None:
                try:
                    self._news = yf.Ticker(self.ticker).news or []
                except Exception as net_err:
                    print(f"Yahoo News network timeout: {net_err}")
                    self._news = []
            return self._news

    def _last_expected_trading_date(self, now=None) -> datetime.date:
        """Most recent trading day whose bar should be final (weekends handled, holidays not)."""
        et_tz = zoneinfo.ZoneInfo("America/New_York")
        now = now or datetime.datetime.now(et_tz)
        d = now.date()
        if now.weekday() < 5 and now.time() >= datetime.time(16, 0):
            return d
        d -= datetime.timedelta(days=1)
        while d.weekday() >= 5:
            d -= datetime.timedelta(days=1)
        return d

    def is_market_open(self) -> bool:
        """Checks if US markets are currently in active trading hours (9:30 AM - 4:00 PM ET, Mon-Fri)."""
        try:
            et_tz = zoneinfo.ZoneInfo("America/New_York")
            now_et = datetime.datetime.now(et_tz)

            # Weekends
            if now_et.weekday() >= 5:
                return False

            market_open = datetime.time(9, 30)
            market_close = datetime.time(16, 0)
            return market_open <= now_et.time() <= market_close
        except Exception as e:
            print(f"Timezone evaluation warning: {e}")
            return True  # Fallback to fetching fresh data if timezone check fails

    def download_data(self, ttl_minutes: int = 15) -> bool:
        """Downloads market data and upserts into core_financials only when data is stale."""
        try:
            check_sql = f"""
                SELECT 
                    MAX(f.Date) as max_date,
                    MAX(r.pipeline_execution_time) as last_execution
                FROM {self.table_name} f
                LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.nitrostox_analysis_results` r 
                    ON f.ticker = r.ticker
                WHERE f.ticker = @ticker
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", self.ticker)]
            )
            check_df = bq_client.query(check_sql, job_config=job_config).to_dataframe()

            if not check_df.empty:
                max_date = check_df["max_date"].iloc[0]
                last_exec = check_df["last_execution"].iloc[0]
                today = datetime.datetime.now(datetime.timezone.utc).date()
                max_date_obj = pd.to_datetime(max_date).date() if pd.notna(max_date) else None

                # Correct method invocation using self
                if self.is_market_open():
                    if pd.notna(last_exec):
                        last_exec_dt = pd.to_datetime(last_exec)
                        now_utc = datetime.datetime.now(datetime.timezone.utc)

                        if (now_utc - last_exec_dt).total_seconds() < (ttl_minutes * 60):
                            print(
                                f"⚡ Intraday cache hit for {self.ticker} (fetched < {ttl_minutes}m ago). Skipping download.")
                            return True
                else:
                    if max_date_obj and max_date_obj >= self._last_expected_trading_date():
                        print(
                            f"⚡ Market closed & data for {self.ticker} is current ({max_date_obj}). Skipping download.")
                        return True

        except Exception as e:
            print(f"Staleness check warning: {e}")

        try:
            data = yf.Ticker(self.ticker)
            df = data.history(period="6mo", interval="1d")

            if df.empty:
                return False

            df = df.reset_index()
            df["ticker"] = self.ticker

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.date

            df = df.rename(columns={"Stock Splits": "Stock_Splits"})

            delete_sql = f"DELETE FROM {self.table_name} WHERE ticker = @ticker"
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", self.ticker)]
            )
            bq_client.query(delete_sql, job_config=job_config).result()

            table_id = f"{PROJECT_ID}.{DATASET_ID}.core_financials"
            load_job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
            job = bq_client.load_table_from_dataframe(df, table_id, job_config=load_job_config)
            job.result()

            print(f"✅ Upserted {len(df)} fresh rows for {self.ticker} into core_financials.")
            return True
        except Exception as e:
            print(f"Failed to download and store data for {self.ticker}: {e}")
            return False

    def analyze_moving_averages(self) -> pd.DataFrame:
        try:
            if self.sector in ["Utilities", "Consumer Defensive"]:
                signal_logic = """
                    WHEN Z_Score < -2.0 AND Volume > (1.5 * Avg_Vol_20_day) AND Price_Change > 0 THEN '⚡ CAPITULATION BUY'
                    WHEN Z_Score > 2.0 AND RSI_14 > 70 THEN '⚠️ EXHAUSTED SELL'
                """
            elif self.sector in ["Technology", "Consumer Cyclical"]:
                signal_logic = """
                    WHEN MA_7_day > MA_60_day AND Close >= Local_High_20_day AND Volume > (1.5 * Avg_Vol_20_day) AND RSI_14 < 70 THEN '🚀 TECH BREAKOUT'
                """
            else:
                signal_logic = """
                    WHEN MA_7_day > MA_60_day AND Close >= Local_High_20_day AND Volume > (1.5 * Avg_Vol_20_day) AND RSI_14 < 70 THEN '🚀 HIGH CONVICTION BUY'
                """

            query_str = f"""
                WITH BaseData AS (
                    SELECT 
                        Date, 
                        Close, 
                        Volume,
                        Close - LAG(Close) OVER(ORDER BY Date ASC) AS Price_Change,
                        (Close - LAG(Close) OVER(ORDER BY Date ASC)) / LAG(Close) OVER(ORDER BY Date ASC) * 100 AS Daily_Return_Pct,
                        MAX(Close) OVER(ORDER BY Date ASC ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS Local_High_20_day,
                        AVG(Volume) OVER(ORDER BY Date ASC ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS Avg_Vol_20_day,
                        AVG(Close) OVER(ORDER BY Date ASC ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS MA_60_day,
                        ROW_NUMBER() OVER(ORDER BY Date ASC) AS rn
                    FROM {self.table_name}
                    WHERE ticker = @ticker
                ),
                GainsLosses AS (
                    SELECT *,
                        CASE WHEN Price_Change > 0 THEN Price_Change ELSE 0 END AS Gain,
                        CASE WHEN Price_Change < 0 THEN ABS(Price_Change) ELSE 0 END AS Loss,
                        (Close - MA_60_day) * (Close - MA_60_day) AS Squared_Dev
                    FROM BaseData
                ),
                WilderDecay AS (
                    SELECT *,
                        Gain * POWER(13.0 / 14.0, -rn) AS Weighted_Gain,
                        Loss * POWER(13.0 / 14.0, -rn) AS Weighted_Loss,
                        POWER(13.0 / 14.0, rn) AS Scale_Factor
                    FROM GainsLosses
                ),
                Averages AS (
                    SELECT *,
                        SUM(Weighted_Gain) OVER(ORDER BY Date ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) * Scale_Factor / 14.0 AS Avg_Gain_14,
                        SUM(Weighted_Loss) OVER(ORDER BY Date ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) * Scale_Factor / 14.0 AS Avg_Loss_14,
                        AVG(Close) OVER(ORDER BY Date ASC ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS MA_7_day,
                        AVG(Squared_Dev) OVER(ORDER BY Date ASC ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS Variance_60
                    FROM WilderDecay
                ),
                RSI_Calc AS (
                    SELECT *,
                        CASE 
                            WHEN Avg_Loss_14 = 0 THEN 100.0 
                            WHEN Avg_Gain_14 = 0 THEN 0.0
                            ELSE 100.0 - (100.0 / (1.0 + (Avg_Gain_14 / Avg_Loss_14))) 
                        END AS RSI_14,
                        CASE 
                            WHEN Variance_60 <= 0 THEN 0 
                            ELSE (Close - MA_60_day) / SQRT(Variance_60) 
                        END AS Z_Score
                    FROM Averages
                )
                SELECT
                    CAST(Date AS DATE) AS Date,
                    ROUND(CAST(Close AS NUMERIC), 2) AS Close_Price,
                    CASE
                        {signal_logic}
                        WHEN MA_7_day > MA_60_day AND Close > Local_High_20_day THEN '⚠️ WEAK BREAKOUT (Check Vol/RSI)'
                        WHEN MA_7_day > MA_60_day AND Close < MA_60_day THEN '⚠️ FLASH BREAKDOWN'
                        WHEN MA_7_day < MA_60_day AND Close < MA_60_day THEN '⚠️ BREAKDOWN'
                        WHEN MA_7_day > MA_60_day AND Close < MA_7_day AND Close >= MA_60_day THEN '📉 PULLBACK'
                        WHEN MA_7_day > MA_60_day AND Close >= MA_7_day THEN '🟢 UPTREND'
                        WHEN MA_7_day < MA_60_day AND Close > MA_7_day THEN '🟡 RELIEF RALLY'
                        WHEN MA_7_day < MA_60_day AND Close <= MA_7_day THEN '🔴 DOWNTREND'
                        ELSE '⚪ NEUTRAL'
                    END AS Signal,
                    CONCAT(CAST(ROUND(CAST((Volume - Avg_Vol_20_day) / 1000000.0 AS NUMERIC), 1) AS STRING), 'M') AS Volume_Deviation,
                    ROUND(CAST(MA_7_day AS NUMERIC), 2) AS MA_7,
                    ROUND(CAST(MA_60_day AS NUMERIC), 2) AS MA_60,
                    ROUND(CAST(RSI_14 AS NUMERIC), 2) AS RSI_14,
                    Volume,
                    ROUND(CAST(Avg_Vol_20_day AS NUMERIC), 0) AS Avg_Vol_20
                FROM RSI_Calc
                ORDER BY Date DESC
                LIMIT 15;
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "ticker", "STRING", self.ticker
                    )
                ]
            )
            df = bq_client.query(query_str, job_config=job_config).to_dataframe()

            df["Volume"] = df["Volume"].apply(
                lambda x: f"{int(x):,}" if pd.notnull(x) else x
            )
            df["Avg_Vol_20"] = df["Avg_Vol_20"].apply(
                lambda x: f"{int(x):,}" if pd.notnull(x) else x
            )

            return df
        except Exception as e:
            print(f"Error calculating moving averages: {e}")
            return pd.DataFrame()

    def _load_ticker_info(self) -> tuple[float, float, float]:
        """Yahoo profile + dividend metrics. No BigQuery access, so it can run alongside download_data()."""
        y_pct, p_pct = 0.0, 0.0
        div_rate = 0.0

        try:
            ticker_data = yf.Ticker(self.ticker)
            info = ticker_data.info

            # Populate class attributes inside the thread to execute only once
            self.ticker_info = info
            self.company_name = info.get("longName", info.get("shortName", self.ticker))
            self.sector = info.get("sector", "General")
            self.industry = info.get("industry", "Unknown")

            raw_rate = info.get("dividendRate")
            div_rate = float(raw_rate) if raw_rate is not None else 0.0
            raw_price = info.get("currentPrice") or info.get("previousClose")
            current_price = float(raw_price) if raw_price is not None else 1.0
            raw_payout = info.get("payoutRatio")
            payout_ratio = float(raw_payout) if raw_payout is not None else 0.0

            if div_rate > 0.0 and current_price > 0.0:
                y_pct = (div_rate / current_price) * 100
            p_pct = payout_ratio * 100
        except Exception as e:
            print(f"API Info Warning: {e}")

        return div_rate, y_pct, p_pct

    def _dividend_history(self) -> pd.DataFrame:
        """Ex-dividend rows from BigQuery. Must run after download_data() has finished."""
        query_df = pd.DataFrame()
        try:
            query_sql = f"""
                SELECT 
                    CAST(Date AS DATE) as Date, 
                    Open, 
                    Close, 
                    Dividends, 
                    ROUND(CAST(Close - Open AS NUMERIC), 2) as Price_Difference, 
                    ROUND(CAST(100 * (Close - Open) / Open AS NUMERIC), 2) as Daily_Percentage_Difference      
                FROM {self.table_name} 
                WHERE ticker = @ticker AND Dividends != 0
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "ticker", "STRING", self.ticker
                    )
                ]
            )
            query_df = bq_client.query(
                query_sql, job_config=job_config
            ).to_dataframe()
        except Exception as e:
            print(f"BigQuery Dividend Query Warning: {e}")

        return query_df

    def get_dividends(self) -> tuple[float, float, float, pd.DataFrame]:
        div_rate, y_pct, p_pct = self._load_ticker_info()
        return div_rate, y_pct, p_pct, self._dividend_history()

    def price_movement_plot(self):
        try:
            query_sql = f"SELECT Date, Close FROM {self.table_name} WHERE ticker = @ticker"
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "ticker", "STRING", self.ticker
                    )
                ]
            )
            df = bq_client.query(query_sql, job_config=job_config).to_dataframe()
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            df = df.sort_values("Date")
            df["60D_MA"] = df["Close"].rolling(window=60).mean()

            six_months_ago = df["Date"].max() - pd.DateOffset(months=6)
            df_viz = df[df["Date"] >= six_months_ago].reset_index(drop=True)

            if df_viz.empty:
                return None, None

            start_val = df_viz["Close"].iloc[0]
            end_val = df_viz["Close"].iloc[-1]
            perf_pct = ((end_val - start_val) / start_val) * 100
            min_row = df_viz.loc[df_viz["Close"].idxmin()]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df_viz["Date"],
                    y=df_viz["Close"],
                    name="Close Price",
                    line=dict(color="#006994", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_viz["Date"],
                    y=df_viz["60D_MA"],
                    name="60D MA",
                    line=dict(color="#CC5500", width=1.5, dash="dash"),
                )
            )

            last_row = df_viz.iloc[-1]
            fig.add_annotation(
                x=last_row["Date"],
                y=last_row["Close"],
                text="  Close Price",
                xanchor="left",
                yanchor="middle",
                font=dict(color="#006994", size=12),
                showarrow=False,
            )
            if not pd.isna(last_row["60D_MA"]):
                fig.add_annotation(
                    x=last_row["Date"],
                    y=last_row["60D_MA"],
                    text="  60D MA",
                    xanchor="left",
                    yanchor="middle",
                    font=dict(color="#CC5500", size=12),
                    showarrow=False,
                )
            fig.add_annotation(
                x=min_row["Date"],
                y=min_row["Close"],
                text=f"Period Low: ${min_row['Close']:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#87CEEB",
                ax=0,
                ay=35,
                font=dict(color="#87CEEB", size=11),
                bgcolor="#232B32",
                bordercolor="#000080",
            )

            fig.update_layout(
                paper_bgcolor="#232B32",
                plot_bgcolor="#232B32",
                hovermode="x unified",
                xaxis_title="",
                yaxis_title="Price ($)",
                margin=dict(l=20, r=90, t=15, b=20),
                showlegend=False,
            )

            return fig, {"perf_pct": perf_pct, "current_price": end_val}
        except Exception as e:
            print(f"Error rendering chart elements: {e}")
            return None, None

    def get_insider_data(self) -> str:
        try:
            company = Company(self.ticker)
            filings = company.get_filings(form="4").head(5)
            if len(filings) == 0:
                return "No recent Form 4 filings found."

            insider_report = []
            for filing in filings:
                form4 = filing.obj()
                summary = form4.get_ownership_summary()
                date_str = str(filing.filing_date)
                if summary.net_change != 0:
                    insider_report.append(
                        f"- {date_str} | {summary.insider_name} ({summary.position}):"
                        f" {summary.primary_activity} {summary.net_change:,} shares"
                    )

            return (
                "\n".join(insider_report)
                if insider_report
                else "No major buy/sell activity in recent filings."
            )
        except Exception as e:
            return f"SEC Tracking Unavailable ({e})"

    def build_sec_knowledge_base(self) -> int:
        with self._kb_lock:
            if self.sec_chunks:
                return len(self.sec_chunks)

            cached = _KB_CACHE.get(self.ticker)
            if cached and time.time() - cached["built_at"] < _KB_CACHE_TTL_SECONDS:
                self.sec_chunks = cached["chunks"]
                self.sec_embeddings = cached["embeddings"]
                print(f"⚡ Loaded {len(self.sec_chunks)} cached SEC chunks for {self.ticker}.")
                return len(self.sec_chunks)

            return self._build_sec_knowledge_base_uncached()

    def _build_sec_knowledge_base_uncached(self) -> int:
        try:
            print(f"📥 Connecting to SEC EDGAR for {self.ticker}...")
            sec_id = os.getenv("SEC_IDENTITY", "DataAnalyst user@example.com")
            set_identity(sec_id)

            company = Company(self.ticker)
            # Inside StockAnalyzer.build_sec_knowledge_base()

            filings = company.get_filings(form="10-Q")

            if len(filings) == 0:
                print(f"ℹ️ No 10-Q found for {self.ticker}, trying 10-K...")
                filings = company.get_filings(form="10-K")

            # Add these lines to support foreign issuers like NU, TSM, MELI
            if len(filings) == 0:
                print(f"ℹ️ No 10-K found, checking foreign issuer forms (20-F)...")
                filings = company.get_filings(form="20-F")

            if len(filings) == 0:
                print(f"ℹ️ Checking foreign issuer quarterly forms (6-K)...")
                filings = company.get_filings(form="6-K")

            doc_text = ""
            try:
                latest_filing = filings.head(1)[0]
                doc_text = latest_filing.text()
            except Exception as primary_err:
                print(f"⚠️ edgar could not read the filing text ({type(primary_err).__name__}: "
                      f"{primary_err}). Falling back to a direct SEC download...")
                doc_text = _fetch_filing_text_direct(company.cik, sec_id)

            if not doc_text:
                self.sec_error = "EDGAR returned an empty filing document."
                print(f"🔴 Document text for {self.ticker} is empty.")
                return 0

            rag = SECVectorRAG(self.ticker)
            raw_chunks = rag.chunk_text(doc_text)
            raw_embeddings = rag.embed_chunks(raw_chunks)

            # Map valid pairs to ensure exact index matching in RAM
            valid_pairs = [(c, e) for c, e in zip(raw_chunks, raw_embeddings) if len(e) > 0]

            if not valid_pairs:
                self.sec_error = (
                    "No embeddings were generated (Gemini embedding quota, key or model problem). "
                    "See the terminal for the 'Batch embedding failed' line."
                )
                print(f"🔴 No valid embeddings generated for {self.ticker}. Check API quota.")
                return 0

            dropped = len(raw_chunks) - len(valid_pairs)
            if dropped:
                print(f"⚠️ {dropped} of {len(raw_chunks)} chunks got no embedding for {self.ticker}; "
                      "retrieval will be incomplete.")

            self.sec_chunks = [p[0] for p in valid_pairs]
            self.sec_embeddings = np.vstack([p[1] for p in valid_pairs]).astype(np.float32)
            _KB_CACHE[self.ticker] = {
                "chunks": self.sec_chunks,
                "embeddings": self.sec_embeddings,
                "built_at": time.time(),
            }

            print(f"✅ Successfully loaded {len(self.sec_chunks)} vector chunks into RAM for {self.ticker}.")
            return len(self.sec_chunks)

        except Exception as e:
            import traceback
            self.sec_error = f"{type(e).__name__}: {e}"
            print(f"🔴 SEC RAM Build Failed for {self.ticker}: {e}")
            traceback.print_exc()  # shows which edgar call actually failed
            return 0

    def search_sec_filings(self, query: str, top_k: int = 3) -> pd.DataFrame:
        try:
            if not self.sec_chunks:
                self.build_sec_knowledge_base()

            if not self.sec_chunks:
                return pd.DataFrame()

            rag = SECVectorRAG(self.ticker)
            return rag.vector_search_in_memory(query, self.sec_chunks, self.sec_embeddings, top_k)
        except Exception as e:
            print(f"⚠️ SEC RAM Vector Search Failed: {e}")
            return pd.DataFrame()

    def analyze_sentiment(
            self, insider_text="", div_yield=0.0, payout_ratio=0.0
    ) -> str:
        try:
            news = self._get_news()
            headlines = [
                f"- {item.get('title', '')} ({item.get('publisher', '')})"
                for item in news[:5]
            ]
            news_text = "\n".join(headlines)

            try:
                query_sql = (
                    f"SELECT Date, Close, Volume FROM {self.table_name} "
                    "WHERE ticker = @ticker ORDER BY Date DESC LIMIT 30"
                )
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", self.ticker)]
                )
                df_recent = bq_client.query(query_sql, job_config=job_config).to_dataframe()
                price_context = df_recent.to_csv(index=False)
            except Exception:
                price_context = "Historical price context unavailable."

            prompt = (
                "You are a Senior Equity Research Analyst. Your objective is to"
                " provide a concise, actionable market sentiment synthesis for"
                f" {self.ticker}.\n\nDATA INPUTS:\n1. PRICE &"
                f" VOLUME:\n{price_context}\n2. RECENT NEWS:\n{news_text}\n3. SEC"
                f" INSIDER ACTIVITY:\n{insider_text}\n4. FUNDAMENTALS: Dividend"
                f" Yield: {div_yield:.2f}% | Payout Ratio:"
                f" {payout_ratio:.2f}%\n\nSTRICT INSTRUCTIONS:\n- Speak in crisp,"
                " practical financial terms. Avoid dramatic or verbose language.\n-"
                " If the Dividend Yield is 0.00%, treat it as a standard"
                " non-dividend paying asset.\n- Limit your reasoning to 2-3 short,"
                " punchy sentences.\n- IMPORTANT: Maintain UPPERCASE for all"
                " SENTIMENT and REASONING labels and use standard professional"
                " sentence casing for the body text.\n- Output exactly in the schema"
                " below No markdown wrappers, no introductory text.\n\nSENTIMENT: [🟢"
                " BULLISH / 🔴 BEARISH / ⚪ NEUTRAL]\nREASONING: [Provide a concise"
                " synthesis of how the price action aligns with the news and insider"
                " activity.]"
            )

            api_key = os.getenv("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)

            try:
                response = client.models.generate_content(
                    model="gemini-3.7-flash", contents=[prompt]
                )
                return response.text.strip() if response.text else "AI Analysis Empty"

            except Exception as api_error:
                return f"Gemini Error: {api_error}"

        except Exception as e:
            return f"Sentiment pipeline failed: {e}"

    def analyze_earnings_deep_dive(self) -> str:
        try:
            if not self.sec_chunks:
                print(f"RAM vectors empty for {self.ticker}. Initiating automatic 10-Q build...")
                self.build_sec_knowledge_base()

            search_themes = [
                "future guidance, revenue outlook, earnings per share projections",
                "profit margin compression, operating expenses, cost structure",
                (
                    "strategic shifts, macro headwinds, capital allocation, red"
                    " flags"
                ),
            ]

            context_chunks = []
            if self.sec_chunks:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_query = {
                        executor.submit(self.search_sec_filings, q, 5): q
                        for q in search_themes
                    }
                    for future in future_to_query:  # submission order keeps the prompt deterministic
                        results = future.result()
                        if not results.empty:
                            context_chunks.extend(results["chunk_text"].tolist())

            context_chunks = list(dict.fromkeys(context_chunks))
            sec_context = "\n\n...".join(context_chunks)

            if not sec_context:
                # Do not call the model on an empty context: it would answer "Insufficient data"
                # five times and that fake result would be cached by Streamlit.
                raise SECDataUnavailable(
                    f"No SEC filing text could be retrieved for {self.ticker}. "
                    f"Reason: {self.sec_error or 'unknown (check the terminal output)'}"
                )

            news = self._get_news()

            earnings_keywords = [
                "earn",
                "quarter",
                "q1",
                "q2",
                "q3",
                "q4",
                "eps",
                "revenue",
                "guidance",
                "margin",
            ]

            earnings_context_list = []
            for item in news:
                content = item.get("content", item)
                title = content.get("title", "") or ""
                summary = content.get("summary", content.get("text", "")) or ""
                if any(kw in title.lower() for kw in earnings_keywords):
                    earnings_context_list.append(f"- {title}: {summary[:400]}")

            earnings_context = "\n".join(earnings_context_list[:5])

            prompt = (
                "You are a Tier-1 Buy-Side Equity Analyst. Perform a ruthless"
                f" 5-pillar financial analysis on {self.ticker} based strictly on the provided context.\n\n"
                "AVAILABLE DATA:\n"
                f"RECENT NEWS:\n{earnings_context}\n\n"
                f"TARGETED SEC FILING RAG CHUNKS:\n{sec_context}\n\n"
                "INSTRUCTIONS:\n"
                "Analyze the provided text using the exact 5-pillar framework below. Your analysis must be "
                "grounded exclusively in the provided News and SEC chunks. If the provided data does not "
                "contain enough information for a specific pillar (e.g., there are no forward-looking statements "
                "to assess future guidance), you must explicitly state 'Insufficient data in available text' "
                "for that section. Do not hallucinate, infer, or use outside knowledge.\n\n"
                "OUTPUT SCHEMA:\n"
                "**1. Future Guidance (The Outlook):** [Address revisions, tone, forward-looking statements, and growth projections]\n"
                "**2. Profit Margins & Cost Pressures:** [Address pricing power, operating expenses, and margin compression/expansion]\n"
                "**3. Management's Discussion (MD&A):** [Address management's tone regarding operational challenges, liquidity, and market positioning]\n"
                "**4. Strategic Shifts & Macro:** [Address capital allocation, structural changes, and macroeconomic headwinds/tailwinds]\n"
                "**5. Red Flags & Risks:** [Address unfamiliar metrics, accounting shifts, explicit risk factors, or management excuses]\n"
                "---\n"
                "**OVERALL SENTIMENT:** [Strictly ONE word: POSITIVE, NEGATIVE, or NEUTRAL]"
            )
            api_key = os.getenv("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model="gemini-3.7-flash", contents=[prompt]
            )
            return (
                response.text.strip() if response.text else "Deep Dive Analysis Empty"
            )
        except SECDataUnavailable:
            raise  # let the caller show the real reason and skip caching
        except Exception as e:
            return f"Earnings Deep Dive pipeline failed: {e}"

    def fetch_pipeline_data_parallel(self):
        # Phase 1: independent network calls (market-data upsert, Yahoo profile, SEC Form 4)
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_download = executor.submit(self.download_data)
            future_info = executor.submit(self._load_ticker_info)
            future_insider = executor.submit(self.get_insider_data)

            success = future_download.result()
            div_rate, y_pct, p_pct = future_info.result()
            insider_text = future_insider.result()

        div_df = pd.DataFrame()
        plot_result = (None, None)
        signals_df = pd.DataFrame()

        if success:
            # Phase 2: BigQuery reads must wait until download_data() has finished its DELETE + reload
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_div = executor.submit(self._dividend_history)
                future_plot = executor.submit(self.price_movement_plot)
                future_signals = executor.submit(self.analyze_moving_averages)

                div_df = future_div.result()
                plot_result = future_plot.result()
                signals_df = future_signals.result()
            _ANALYZERS[self.ticker] = self

        self.view = {
            "plot": plot_result,
            "signals": signals_df,
            "dividends": (div_rate, y_pct, p_pct, div_df),
            "insider": insider_text,
        }
        return success, (div_rate, y_pct, p_pct, div_df), insider_text

    def run_ai_analysis_parallel(
            self, insider_text="", div_yield=0.0, payout_ratio=0.0
    ) -> tuple[str, str]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_sentiment = executor.submit(
                self.analyze_sentiment, insider_text, div_yield, payout_ratio
            )
            future_earnings = executor.submit(self.analyze_earnings_deep_dive)

            sentiment = future_sentiment.result()
            earnings = future_earnings.result()

        return sentiment, earnings

    def export_to_bigquery(self, df: pd.DataFrame, sentiment: str, earnings: str):
        """Persists macro sentiment and 5-pillar earnings analysis directly to BigQuery."""
        try:
            clean_ticker = str(self.ticker).upper().strip()[:10]
            clean_sector = str(self.sector)[:50] if self.sector else "General"
            clean_sentiment = (
                str(sentiment) if (pd.notna(sentiment) and sentiment) else "No Data"
            )
            clean_earnings = (
                str(earnings) if (pd.notna(earnings) and earnings) else "No Data"
            )

            print(
                f"📥 Writing {clean_ticker} record to"
                f" {PROJECT_ID}.{DATASET_ID}.nitrostox_analysis_results..."
            )

            insert_sql = f"""
                INSERT INTO `{PROJECT_ID}.{DATASET_ID}.nitrostox_analysis_results` (
                    ticker, 
                    sector, 
                    ai_macro_sentiment, 
                    ai_earnings_summary, 
                    pipeline_execution_time
                ) VALUES (
                    @ticker, 
                    @sector, 
                    @sentiment, 
                    @earnings, 
                    CURRENT_TIMESTAMP()
                );
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("ticker", "STRING", clean_ticker),
                    bigquery.ScalarQueryParameter("sector", "STRING", clean_sector),
                    bigquery.ScalarQueryParameter("sentiment", "STRING", clean_sentiment),
                    bigquery.ScalarQueryParameter("earnings", "STRING", clean_earnings),
                ]
            )
            bq_client.query(insert_sql, job_config=job_config).result()

            print(f"✅ Successfully committed {clean_ticker} record to BigQuery.")
            return True

        except Exception as e:
            print(f"🔴 DB Write Error in export_to_bigquery for {self.ticker}: {e}")
            raise e

    def run_pipeline(self, use_ai=True):
        print(f"Initializing pipeline execution for: {self.ticker}")

        success, (div_rate, y_pct, p_pct, div_df), insider_text = (
            self.fetch_pipeline_data_parallel()
        )

        if not success:
            print(
                "Critical pipeline stoppage: Unable to fetch core market asset"
                " history."
            )
            return

        ma_df = self.view["signals"]
        print("\nAlgorithmic Trading Signal Context:")
        print(
            ma_df.head(14).to_string(index=False)
            if not ma_df.empty
            else "No metrics generated."
        )

        print(
            f"\nDividend Metrics -> Rate: ${div_rate:.2f} | Yield: {y_pct:.2f}% |"
            f" Payout Ratio: {p_pct:.2f}%"
        )

        if not div_df.empty:
            print("\nRecent Ex-Dividend Price Action:")
            print(div_df.to_string(index=False))
        else:
            print("No dividend history found in the current timeframe.")

        print(f"\nSEC Insider Trading Profiles:\n{insider_text}")

        sentiment = "AI Bypassed"
        earnings_summary = "AI Bypassed"

        if use_ai:
            print("\nExecuting AI Analysis Pipeline (Macro & Earnings Deep-Dive)...")
            try:
                sentiment, earnings_summary = self.run_ai_analysis_parallel(
                    insider_text, y_pct, p_pct
                )
            except SECDataUnavailable as e:
                print(f"\n🔴 {e}\nNothing was saved to BigQuery.")
                return
            print(f"\nAI Macro Sentiment:\n{sentiment}")
            print(f"\nAI Earnings Summary:\n{earnings_summary}")
        else:
            print("\nAI Sentiment & Earnings Evaluation Bypassed.")

        self.export_to_bigquery(ma_df, sentiment, earnings_summary)


# STEP 5: AUTONOMOUS GEMINI 3.7 FLASH AGENT TOOL WRAPPERS & EXECUTOR


def _analyzer_for(ticker: str) -> "StockAnalyzer":
    """Reuses the analyzer the dashboard already loaded (sector, data); otherwise a fresh one."""
    t = str(ticker).upper().strip()
    return _ANALYZERS.get(t) or StockAnalyzer(t)


def tool_query_sec_filings(ticker: str, query: str) -> str:
    """Queries the in-memory vector database for SEC 10-Q filing text chunks matching a specific semantic query.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'PLTR').
        query: Specific keyword or financial topic to search (e.g. 'supply chain',
          'revenue guidance', 'capex').
    """
    try:
        analyzer = _analyzer_for(ticker)
        results_df = analyzer.search_sec_filings(query, top_k=3)
        if results_df.empty:
            return (
                f"No relevant SEC filing chunks found for {ticker} regarding"
                f" '{query}'."
            )
        chunks = results_df["chunk_text"].tolist()
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        return f"Error executing SEC vector search: {e}"


def tool_get_technical_signals(ticker: str) -> str:
    """Retrieves the latest BigQuery window calculated moving averages (7D/60D), RSI 14, and volume indicators for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'TSLA').
    """
    try:
        analyzer = _analyzer_for(ticker)
        df = analyzer.analyze_moving_averages()
        if df.empty:
            return f"No technical indicators available for {ticker}."
        return df.to_string(index=False)
    except Exception as e:
        return f"Error retrieving technical signals: {e}"


def tool_manage_watchlist(ticker: str, action: str) -> str:
    """Adds or removes a ticker symbol from the BigQuery watchlist dataset table.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        action: Strictly either 'ADD' or 'REMOVE'.
    """
    ticker_clean = str(ticker).upper().strip()
    action_clean = str(action).upper().strip()

    try:
        if action_clean == "ADD":
            query = f"""
                MERGE `{PROJECT_ID}.{DATASET_ID}.watchlist` T
                USING (SELECT @ticker AS symbol) S
                ON T.symbol = S.symbol
                WHEN NOT MATCHED THEN
                    INSERT (symbol, updated_at) VALUES (S.symbol, CURRENT_TIMESTAMP())
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker_clean)]
            )
            bq_client.query(query, job_config=job_config).result()
            return f"Successfully added {ticker_clean} to BigQuery watchlist."
        elif action_clean == "REMOVE":
            query = f"DELETE FROM `{PROJECT_ID}.{DATASET_ID}.watchlist` WHERE symbol = @ticker;"
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker_clean)]
            )
            bq_client.query(query, job_config=job_config).result()
            return f"Successfully removed {ticker_clean} from BigQuery watchlist."
        else:
            return f"Invalid action '{action}'. Action must be 'ADD' or 'REMOVE'."
    except Exception as e:
        return f"Failed to modify watchlist: {e}"


def run_gemini_agent(user_prompt: str, active_ticker: str = "AAPL", history: list | None = None) -> str:
    """Executes an autonomous ReAct chat agent turn using Gemini 3.7 Flash and tool calling."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        system_instruction = (
            "You are NitroStox Agent, an autonomous buy-side equity research"
            " assistant.\n"
            f"The user's currently selected ticker context is {active_ticker}.\n"
            "You have direct execution access to tools:\n"
            "1. tool_query_sec_filings (In-memory vector search on SEC 10-Q"
            " filings)\n"
            "2. tool_get_technical_signals (BigQuery window function indicator"
            " math)\n"
            "3. tool_manage_watchlist (Add or remove tickers from BigQuery"
            " watchlist)\n\n"
            "Autonomously call these tools when required to answer user questions."
            " Be concise, analytical, and professional."
        )

        chat = client.chats.create(
            model="gemini-3.7-flash",
            config={
                "tools": [
                    tool_query_sec_filings,
                    tool_get_technical_signals,
                    tool_manage_watchlist,
                ],
                "system_instruction": system_instruction,
            },
        )

        if history:
            transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:])
            user_prompt = f"Conversation so far:\n{transcript}\n\nNew message: {user_prompt}"

        response = chat.send_message(user_prompt)
        return (
            response.text
            if response.text
            else "Agent completed tool calls without narrative output."
        )

    except Exception as e:
        return f"Gemini Agent execution failed: {e}"


if __name__ == "__main__":
    user_ticker = input(
        "Please enter the ticker symbol to test backend standalone: "
    ).strip()
    if user_ticker:
        analyzer = StockAnalyzer(user_ticker)
        analyzer.run_pipeline(use_ai=True)