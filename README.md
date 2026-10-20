# 📈 NitroStox: NYSE Financial Intelligence Platform
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GCP-Cloud_Infrastructure-4285F4.svg?logo=googlecloud&logoColor=white)
![BigQuery](https://img.shields.io/badge/BigQuery-Data_Warehouse-669DF6.svg?logo=googlebigquery&logoColor=white)
![NumPy](https://img.shields.io/badge/Vector_Search-NumPy_In--Memory-013243.svg?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458.svg?logo=pandas&logoColor=white)
![SEC EDGAR](https://img.shields.io/badge/SEC-EDGAR_Data-005B9A.svg)
![Gemini AI](https://img.shields.io/badge/AI-Gemini_3.7_Flash-8E75B2.svg?logo=google-gemini&logoColor=white)
![ReAct Agent](https://img.shields.io/badge/Agent-ReAct_Tool_Calling-00A86B.svg?logo=google-gemini&logoColor=white)
![Eval Judge](https://img.shields.io/badge/Eval_Judge-Claude_Sonnet_5.5-D97757.svg?logo=anthropic&logoColor=white)

Pick a NYSE ticker and get, in one dashboard: rule-based trade signals computed in **BigQuery SQL**, an AI audit of the latest **SEC 10-Q** grounded in the filing text (RAG), live insider trades, dividend health, and a **ReAct chat agent** that can search the filing and query the warehouse.

<p align="center">
  <img src="docs/App_in_use.gif" alt="Agent checks PLTR's 10-Q and RSI, then adds PLTR to the watchlist" width="100%"/>
</p>

![NitroStox App Demo](docs/App_IN_USE.gif)
> [!WARNING]
> **Not investment advice.** An educational portfolio project. Signals and AI summaries are for information only, data may be delayed or wrong, and AI output can contain errors. See the [full disclaimer](#disclaimer).

## 💼 What It Does

* **Earnings audit (RAG):** Gemini audits the latest 10-Q against a 5-pillar framework (guidance, margins, MD&A, strategy and macro, red flags), using only retrieved filing text and recent earnings news. Pillars the text doesn't support are marked "Insufficient data in available text".
* **Trade signals:** BigQuery SQL classifies each ticker from moving averages, volume, RSI and Z-scores, with sector-specific rules (e.g. `TECH BREAKOUT` for Technology, `CAPITULATION BUY` for Utilities).
* **Insider activity:** net buying and selling from the five most recent Form 4 filings.
* **Sentiment:** a BULLISH / BEARISH / NEUTRAL read from price, volume, news, insider trades and dividends.
* **Dividend profile:** yield, payout ratio and history; payout above 100% is flagged as high risk.
* **Agent chat:** ask questions in plain English; the agent picks a tool, reads the result and answers.

> **One request, three tools.** The agent searched PLTR's 10-Q for cost risks, read its RSI from BigQuery, and, because both conditions were met, added PLTR to the watchlist table (left sidebar). I checked every point in this answer against the 10-Q and BigQuery: the quotes, the RSI and the watchlist entry are all correct.

<p align="center">
  <img src="docs/agent_watchlist_demo_arrow.png" alt="Agent checks PLTR's 10-Q and RSI, then adds PLTR to the watchlist" width="100%"/>
</p>

## 🏗️ Architecture

<p align="center">
  <img src="docs/Nitrostox_architecture.png" alt="NitroStox Architecture" width="100%"/>
</p>

* **In-warehouse analytics (ELT):** prices are loaded into BigQuery, where chained CTEs and window functions (`LAG`, `AVG`/`MAX`/`SUM OVER`) compute Wilder's RSI, moving averages and Z-scores.
* **In-memory hybrid RAG:** the filing is downloaded as Markdown and split by paragraph and table (up to 2,000 characters, 300 overlap). Every chunk starts with its section name, and every table chunk repeats the table's caption and column headers, so a row never loses its period. Each question is searched two ways: by meaning (`gemini-embedding-001` embeddings, ranked by NumPy cosine similarity) and by exact words (a small BM25 keyword search, so figures, years and line items like "net income" are matched literally). The two result lists are merged with Reciprocal Rank Fusion. Set `NITROSTOX_SEARCH_MODE` to `vector` or `keyword` to use one method alone. The index is cached per ticker for 24 hours, not stored in BigQuery.
* **ReAct agent:** Gemini 3.7 Flash calls three Python tools: filing search, BigQuery indicator lookup, and a BigQuery watchlist manager. It remembers the last six messages.
* **Cited answers:** the agent must tag every fact with the chunk it came from, like `[chunk 12]`, and say "The filing doesn't say" when the search finds nothing. The app checks each answer in code: cited chunks must be ones the search returned, every number needs a citation, and the number must appear in the cited chunk. If a check fails, the agent rewrites once; if it still fails, the answer shows a warning. The chat shows the cited text under each answer.
* **Parallel ingestion:** `ThreadPoolExecutor` fetches market data, dividends and insider trades concurrently, then runs the dependent BigQuery reads in parallel.
* **Cost control:** Streamlit caching (1-hour TTL on AI results) and the 24-hour filing cache stop widget clicks from re-querying BigQuery, Yahoo Finance, SEC EDGAR or the LLM.

<details>
<summary><b>🎯 Trade signal rules (click to expand)</b></summary>

Rules run top to bottom in BigQuery SQL; the first match wins. Sector rules come first.

| Signal | Applies to | Logic |
| :--- | :--- | :--- |
| ⚡ **CAPITULATION BUY** | Utilities, Consumer Defensive | `Z_Score < -2.0`, volume > 1.5x 20-day average, positive price change |
| ⚠️ **EXHAUSTED SELL** | Utilities, Consumer Defensive | `Z_Score > 2.0` and `RSI_14 > 70` |
| 🚀 **TECH BREAKOUT** | Technology, Consumer Cyclical | `MA_7 > MA_60`, `Close >= Local_High_20_day`, volume > 1.5x average, `RSI_14 < 70` |
| 🚀 **HIGH CONVICTION BUY** | Other sectors | Same as TECH BREAKOUT |
| ⚠️ **WEAK BREAKOUT** | All | Breakout without the volume or RSI conditions |
| ⚠️ **FLASH BREAKDOWN** | All | `Close < MA_60` while `MA_7 > MA_60` |
| ⚠️ **BREAKDOWN** | All | `MA_7 < MA_60` and `Close < MA_60` |
| 📉 **PULLBACK** | All | `MA_7 > MA_60`, `Close >= MA_60`, `Close < MA_7` |
| 🟢 **UPTREND** | All | `MA_7 > MA_60` and `Close >= MA_7` |
| 🟡 **RELIEF RALLY** | All | `MA_7 < MA_60` and `Close > MA_7` |
| 🔴 **DOWNTREND** | All | `MA_7 < MA_60` and `Close <= MA_7` |
| ⚪ **NEUTRAL** | All | No rule matches |

Utilities and Consumer Defensive have no volume-backed breakout rule, so their breakouts are labelled WEAK BREAKOUT. Labels such as BUY or SELL are not recommendations.
</details>

## 🧪 Evaluation

I tested the filing Q&A agent on **Amazon's latest 10-Q**: 20 questions, each asked twice, so **40 answers**. 14 questions are answered in the filing; 6 are not, to see if the agent admits it doesn't know. Every answer was graded by Claude (a different AI from the Gemini agent) against my own answer key.

<p align="center">
  <img src="docs/Nitrostox_eval_flow.png" alt="NitroStox evaluation flow" width="100%"/>
</p>

| Of 40 answers | Count | Example |
| :--- | :---: | :--- |
| ✅ Correct | **28** | Net income $62.6B, up from $18.2B (18 right answers + 10 correct "not in the filing") |
| 🟡 Partly right | **10** | Right numbers, but left out part of the explanation |
| ❌ Wrong | **0** | |
| 🚫 Made up or from outside knowledge | **0** | Gross margin isn't in the filing: the agent now says so instead of inventing 52.3% |

Every one of the 40 answers passed the code citation check, and the numbers in them all appear in the filing text the agent was given. The search step found the right passage for **26 of 28** answers to answerable questions.

**What the first test found**
1. **Tables lose their headers when chunked.** The agent reported $18.2B (last year's net income) instead of $62.6B (this quarter's).
2. **The chat agent isn't told to stick to the filing,** so it answers from general knowledge (CEO pay, market share).

**What I changed because of it**
- I made the agent cite the filing chunk behind every fact (for example `[chunk 14]`) and added a code check on those citations. Before this change the agent made up or used outside knowledge in 5 of 40 answers; now it says "The filing doesn't say" instead (10 of 12 unanswerable questions, and the other 2 were a stock price question it answered from the price table with a citation). In 16 of 40 answers the agent first had to rewrite its citations.
- I rebuilt the chunking ([`chunking.py`](src/chunking.py)) so every table chunk keeps its title and column headers. Net income is now right in both runs, and no answer in the re-test is wrong.
- I added keyword search next to the vector search, so the search is now **hybrid** ([`keyword_search.py`](src/keyword_search.py)). I ran the same test with each method (before the citations were added):

| Same 40 answers | Vector search only | Hybrid search |
| :--- | :---: | :---: |
| Fully correct (of 28 answerable) | 17 | **18** |
| Wrong | 0 | 0 |
| Said "not in the filing" when it should (of 12) | 6 | **7** |
| Right passage found by the search (of 28) | 26 | **27** |

  The difference is one answer, so hybrid matches vector on this test. It stays on as the default because it is safer for exact figures and years, and `NITROSTOX_SEARCH_MODE=vector` switches it off.

The first test was graded by hand and these by Claude, so their counts are not directly comparable.

**Still to do:** the 10 partly right answers are correct but incomplete (for example, they do not mention that the jump in net income came mostly from investment gains), and the judge found only 27 of 40 citations fully supported the claim next to them.

**How to run**

```bash
cd src
python preview_chunks.py                                   # saves the latest 10-Q as filing_<TICKER>.md and previews its chunks
python run_eval.py --filing filing_AMZN.md                 # retrieval test only
python run_eval.py --filing filing_AMZN.md --agent --runs 2  # also asks the agent every question and grades the answers
python run_eval.py --filing filing_AMZN.md --agent --runs 2 --search vector  # same test with vector search only (default is hybrid)
```

The default search is hybrid. To compare search methods, add `--search vector`, `--search hybrid` or `--search keyword`; `--search all` runs the retrieval test for all three side by side.

With `--agent`, a script flags figures that are not in the retrieved text and checks the citations (valid, present, and pointing at a chunk that holds the figure), and **Claude Sonnet 5.5** (a different model from the Gemini agent) grades each answer. This needs `ANTHROPIC_API_KEY` in `.env` and `pip install anthropic`; change the judge with `--judge-model` or skip it with `--no-judge`. Each run saves a results CSV, a trace of what the agent retrieved, and a JSON summary, with the search method in the file name.

<details>
<summary><b>📖 Glossary for beginners (click to expand)</b></summary>

* **Moving average (MA):** the average price over a set number of days; smooths out daily noise to show the trend.
* **Moving-average crossover:** the short-term average (7-day) crosses above the long-term one (60-day), a sign that new buyers are coming in.
* **RSI (Relative Strength Index):** a 0–100 momentum score. Above 70 is "overbought"; below 30 is "oversold".
* **Form 4:** the legal filing executives make when they buy or sell their own company's stock.
* **Dividend:** cash a company pays to its shareholders out of profits.
</details>

<details>
<summary><b>🚀 Setup & deployment (click to expand)</b></summary>

**1. Install**
```bash
pip install -r requirements.txt
```

**2. Create `.env`** (SEC EDGAR requires an email address as `SEC_IDENTITY`)
```env
GCP_PROJECT_ID=your_gcp_project_id
BQ_DATASET=nitrostox_analytics
GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account_key.json
SEC_IDENTITY=your_email@example.com
GEMINI_API_KEY=your_gemini_api_key
```

**3. Run**
```bash
cd src
python upgraded_nitrostox.py   # pipeline only: ingest, compute signals, run AI, save to BigQuery
streamlit run app.py           # dashboard
```

**4. Deploy to Cloud Run**
```bash
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/nitrostox-app
gcloud run deploy nitrostox-app \
  --image gcr.io/$GCP_PROJECT_ID/nitrostox-app \
  --platform managed --region us-central1 --allow-unauthenticated \
  --set-env-vars GCP_PROJECT_ID=$GCP_PROJECT_ID,BQ_DATASET=$BQ_DATASET,SEC_IDENTITY=$SEC_IDENTITY \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```
</details>

## Disclaimer

NitroStox is an educational portfolio project, not a financial product. Its signals, sentiment reads and filing summaries come from fixed rules and an AI model and are for information only. They are not investment, financial, tax or legal advice, nor a recommendation to buy or sell any security. Market data comes from public sources (Yahoo Finance, SEC EDGAR) and may be delayed, incomplete or wrong, and AI output can contain errors. Do your own research and consult a licensed financial advisor before investing. The author is not a licensed financial advisor and accepts no liability for decisions made using this tool.
