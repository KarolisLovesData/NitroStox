/*
  NitroStox Analytical Engine - Technical Indicator Pipeline
  Calculates 14-period RSI, Moving Averages (7D/60D), and Z-Scores using advanced Window Functions.
*/

WITH BaseData AS (
    -- Step 1: Establish foundational metrics, rolling highs, and daily returns.
    SELECT
        "Date",
        "Close",
        "Volume",
        -- Calculate day-over-day price change using LAG
        "Close" - LAG("Close") OVER(ORDER BY "Date" ASC) AS "Price_Change",
        ("Close" - LAG("Close") OVER(ORDER BY "Date" ASC)) / LAG("Close") OVER(ORDER BY "Date" ASC) * 100 AS "Daily_Return_Pct",
        -- Identify the local 20-day high and average volume for breakout detection
        MAX("Close") OVER(ORDER BY "Date" ASC ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS "Local_High_20_day",
        AVG("Volume") OVER(ORDER BY "Date" ASC ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS "Avg_Vol_20_day",
        -- Establish the 60-day baseline moving average
        AVG("Close") OVER(ORDER BY "Date" ASC ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS "MA_60_day",
        ROW_NUMBER() OVER(ORDER BY "Date" ASC) AS "rn"
    FROM market_data_table
),
GainsLosses AS (
    -- Step 2: Isolate positive and negative price movements for RSI calculation.
    SELECT *,
        CASE WHEN "Price_Change" > 0 THEN "Price_Change" ELSE 0 END AS "Gain",
        CASE WHEN "Price_Change" < 0 THEN ABS("Price_Change") ELSE 0 END AS "Loss",
        -- Calculate squared deviation for standard deviation/Z-Score math later
        ("Close" - "MA_60_day") * ("Close" - "MA_60_day") AS "Squared_Dev"
    FROM BaseData
),
WilderDecay AS (
    -- Step 3: Apply Wilder's Exponential Decay Factor: (13/14)^rn
    -- This weights recent price action more heavily for the RSI indicator.
    SELECT *,
        "Gain" * POWER(13.0 / 14.0, -"rn") AS "Weighted_Gain",
        "Loss" * POWER(13.0 / 14.0, -"rn") AS "Weighted_Loss",
        POWER(13.0 / 14.0, "rn") AS "Scale_Factor"
    FROM GainsLosses
),
Averages AS (
    -- Step 4: Vectorized Exponential Smoothing for Gains & Losses
    SELECT *,
        -- Smooth the weighted gains and losses over the 14-day period
        SUM("Weighted_Gain") OVER(ORDER BY "Date" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) * "Scale_Factor" / 14.0 AS "Avg_Gain_14",
        SUM("Weighted_Loss") OVER(ORDER BY "Date" ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) * "Scale_Factor" / 14.0 AS "Avg_Loss_14",
        -- Calculate the fast 7-day moving average
        AVG("Close") OVER(ORDER BY "Date" ASC ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS "MA_7_day",
        -- Calculate the 60-day variance
        AVG("Squared_Dev") OVER(ORDER BY "Date" ASC ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS "Variance_60"
    FROM WilderDecay
),
RSI_Calc AS (
    -- Step 5: Finalize Indicator Math (RSI and Z-Score)
    SELECT *,
        -- Prevent division by zero and calculate the final 14-period RSI
        CASE
            WHEN "Avg_Loss_14" = 0 THEN 100.0
            WHEN "Avg_Gain_14" = 0 THEN 0.0
            ELSE 100.0 - (100.0 / (1.0 + ("Avg_Gain_14" / "Avg_Loss_14")))
        END AS "RSI_14",
        -- Calculate Z-Score relative to the 60-day moving average
        CASE
            WHEN "Variance_60" <= 0 THEN 0
            ELSE ("Close" - "MA_60_day") / SQRT("Variance_60")
        END AS "Z_Score"
    FROM Averages
)
-- Step 6: Final Output and Categorical Signal Generation
SELECT
    CAST("Date" AS DATE) AS "Date",
    ROUND(CAST("Close" AS NUMERIC), 2) AS "Close_Price",
    CASE
        -- Dynamic High-Conviction Signal Logic
        WHEN "MA_7_day" > "MA_60_day" AND "Close" >= "Local_High_20_day" AND "Volume" > (1.5 * "Avg_Vol_20_day") AND "RSI_14" < 70 THEN '🚀 HIGH CONVICTION BUY'

        -- Breakout Warning: Price clears previous 20-day high but lacks optimal volume/RSI parameters
        WHEN "MA_7_day" > "MA_60_day" AND "Close" > "Local_High_20_day" THEN '⚠️ WEAK BREAKOUT (Check Vol/RSI)'

        -- Flash Breakdown: Macro fast trend is intact (MA_7 > MA_60), but price temporarily pierces below 60D
        WHEN "MA_7_day" > "MA_60_day" AND "Close" < "MA_60_day" THEN '⚠️ FLASH BREAKDOWN'

        -- Structural Breakdown: Fast MA has crossed below 60D AND price is under the 60D baseline
        WHEN "MA_7_day" < "MA_60_day" AND "Close" < "MA_60_day" THEN '⚠️ BREAKDOWN'

        -- Short-term Pullback: Macro trend intact, price dips below fast MA but holds above 60D support
        WHEN "MA_7_day" > "MA_60_day" AND "Close" < "MA_7_day" AND "Close" >= "MA_60_day" THEN '📉 PULLBACK'

        -- Refined Uptrend Confirmation
        WHEN "MA_7_day" > "MA_60_day" AND "Close" >= "MA_7_day" THEN '🟢 UPTREND'

        -- Relief Rally: Macro trend is down, but short-term price bounces above fast MA
        WHEN "MA_7_day" < "MA_60_day" AND "Close" > "MA_7_day" THEN '🟡 RELIEF RALLY'

        -- Confirmed Downtrend: Macro trend is down AND price is actively falling below fast MA
        WHEN "MA_7_day" < "MA_60_day" AND "Close" <= "MA_7_day" THEN '🔴 DOWNTREND'

        ELSE '⚪ NEUTRAL'
    END AS "Signal",
    -- Format Volume Deviation as readable text (e.g., '1.2M')
    CAST(ROUND(CAST(("Volume" - "Avg_Vol_20_day") / 1000000.0 AS NUMERIC), 1) AS TEXT) || 'M' AS "Volume_Deviation",
    ROUND(CAST("MA_7_day" AS NUMERIC), 2) AS "MA_7",
    ROUND(CAST("MA_60_day" AS NUMERIC), 2) AS "MA_60",
    ROUND(CAST("RSI_14" AS NUMERIC), 2) AS "RSI_14",
    "Volume",
    ROUND(CAST("Avg_Vol_20_day" AS NUMERIC), 0) AS "Avg_Vol_20"
FROM RSI_Calc
ORDER BY "Date" DESC
LIMIT 15;


/*
  Persists the output of the macro sentiment and earnings LLM pipelines
  into the results lakebase alongside execution metadata.
*/

INSERT INTO public.nitrostox_analysis_results (
    ticker,
    sector,
    ai_macro_sentiment,
    ai_earnings_summary,
    pipeline_execution_time
) VALUES (
    :ticker,
    :sector,
    :sentiment,
    :earnings,
    NOW()
);