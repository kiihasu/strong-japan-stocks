import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import yfinance as yf


JPX_LISTED_XLS_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

MIN_1Y_RETURN = 0.30
MIN_SALES_GROWTH = 0.10
MAX_WORKERS = 8
SLEEP_BETWEEN = 0.2
OUTPUT_CSV = "strong_japan_stocks.csv"


def download_jpx_listed_companies() -> pd.DataFrame:
    r = requests.get(JPX_LISTED_XLS_URL, timeout=60)
    r.raise_for_status()

    xls_bytes = io.BytesIO(r.content)
    df = pd.read_excel(xls_bytes)
    df.columns = [str(c).strip() for c in df.columns]

    code_col = next((c for c in df.columns if "コード" in c), None)
    name_col = next((c for c in df.columns if "銘柄名" in c or "銘柄略称" in c), None)
    market_col = next((c for c in df.columns if "市場・商品区分" in c), None)
    industry_col = next((c for c in df.columns if "33業種区分" in c or "17業種区分" in c), None)

    if code_col is None or name_col is None:
        raise ValueError(f"JPX一覧の列構造を認識できません: {df.columns.tolist()}")

    if market_col is not None:
        market_series = df[market_col].astype(str)
        df = df[market_series.str.contains("内国株式", na=False)].copy()

    df[code_col] = df[code_col].astype(str).str.extract(r"(\d{4})")
    df = df.dropna(subset=[code_col]).copy()

    out = df[[code_col, name_col]].copy()
    out.columns = ["code", "name"]
    out["market"] = df[market_col].values if market_col is not None else ""
    out["industry"] = df[industry_col].values if industry_col is not None else ""
    out["ticker"] = out["code"] + ".T"

    return out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def pick_first_existing_row(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None

    index_map = {str(idx).strip().lower(): idx for idx in df.index}
    for cand in candidates:
        key = cand.strip().lower()
        if key in index_map:
            return index_map[key]
    return None


def get_latest_two_year_values(income_stmt: pd.DataFrame, row_name: str):
    if income_stmt is None or income_stmt.empty or row_name not in income_stmt.index:
        return None, None

    cols = list(income_stmt.columns)
    try:
        cols_sorted = sorted(cols)
        s = income_stmt.loc[row_name, cols_sorted]
    except Exception:
        s = income_stmt.loc[row_name]

    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return None, None

    s = s.sort_index(ascending=False)
    latest = s.iloc[0]
    prev = s.iloc[1]
    return latest, prev


def calc_financial_conditions(ticker: yf.Ticker):
    income_stmt = ticker.income_stmt
    if income_stmt is None or income_stmt.empty:
        return None

    revenue_candidates = [
        "Total Revenue",
        "Revenue",
        "Operating Revenue",
        "Net Sales",
        "Total Revenues",
    ]
    op_income_candidates = [
        "Operating Income",
        "EBIT",
    ]

    revenue_row = pick_first_existing_row(income_stmt, revenue_candidates)
    op_income_row = pick_first_existing_row(income_stmt, op_income_candidates)

    if revenue_row is None or op_income_row is None:
        return None

    rev_latest, rev_prev = get_latest_two_year_values(income_stmt, revenue_row)
    op_latest, op_prev = get_latest_two_year_values(income_stmt, op_income_row)

    if any(v is None for v in [rev_latest, rev_prev, op_latest, op_prev]):
        return None
    if rev_prev == 0 or rev_latest == 0:
        return None

    sales_growth = (rev_latest / rev_prev) - 1.0
    op_margin_latest = op_latest / rev_latest
    op_margin_prev = op_prev / rev_prev
    op_margin_uptrend = op_margin_latest > op_margin_prev

    return {
        "sales_growth": sales_growth,
        "op_margin_latest": op_margin_latest,
        "op_margin_prev": op_margin_prev,
        "op_margin_uptrend": op_margin_uptrend,
    }


def calc_price_conditions(ticker_symbol: str):
    hist = yf.download(
        ticker_symbol,
        period="18mo",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if hist is None or hist.empty:
        return None

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    if "Close" not in hist.columns:
        return None

    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if len(close) < 220:
        return None

    ma200 = close.rolling(200).mean().iloc[-1]
    latest_close = close.iloc[-1]

    idx_latest = close.index[-1]
    one_year_ago = idx_latest - pd.Timedelta(days=365)
    past_series = close[close.index <= one_year_ago]
    if len(past_series) == 0:
        return None

    close_1y_ago = past_series.iloc[-1]
    one_year_return = (latest_close / close_1y_ago) - 1.0
    above_ma200 = latest_close > ma200

    return {
        "latest_close": float(latest_close),
        "ma200": float(ma200),
        "one_year_return": float(one_year_return),
        "above_ma200": bool(above_ma200),
        "price_date": str(idx_latest.date()),
    }


def analyze_one(row: pd.Series):
    ticker_symbol = row["ticker"]
    code = row["code"]
    name = row["name"]
    market = row.get("market", "")
    industry = row.get("industry", "")

    try:
        price_data = calc_price_conditions(ticker_symbol)
        if price_data is None:
            return None

        if not (
            price_data["one_year_return"] >= MIN_1Y_RETURN and
            price_data["above_ma200"]
        ):
            return None

        ticker = yf.Ticker(ticker_symbol)
        fin_data = calc_financial_conditions(ticker)
        if fin_data is None:
            return None

        passed = (
            fin_data["sales_growth"] >= MIN_SALES_GROWTH and
            fin_data["op_margin_uptrend"]
        )

        return {
            "code": code,
            "ticker": ticker_symbol,
            "name": name,
            "market": market,
            "industry": industry,
            "price_date": price_data["price_date"],
            "latest_close": price_data["latest_close"],
            "ma200": price_data["ma200"],
            "one_year_return_pct": round(price_data["one_year_return"] * 100, 2),
            "above_ma200": price_data["above_ma200"],
            "sales_growth_pct": round(fin_data["sales_growth"] * 100, 2),
            "op_margin_latest_pct": round(fin_data["op_margin_latest"] * 100, 2),
            "op_margin_prev_pct": round(fin_data["op_margin_prev"] * 100, 2),
            "op_margin_uptrend": fin_data["op_margin_uptrend"],
            "passed": passed,
        }

    except Exception as e:
        return {
            "code": code,
            "ticker": ticker_symbol,
            "name": name,
            "market": market,
            "industry": industry,
            "error": str(e),
            "passed": False,
        }
    finally:
        time.sleep(SLEEP_BETWEEN)


def main():
    print("JPX 上場銘柄一覧を取得中...")
    universe = download_jpx_listed_companies()
    print(f"対象銘柄数: {len(universe)}")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(analyze_one, row) for _, row in universe.iterrows()]
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res is not None:
                results.append(res)

            if i % 100 == 0:
                print(f"{i} / {len(futures)} 完了")

    df = pd.DataFrame(results)

    if df.empty:
        print("結果が空です。")
        return

    passed_df = df[df["passed"] == True].copy()

    sort_cols = []
    if "one_year_return_pct" in passed_df.columns:
        sort_cols.append("one_year_return_pct")
    if "sales_growth_pct" in passed_df.columns:
        sort_cols.append("sales_growth_pct")

    if sort_cols:
        passed_df = passed_df.sort_values(sort_cols, ascending=False)

    passed_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"通過銘柄数: {len(passed_df)}")
    print(f"保存先: {OUTPUT_CSV}")
    if len(passed_df) > 0:
        print(passed_df[[
            "code", "name", "one_year_return_pct",
            "sales_growth_pct", "op_margin_prev_pct", "op_margin_latest_pct"
        ]].head(30).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
