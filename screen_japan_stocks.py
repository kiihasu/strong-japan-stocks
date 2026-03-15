import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import yfinance as yf


JPX_LISTED_XLS_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

# ====== 条件設定 ======
MIN_1Y_RETURN = 0.30               # 1年騰落率 +30%以上
MIN_SALES_GROWTH = 0.10            # 売上成長率 +10%以上
MIN_MARKET_CAP = 30_000_000_000    # 300億円以上
MIN_AVG_TRADING_VALUE = 200_000_000  # 平均売買代金 2億円以上
AVG_TRADING_VALUE_DAYS = 20        # 平均売買代金の計算日数
MAX_WORKERS = 3                    # Yahoo Financeの制限回避のため控えめ
SLEEP_BETWEEN = 0.8                # アクセス間隔
OUTPUT_CSV = "strong_japan_stocks.csv"
OUTPUT_TICKERS = "tickers.txt"
# ======================


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


def calc_financial_conditions(ticker_obj: yf.Ticker):
    income_stmt = ticker_obj.income_stmt
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
        "sales_growth": float(sales_growth),
        "op_margin_latest": float(op_margin_latest),
        "op_margin_prev": float(op_margin_prev),
        "op_margin_uptrend": bool(op_margin_uptrend),
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

    required_cols = {"Close", "Volume"}
    if not required_cols.issubset(set(hist.columns)):
        return None

    hist = hist.copy()
    hist["Close"] = pd.to_numeric(hist["Close"], errors="coerce")
    hist["Volume"] = pd.to_numeric(hist["Volume"], errors="coerce")
    hist = hist.dropna(subset=["Close", "Volume"])

    if len(hist) < 220:
        return None

    close = hist["Close"]
    volume = hist["Volume"]

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

    trading_value = close * volume
    avg_trading_value = trading_value.tail(AVG_TRADING_VALUE_DAYS).mean()

    return {
        "latest_close": float(latest_close),
        "ma200": float(ma200),
        "one_year_return": float(one_year_return),
        "above_ma200": bool(above_ma200),
        "avg_trading_value": float(avg_trading_value),
        "price_date": str(idx_latest.date()),
    }


def get_market_cap(ticker_obj: yf.Ticker):
    market_cap = None

    try:
        info = ticker_obj.info
        market_cap = info.get("marketCap")
        if market_cap is not None:
            return float(market_cap)
    except Exception:
        pass

    try:
        fast_info = ticker_obj.fast_info
        market_cap = fast_info.get("market_cap")
        if market_cap is not None:
            return float(market_cap)
    except Exception:
        pass

    return None


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

        # まず価格条件
        if not (
            price_data["one_year_return"] >= MIN_1Y_RETURN
            and price_data["above_ma200"]
        ):
            return None

        # 売買代金フィルター
        if price_data["avg_trading_value"] < MIN_AVG_TRADING_VALUE:
            return None

        ticker_obj = yf.Ticker(ticker_symbol)

        # 時価総額フィルター
        market_cap = get_market_cap(ticker_obj)
        if market_cap is None or market_cap < MIN_MARKET_CAP:
            return None

        fin_data = calc_financial_conditions(ticker_obj)
        if fin_data is None:
            return None

        # ファンダ条件
        passed = (
            fin_data["sales_growth"] >= MIN_SALES_GROWTH
            and fin_data["op_margin_uptrend"]
            and fin_data["op_margin_latest"] >= 0
        )

        if not passed:
            return None

        return {
            "code": code,
            "ticker": ticker_symbol,
            "name": name,
            "market": market,
            "industry": industry,
            "price_date": price_data["price_date"],
            "latest_close": round(price_data["latest_close"], 2),
            "ma200": round(price_data["ma200"], 2),
            "one_year_return_pct": round(price_data["one_year_return"] * 100, 2),
            "above_ma200": price_data["above_ma200"],
            "avg_trading_value": int(price_data["avg_trading_value"]),
            "avg_trading_value_oku": round(price_data["avg_trading_value"] / 100_000_000, 2),
            "market_cap": int(market_cap),
            "market_cap_oku": round(market_cap / 100_000_000, 2),
            "sales_growth_pct": round(fin_data["sales_growth"] * 100, 2),
            "op_margin_prev_pct": round(fin_data["op_margin_prev"] * 100, 2),
            "op_margin_latest_pct": round(fin_data["op_margin_latest"] * 100, 2),
            "op_margin_uptrend": fin_data["op_margin_uptrend"],
            "passed": True,
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

    sort_cols = [
        "one_year_return_pct",
        "sales_growth_pct",
    ]
    existing_sort_cols = [c for c in sort_cols if c in passed_df.columns]
    if existing_sort_cols:
        passed_df = passed_df.sort_values(existing_sort_cols, ascending=False)

    passed_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    tickers_only = ",".join(passed_df["ticker"].tolist())
    with open(OUTPUT_TICKERS, "w", encoding="utf-8") as f:
        f.write(tickers_only)

    print("=" * 60)
    print(f"通過銘柄数: {len(passed_df)}")
    print(f"保存先: {OUTPUT_CSV}")
    print(f"保存先: {OUTPUT_TICKERS}")
    if len(passed_df) > 0:
        print("=== ティッカー一覧 ===")
        print(tickers_only)
        print("=" * 60)
        print(
            passed_df[
                [
                    "code",
                    "name",
                    "one_year_return_pct",
                    "sales_growth_pct",
                    "op_margin_prev_pct",
                    "op_margin_latest_pct",
                    "market_cap_oku",
                    "avg_trading_value_oku",
                ]
            ].head(30).to_string(index=False)
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
