import pandas as pd
import yfinance as yf

import continuous_scanner

scanner_data = continuous_scanner.scanner_data

_ORIG_GOLD = scanner_data.twelve
# Yahoo no longer serves the old XAUUSD=X symbol reliably.  GC=F provides
# liquid COMEX gold futures candles and is used only as the automatic backup
# when Twelve Data is unavailable/rate-limited.
GOLD_YAHOO_SYMBOL = "GC=F"


def _normalize(df):
    if df is None or df.empty:
        raise RuntimeError("Yahoo gold fallback returned no data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={
        "Open": "o",
        "High": "h",
        "Low": "l",
        "Close": "c",
        "Volume": "v",
    })
    if "v" not in df:
        df["v"] = 0.0
    df = df[["o", "h", "l", "c", "v"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df["v"] = pd.to_numeric(df["v"], errors="coerce").fillna(0.0)
    return df.dropna(subset=["o", "h", "l", "c"]).sort_index()


def _yahoo_gold(tf, now):
    if tf == "15m":
        interval, period = "15m", "1mo"
    elif tf in ("1h", "4h"):
        interval, period = "1h", "6mo"
    elif tf == "1d":
        interval, period = "1d", "2y"
    else:
        raise RuntimeError(f"Unsupported gold timeframe: {tf}")

    raw = _normalize(yf.download(
        GOLD_YAHOO_SYMBOL,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    ))

    if tf == "4h":
        raw = raw.resample("4h", label="left", closed="left").agg({
            "o": "first",
            "h": "max",
            "l": "min",
            "c": "last",
            "v": "sum",
        }).dropna(subset=["o", "h", "l", "c"])

    frame = scanner_data.clean(raw, tf, now)
    frame.attrs["symbol"] = GOLD_YAHOO_SYMBOL
    return frame


def resilient_gold(symbol, tf, now):
    try:
        return _ORIG_GOLD(symbol, tf, now)
    except Exception as exc:
        print(
            f"GOLD {tf}: Twelve Data unavailable -> Yahoo {GOLD_YAHOO_SYMBOL} ({exc})",
            flush=True,
        )
        return _yahoo_gold(tf, now)


scanner_data.twelve = resilient_gold


if __name__ == "__main__":
    continuous_scanner.main()
