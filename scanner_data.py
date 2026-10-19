import logging
import os
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests


LOG = logging.getLogger(__name__)

DUR = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}

REDACTIONS = (
    "TWELVE_KEY", "GEMINI_KEY", "TG_TOKEN", "TG_CHAT"
)


def safe_error(exc):
    text = str(exc)

    for name in REDACTIONS:
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[REDACTED]")

    return text[:250]


def get_json(url, params=None):
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=(8, 25),
            )
        except requests.RequestException:
            if attempt == 2:
                raise RuntimeError(
                    f"{urlparse(url).hostname}: network failure"
                ) from None

            time.sleep(1 + attempt)
            continue

        if (
            response.status_code in (429, 500, 502, 503, 504)
            and attempt < 2
        ):
            delay = int(
                response.headers.get("Retry-After", "2")
            )
            time.sleep(min(10, max(1, delay)))
            continue

        if not response.ok:
            raise RuntimeError(
                f"{urlparse(url).hostname}: "
                f"HTTP {response.status_code}"
            )

        try:
            return response.json()
        except ValueError:
            raise RuntimeError(
                f"{urlparse(url).hostname}: invalid JSON"
            ) from None

    raise RuntimeError("request failed")


def clean(data, tf, now, close_times=None):
    data = data.copy()
    data.index = pd.to_datetime(data.index, utc=True)

    if data.index.has_duplicates:
        raise ValueError("duplicate candle timestamps")

    data = data.sort_index()

    for column in ("o", "h", "l", "c", "v"):
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    if close_times is not None:
        data["end"] = pd.Series(
            pd.to_datetime(close_times, utc=True),
            index=data.index,
        )
    else:
        data["end"] = data.index + DUR[tf]

    data = data[
        data["end"] <= now - pd.Timedelta(seconds=2)
    ]

    if data.empty:
        raise ValueError("no closed candles")

    columns = ["o", "h", "l", "c", "v"]

    if not np.isfinite(data[columns].to_numpy()).all():
        raise ValueError("nonfinite OHLCV")

    invalid = (
        (data[["o", "h", "l", "c"]] <= 0).any().any()
        or (data["v"] < 0).any()
        or (
            data["h"]
            < data[["o", "l", "c"]].max(axis=1)
        ).any()
        or (
            data["l"]
            > data[["o", "h", "c"]].min(axis=1)
        ).any()
    )

    if invalid:
        raise ValueError("invalid OHLCV bounds")

    return data


def binance(symbol, tf, now):
    hosts = (
        "https://data-api.binance.vision",
        "https://api.binance.com",
    )

    for host in hosts:
        try:
            rows = get_json(
                host + "/api/v3/klines",
                {
                    "symbol": symbol,
                    "interval": tf,
                    "limit": 500,
                },
            )

            if not isinstance(rows, list) or not rows:
                raise ValueError("no klines")

            data = pd.DataFrame(rows).iloc[
                :, [0, 1, 2, 3, 4, 5, 6, 9]
            ]

            data.columns = [
                "t", "o", "h", "l",
                "c", "v", "end", "tb",
            ]

            data.index = pd.to_datetime(
                data.pop("t").astype("int64"),
                unit="ms",
                utc=True,
            )

            ends = pd.to_datetime(
                data.pop("end").astype("int64") + 1,
                unit="ms",
                utc=True,
            )

            data["tb"] = pd.to_numeric(data["tb"])
            output = clean(data, tf, now, ends)

            if (
                (output["tb"] < 0)
                | (output["tb"] > output["v"])
            ).any():
                raise ValueError("invalid taker volume")

            return output

        except Exception as exc:
            LOG.warning(
                "BINANCE %s %s %s: %s",
                symbol, tf, host, safe_error(exc),
            )

    raise RuntimeError(
        "both Binance public endpoints failed"
    )


def okx(symbol, tf, now):
    kind = "SWAP" if symbol.endswith("-SWAP") else "SPOT"

    instruments = get_json(
        "https://www.okx.com/api/v5/public/instruments",
        {
            "instType": kind,
            "instId": symbol,
        },
    )

    valid = (
        instruments.get("code") == "0"
        and any(
            row.get("instId") == symbol
            and row.get("state") == "live"
            for row in instruments.get("data", [])
        )
    )

    if not valid:
        raise ValueError(
            f"OKX instrument not verified: {symbol}"
        )

    bar = {
        "15m": "15m",
        "1h": "1H",
        "4h": "4H",
        "1d": "1Dutc",
    }[tf]

    payload = get_json(
        "https://www.okx.com/api/v5/market/candles",
        {
            "instId": symbol,
            "bar": bar,
            "limit": 300,
        },
    )

    if (
        payload.get("code") != "0"
        or not payload.get("data")
    ):
        raise RuntimeError("OKX candles rejected")

    rows = [
        row for row in payload["data"]
        if row[8] == "1"
    ]

    data = pd.DataFrame(rows).iloc[:, :6]
    data.columns = ["t", "o", "h", "l", "c", "v"]

    data.index = pd.to_datetime(
        data.pop("t").astype("int64"),
        unit="ms",
        utc=True,
    )

    return clean(data, tf, now)


def twelve(symbol, tf, now):
    key = os.environ.get("TWELVE_KEY")

    if not key:
        raise ValueError("TWELVE_KEY missing")

    interval = {
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day",
    }[tf]

    payload = get_json(
        "https://api.twelvedata.com/time_series",
        {
            "symbol": symbol,
            "interval": interval,
            "outputsize": 400,
            "timezone": "UTC",
            "apikey": key,
        },
    )

    if not payload.get("values"):
        raise ValueError(
            "Twelve Data unavailable: "
            + str(payload.get("code", "no values"))
        )

    data = pd.DataFrame(payload["values"])
    data.index = pd.to_datetime(
        data.pop("datetime"), utc=True
    )

    data = data.rename(columns={
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
    })

    if "v" not in data:
        data["v"] = 0.0

    return clean(
        data[["o", "h", "l", "c", "v"]],
        tf,
        now,
    )


def yahoo(symbol, tf, now):
    import yfinance as yf

    interval = (
        "1h"
        if tf == "4h"
        else {
            "15m": "15m",
            "1h": "1h",
            "1d": "1d",
        }[tf]
    )

    period = (
        "1mo"
        if tf == "15m"
        else ("2y" if tf == "1d" else "6mo")
    )

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if data.empty:
        raise ValueError("Yahoo empty response")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns={
        "Open": "o",
        "High": "h",
        "Low": "l",
        "Close": "c",
        "Volume": "v",
    })

    data = data[["o", "h", "l", "c", "v"]].copy()
    data.index = pd.to_datetime(data.index, utc=True)

    if symbol == "^NDX" and tf != "1d":
        import pandas_market_calendars as mcal

        schedule = mcal.get_calendar(
            "NASDAQ"
        ).schedule(
            start_date=data.index.min().date(),
            end_date=now.date(),
        )

        pieces = []

        for _, session in schedule.iterrows():
            opened = session["market_open"]
            closed = session["market_close"]

            part = data[
                (data.index >= opened)
                & (data.index < closed)
            ].copy()

            if part.empty:
                continue

            base_tf = "1h" if tf == "4h" else tf

            ends = [
                min(t + DUR[base_tf], closed)
                for t in part.index
            ]

            available = [
                i for i, end in enumerate(ends)
                if end <= now - pd.Timedelta(seconds=2)
            ]

            if not available:
                continue

            part = part.iloc[available]
            ends = [ends[i] for i in available]
            part = clean(part, base_tf, now, ends)

            if tf == "4h":
                part = part[
                    (part["end"] - part.index) == DUR["1h"]
                ]

                part["bucket"] = (
                    (part.index - opened) // DUR["4h"]
                ).astype(int)

                for _, group in part.groupby("bucket"):
                    if (
                        len(group) != 4
                        or (
                            group["end"].iloc[-1]
                            - group.index[0]
                        ) != DUR["4h"]
                    ):
                        continue

                    pieces.append(pd.DataFrame({
                        "o": [group["o"].iloc[0]],
                        "h": [group["h"].max()],
                        "l": [group["l"].min()],
                        "c": [group["c"].iloc[-1]],
                        "v": [group["v"].sum()],
                        "end": [group["end"].iloc[-1]],
                    }, index=[group.index[0]]))

            else:
                pieces.append(part)

        if not pieces:
            raise ValueError(
                "no completed regular-session candles"
            )

        data = pd.concat(pieces).sort_index()

        return clean(
            data, tf, now, data["end"]
        )

    return clean(data, tf, now)


def specs(key):
    if key in ("BTC", "ETH", "SOL"):
        return [("binance", key + "USDT")]

    if key == "GOLD":
        provider = os.environ.get(
            "GOLD_PROVIDER", "twelve"
        )
        symbol = os.environ.get(
            "GOLD_SYMBOL",
            "XAU/USD" if provider == "twelve" else "",
        )
    else:
        provider = os.environ.get(
            "NDX_PROVIDER", "yahoo"
        )
        symbol = os.environ.get(
            "NDX_SYMBOL",
            "^NDX" if provider == "yahoo" else "",
        )

    if not symbol:
        raise ValueError(
            "explicit instrument symbol required "
            "for this provider"
        )

    if provider not in ("twelve", "okx", "yahoo"):
        raise ValueError("unsupported provider")

    return [(provider, symbol)]


def frames(key, now):
    provider, symbol = specs(key)[0]

    getter = {
        "binance": binance,
        "okx": okx,
        "twelve": twelve,
        "yahoo": yahoo,
    }[provider]

    output = {}

    for tf in ("15m", "1h", "4h", "1d"):
        if key in ("BTC", "ETH", "SOL") and tf == "1d":
            continue

        try:
            output[tf] = getter(symbol, tf, now)
            data = output[tf]

            LOG.info(
                "SOURCE asset=%s provider=%s symbol=%s "
                "tf=%s bars=%d last_close=%s "
                "age_seconds=%.0f volume=%s",
                key,
                provider,
                symbol,
                tf,
                len(data),
                data["end"].iloc[-1].isoformat(),
                (
                    now - data["end"].iloc[-1]
                ).total_seconds(),
                (
                    "present"
                    if data["v"].sum() > 0
                    else "unavailable"
                ),
            )

        except Exception as exc:
            if tf != "1d":
                raise RuntimeError(
                    f"{provider}:{symbol} {tf}: "
                    f"{safe_error(exc)}"
                ) from None

            LOG.warning(
                "%s daily context unavailable: %s",
                key,
                safe_error(exc),
            )

    output["_source"] = provider + ":" + symbol
    output["_provider"] = provider
    output["_symbol"] = symbol

    return output


def freshness(fr, key, now):
    needed = {
        "15m": 150,
        "1h": 220,
        "4h": 60,
    }

    for tf, minimum in needed.items():
        if len(fr[tf]) < minimum:
            return (
                False,
                f"{tf}: only {len(fr[tf])}/{minimum} candles",
            )

    if (
        now - fr["15m"]["end"].iloc[-1]
        > pd.Timedelta(minutes=25)
    ):
        return False, "15m data stale or market closed"

    if key in ("BTC", "ETH", "SOL"):
        for tf in needed:
            if (
                now - fr[tf]["end"].iloc[-1]
                > DUR[tf] + pd.Timedelta(minutes=25)
            ):
                return False, f"{tf}: stale"

            differences = (
                fr[tf].tail(60).index.to_series()
                .diff().dropna()
            )

            if (differences != DUR[tf]).any():
                return False, f"{tf}: missing candle(s)"

    else:
        for tf, bound in (("1h", 2), ("4h", 96)):
            if (
                now - fr[tf]["end"].iloc[-1]
                > pd.Timedelta(hours=bound)
            ):
                return False, f"{tf}: context too old"

    return True, "OK"