import logging
import math
import os

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
_CACHE = None
_CACHE_TIME = None
MINUTE = pd.Timedelta(minutes=15)


def utc(value):
    t = pd.Timestamp(value)
    return (
        t.tz_localize("UTC")
        if t.tzinfo is None
        else t.tz_convert("UTC")
    )


def distinct(data):
    d = data.copy()
    d.index = pd.to_datetime(d.index, utc=True)

    for c in ("o", "h", "l", "c", "v"):
        d[c] = pd.to_numeric(d[c], errors="raise")

    if d.index.has_duplicates:
        columns = [
            c for c in ("o", "h", "l", "c", "v", "tb", "end")
            if c in d
        ]

        different = d.groupby(level=0)[columns].nunique(
            dropna=False
        )

        if (different > 1).any().any():
            bad = different.index[
                (different > 1).any(axis=1)
            ]

            raise ValueError(
                "conflicting duplicate candles: "
                f"{len(bad)} timestamps; first={bad[0]}"
            )

        count = int(d.index.duplicated().sum())

        LOG.warning(
            "Removed %d identical duplicate candle rows",
            count,
        )

        d = d[~d.index.duplicated(keep="first")]

    return d.sort_index()


def download(start, end):
    import yfinance as yf
    from scanner_data import clean

    symbol = os.environ.get("GC_SYMBOL", "GC=F")
    end = utc(end)

    start = max(
        utc(start),
        end - pd.Timedelta(days=58),
    )

    d = yf.download(
        symbol,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        interval="15m",
        auto_adjust=False,
        progress=False,
        threads=False,
        prepost=True,
        timeout=25,
    )

    if d is None or d.empty:
        raise ValueError(
            "GC futures: Yahoo returned no candles"
        )

    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

    d = d.rename(columns={
        "Open": "o",
        "High": "h",
        "Low": "l",
        "Close": "c",
        "Volume": "v",
    })

    if "v" not in d:
        raise ValueError(
            "GC futures: volume field missing"
        )

    d = distinct(d[["o", "h", "l", "c", "v"]])
    d = clean(d, "15m", end)

    if float(d["v"].sum()) <= 0:
        raise ValueError(
            "GC futures: no positive reported volume"
        )

    d.attrs["symbol"] = symbol
    return d


def live_history(now):
    global _CACHE, _CACHE_TIME

    if (
        _CACHE_TIME is None
        or now - _CACHE_TIME >= MINUTE
    ):
        _CACHE_TIME = now
        _CACHE = download(
            now - pd.Timedelta(days=7),
            now,
        )

    return _CACHE


def profile(data, bins=64, fraction=0.70, tick=0.1):
    if (
        not 0 < fraction <= 1
        or bins < 2
        or tick <= 0
    ):
        raise ValueError("invalid profile settings")

    d = distinct(data)

    x = d[["o", "h", "l", "c", "v"]].to_numpy(
        dtype=float
    )

    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError(
            "empty/nonfinite profile input"
        )

    if (
        (d["v"] < 0).any()
        or (d["l"] > d["h"]).any()
        or (d[["o", "h", "l", "c"]] <= 0).any().any()
    ):
        raise ValueError("invalid profile values")

    d = d[d["v"] > 0]

    if d.empty:
        raise ValueError(
            "volume profile unavailable: volume is zero"
        )

    low = float(d["l"].min())
    high = float(d["h"].max())

    step = max(
        tick,
        math.ceil(
            max(high - low, tick) / bins / tick
        ) * tick,
    )

    left = math.floor(low / step) * step
    n = max(2, math.ceil((high - left) / step) + 1)

    edges = left + np.arange(n + 1) * step
    volumes = np.zeros(n)

    for _, row in d.iterrows():
        lo = float(row["l"])
        hi = float(row["h"])
        volume = float(row["v"])

        if hi == lo:
            index = int(np.clip(
                np.searchsorted(
                    edges,
                    float(row["c"]),
                    side="right",
                ) - 1,
                0,
                n - 1,
            ))

            volumes[index] += volume

        else:
            overlap = np.maximum(
                0,
                np.minimum(edges[1:], hi)
                - np.maximum(edges[:-1], lo),
            )

            if overlap.sum() <= 0:
                raise ValueError(
                    "profile range allocation failed"
                )

            volumes += (
                volume * overlap / overlap.sum()
            )

    poc = int(volumes.argmax())
    lower = upper = poc

    accumulated = float(volumes[poc])
    target = fraction * float(volumes.sum())

    while (
        accumulated + 1e-8 < target
        and (lower > 0 or upper < n - 1)
    ):
        up = (
            float(volumes[upper + 1])
            if upper < n - 1
            else -1
        )

        down = (
            float(volumes[lower - 1])
            if lower > 0
            else -1
        )

        if up >= down:
            upper += 1
            accumulated += up
        else:
            lower -= 1
            accumulated += down

    return {
        "poc": float(
            (edges[poc] + edges[poc + 1]) / 2
        ),
        "val": float(edges[lower]),
        "vah": float(edges[upper + 1]),
        "total_volume": float(volumes.sum()),
        "value_fraction": (
            accumulated / float(volumes.sum())
        ),
        "bin_width": step,
        "histogram": volumes.tolist(),
        "edges": edges.tolist(),
        "method": (
            "reported OHLCV / approximate price allocation"
        ),
    }


def session_labels(index):
    local = (
        index.tz_convert("America/New_York")
        .tz_localize(None)
    )

    return (
        local + pd.Timedelta(hours=6)
    ).normalize()


def context(data, decision):
    decision = utc(decision)

    lag = pd.Timedelta(minutes=float(
        os.environ.get("GC_DELAY_MINUTES", "20")
    ))

    if lag < MINUTE:
        raise ValueError(
            "GC_DELAY_MINUTES must be >=15 "
            "for delayed Yahoo context"
        )

    cutoff = decision - lag
    d = distinct(data)

    if "end" not in d:
        d["end"] = d.index + MINUTE

    d = d[d["end"] <= cutoff]

    if len(d) < 60:
        raise ValueError(
            "insufficient GC history "
            "before availability cutoff"
        )

    last_end = utc(d["end"].iloc[-1])

    age = (
        decision - last_end
    ).total_seconds() / 60

    if age > 75:
        raise ValueError(
            f"GC snapshot stale: {age:.0f} minutes old"
        )

    labels = session_labels(d.index)
    chosen = None

    for label in reversed(sorted(set(labels))):
        close = (
            pd.Timestamp(label)
            + pd.Timedelta(hours=17)
        ).tz_localize(
            "America/New_York"
        ).tz_convert("UTC")

        opened = (
            pd.Timestamp(label)
            - pd.Timedelta(days=1)
            + pd.Timedelta(hours=18)
        ).tz_localize(
            "America/New_York"
        ).tz_convert("UTC")

        if close > cutoff:
            continue

        group = d[
            (labels == label)
            & (d.index >= opened)
            & (d["end"] <= close)
        ]

        if (
            len(group) < 88
            or group.index[0] > opened + MINUTE
            or utc(group["end"].iloc[-1]) < close - MINUTE
        ):
            continue

        gaps = (
            group.index.to_series().diff().dropna()
            > pd.Timedelta(minutes=30)
        )

        if gaps.any():
            continue

        if decision - close > pd.Timedelta(days=5):
            continue

        chosen = (group, close)
        break

    if chosen is None:
        raise ValueError(
            "no completed GC session "
            "with sufficient coverage"
        )

    group, session_end = chosen
    result = profile(group)

    price = float(d["c"].iloc[-1])
    last = d.iloc[-1]

    if (
        float(last["l"]) <= result["val"]
        and price > result["val"]
    ):
        bias = 1
        reason = "رفض أسفل VAL في العقود"

    elif (
        float(last["h"]) >= result["vah"]
        and price < result["vah"]
    ):
        bias = -1
        reason = "رفض أعلى VAH في العقود"

    elif price > result["poc"]:
        bias = 1
        reason = "إغلاق العقود فوق POC"

    elif price < result["poc"]:
        bias = -1
        reason = "إغلاق العقود تحت POC"

    else:
        bias = 0
        reason = "إغلاق العقود عند POC"

    result.update(
        bias=bias,
        reason=reason,
        last_price=price,
        last_close=last_end.isoformat(),
        session_end=session_end.isoformat(),
        age_minutes=round(age, 1),
        symbol=data.attrs.get(
            "symbol",
            os.environ.get("GC_SYMBOL", "GC=F"),
        ),
    )

    return result


def analyze_gold(fr, strategy):
    result = strategy["zone_strat"]("GOLD", fr)
    result.setdefault("miss", [])

    result["style"] = (
        result.get("style") or "GoldZones+GC"
    )

    mode = os.environ.get(
        "GOLD_VOLUME_MODE", "confirm"
    ).lower()

    if mode not in ("confirm", "context", "off"):
        raise ValueError(
            "GOLD_VOLUME_MODE must be "
            "confirm, context or off"
        )

    now = utc(fr.get(
        "_decision_at",
        pd.Timestamp.now(tz="UTC"),
    ))

    valid = None
    problem = None

    if mode != "off":
        try:
            history = fr.get("_gc15")

            if history is None:
                if fr.get("_backtest"):
                    raise ValueError(
                        "historical GC snapshot not supplied; "
                        "no live fetch allowed"
                    )

                history = live_history(now)

            valid = context(history, now)

        except Exception as exc:
            from scanner_data import safe_error
            problem = safe_error(exc)

    if valid:
        fmt = strategy["fmt"]
        result["gc_profile"] = valid

        result["obs"] += (
            "\nGC futures فقط، حجم مبلغ من Yahoo "
            "وبروفايل OHLCV تقريبي: "
            f"POC {fmt(valid['poc'])} | "
            f"VAH {fmt(valid['vah'])} | "
            f"VAL {fmt(valid['val'])}"
            f"\nمصدر الحجم yahoo:{valid['symbol']} | "
            f"حجم الجلسة {valid['total_volume']:,.0f} عقد"
            f"\nجلسة البروفايل انتهت {valid['session_end']} | "
            f"آخر إغلاق GC {valid['last_close']}"
            f"\nعمر اللقطة {valid['age_minutes']} دقيقة | "
            f"{valid['reason']}"
            "\nمستويات العقود لا تُستعمل كأسعار دخول "
            "أو أهداف XAU/USD. لا يوجد Order Flow "
            "حقيقي من هذه الشموع."
        )

    elif mode != "off":
        result["obs"] += (
            "\nحجم GC غير متاح/غير صالح: "
            + str(problem)
        )

        result["miss"].append(
            "حجم عقود GC غير متاح أو قديم"
        )

    else:
        result["obs"] += (
            "\nفلتر حجم GC معطل بإعداد المستخدم؛ "
            "تحليل مناطق السعر فقط."
        )

    blocked = (
        mode == "confirm"
        and (
            valid is None
            or valid["bias"] != result["side"]
        )
    )

    if (
        valid
        and mode == "confirm"
        and valid["bias"] != result["side"]
    ):
        result["miss"].append(
            "سياق حجم GC لا يؤيد اتجاه الإعداد"
        )

    if blocked and result.get("tier") == "A":
        stop = result["plan"]["stop"]

        result.update(
            tier="B",
            cond=(
                "بقاء تأكيد المنطقة مع توفر حجم GC "
                "حديث وسياق يؤيد الاتجاه"
            ),
            inv=(
                "كسر "
                if result["side"] == 1
                else "اختراق "
            ) + strategy["fmt"](stop),
        )

    result["obs"] += (
        "\nقاعدة الذهب: مناطق XAU/USD مع سياق حجم GC؛ "
        "قواعد تجريبية غير مثبتة الربحية."
    )

    LOG.info(
        "GOLD_VOLUME mode=%s valid=%s reason=%s",
        mode,
        valid is not None,
        problem or (valid or {}).get("reason"),
    )

    return result