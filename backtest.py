"""Historical paper replay using scanner.py.
Does not read or write the live state.json.
"""
import csv
import json
import logging
import os
from pathlib import Path
import sys
import time
import types

import pandas as pd

import candle_archive
import scanner as engine
import scanner_data as md
import gold_volume as gv
from upgrade_agent import patch_source

LOG = logging.getLogger(__name__)
OUT = Path("backtest_results")
DAYS = int(os.environ.get("BT_DAYS", "90"))
STEP = int(os.environ.get("BT_STEP", "1"))
SCAN_DELAY = pd.Timedelta(
    seconds=float(os.environ.get("BT_SCAN_DELAY_SECONDS", "120"))
)
HOSTS = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
)


def load_strategy():
    for key in ("TG_TOKEN", "TG_CHAT", "GEMINI_KEY"):
        os.environ.setdefault(key, "")

    module = types.ModuleType("backtest_strategy")
    module.__file__ = "agent.py"
    sys.modules[module.__name__] = module

    source = patch_source(
        Path("agent.py").read_text(encoding="utf-8")
    )
    exec(compile(source, "agent.py", "exec"), module.__dict__)
    return module


def binance_range(symbol, tf, start, end):
    rows = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000) - 1

    for _ in range(1000):
        if cursor > end_ms:
            break

        batch = None

        for host in HOSTS:
            try:
                batch = md.get_json(
                    host + "/api/v3/klines",
                    {
                        "symbol": symbol,
                        "interval": tf,
                        "startTime": cursor,
                        "endTime": end_ms,
                        "limit": 1000,
                    },
                )

                if not isinstance(batch, list):
                    raise ValueError("non-list Binance payload")

                break

            except Exception as exc:
                LOG.warning(
                    "Historical Binance %s %s %s",
                    symbol, tf, md.safe_error(exc),
                )

        if batch is None:
            raise ValueError(
                "both Binance historical endpoints failed"
            )

        if not batch:
            break

        next_cursor = int(batch[-1][0]) + 1

        if next_cursor <= cursor:
            raise ValueError(
                "historical pagination did not advance"
            )

        rows.extend(batch)
        cursor = next_cursor

        if len(batch) < 1000:
            break

    if not rows:
        raise ValueError("empty Binance history")

    d = pd.DataFrame(rows).iloc[
        :, [0, 1, 2, 3, 4, 5, 6, 9]
    ]
    d.columns = ["t", "o", "h", "l", "c", "v", "end", "tb"]
    d.index = pd.to_datetime(
        d.pop("t").astype("int64"),
        unit="ms",
        utc=True,
    )
    d["end"] = pd.to_datetime(
        d["end"].astype("int64") + 1,
        unit="ms",
        utc=True,
    )
    d["tb"] = pd.to_numeric(d["tb"])
    d = gv.distinct(d)

    return md.clean(
        d, tf, end + pd.Timedelta(seconds=2), d["end"]
    )


def twelve_range(symbol, tf, start, end):
    key = os.environ.get("TWELVE_KEY")

    if not key:
        raise ValueError("TWELVE_KEY missing")

    parts = []
    cursor = end

    for _ in range(20):
        payload = md.get_json(
            "https://api.twelvedata.com/time_series",
            {
                "symbol": symbol,
                "interval": {
                    "15m": "15min",
                    "1h": "1h",
                    "4h": "4h",
                    "1d": "1day",
                }[tf],
                "outputsize": 5000,
                "timezone": "UTC",
                "apikey": key,
                "end_date": cursor.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            },
        )

        if not payload.get("values"):
            raise ValueError(
                "Twelve historical response missing values: "
                + str(payload.get("code", "unknown"))
            )

        d = md.twelve_payload(
            payload, tf, end + pd.Timedelta(seconds=2)
        )

        if d.empty:
            break

        oldest = d.index[0]

        if oldest >= cursor:
            raise ValueError(
                "Twelve pagination did not advance"
            )

        parts.append(d)

        if oldest <= start or len(payload["values"]) < 5000:
            break

        cursor = oldest - pd.Timedelta(seconds=1)
        time.sleep(8)

    if not parts:
        raise ValueError("empty Twelve history")

    d = gv.distinct(pd.concat(parts))
    return d[(d.index >= start) & (d["end"] <= end)]


def okx_range(symbol, tf, start, end):
    md.okx(symbol, tf, end)

    rows = []
    cursor = int(end.timestamp() * 1000)
    oldest_needed = int(start.timestamp() * 1000)

    for _ in range(1200):
        payload = md.get_json(
            "https://www.okx.com/api/v5/market/history-candles",
            {
                "instId": symbol,
                "bar": {
                    "15m": "15m",
                    "1h": "1H",
                    "4h": "4H",
                    "1d": "1Dutc",
                }[tf],
                "after": str(cursor),
                "limit": 100,
            },
        )

        if payload.get("code") != "0":
            raise ValueError(
                "OKX historical request rejected"
            )

        page = payload.get("data", [])

        if not page:
            break

        rows.extend(row for row in page if row[8] == "1")
        oldest = min(int(row[0]) for row in page)

        if oldest >= cursor:
            raise ValueError(
                "OKX pagination did not advance"
            )

        if oldest <= oldest_needed:
            break

        cursor = oldest
        time.sleep(0.12)

    if not rows:
        raise ValueError("empty OKX history")

    d = pd.DataFrame(rows).iloc[:, :6]
    d.columns = ["t", "o", "h", "l", "c", "v"]
    d.index = pd.to_datetime(
        d.pop("t").astype("int64"),
        unit="ms",
        utc=True,
    )
    d = gv.distinct(d)
    d = md.clean(d, tf, end + pd.Timedelta(seconds=2))

    return d[(d.index >= start) & (d["end"] <= end)]


def load_asset(key, start, end):
    provider, symbol = md.specs(key)[0]
    getter = {
        "binance": binance_range,
        "twelve": twelve_range,
        "okx": okx_range,
    }.get(provider)

    warm = {"15m": 7, "1h": 20, "4h": 70, "1d": 420}
    frames = {}

    for tf in ("15m", "1h", "4h", "1d"):
        if key in ("BTC", "ETH", "SOL") and tf == "1d":
            continue

        try:
            if provider == "yahoo":
                frames[tf] = md.yahoo(
                    symbol,
                    tf,
                    end,
                    start=start - pd.Timedelta(days=warm[tf]),
                )
            else:
                frames[tf] = getter(
                    symbol,
                    tf,
                    start - pd.Timedelta(days=warm[tf]),
                    end,
                )

            d = frames[tf]

            LOG.info(
                "BT_SOURCE asset=%s source=%s:%s tf=%s "
                "bars=%d first=%s last_end=%s",
                key, provider, symbol, tf,
                len(d), d.index[0], d["end"].iloc[-1],
            )

        except Exception as exc:
            if tf != "1d":
                raise

            LOG.warning(
                "%s daily context unavailable: %s",
                key, md.safe_error(exc),
            )

    frames["_source"] = provider + ":" + symbol
    frames["_provider"] = provider
    frames["_symbol"] = symbol
    return frames


def snapshot(frames, decision, gc=None):
    out = {
        key: frames[key][
            frames[key]["end"] <= decision
        ].tail(
            500 if frames["_provider"] == "binance" else 400
        )
        for key in ("15m", "1h", "4h", "1d")
        if key in frames
    }

    out.update({
        key: value
        for key, value in frames.items()
        if key.startswith("_")
    })

    out["_decision_at"] = decision
    out["_backtest"] = True

    if gc is not None:
        out["_gc15"] = gc

    return out


def empty_state():
    return {
        "n": 0,
        "active": {},
        "watch": {},
        "closed": [],
        "cool": {},
        "outbox": [],
        "event_ids": [],
    }


def run_asset(a, key, frames, btc, start, end, gc=None):
    st = empty_state()

    metrics = {
        "analyses": 0,
        "analysis_errors": 0,
        "missing_context": 0,
        "cost_rejections": 0,
        "first_decision": None,
        "last_decision": None,
        "gc_available_decisions": 0,
        "first_gc_decision": None,
        "last_gc_decision": None,
        "open_at_end": None,
        "source": frames["_source"],
    }

    decisions = [
        engine.stamp(t) + SCAN_DELAY
        for t in frames["15m"]["end"]
        if start <= engine.stamp(t) + SCAN_DELAY <= end
    ]

    if not decisions:
        raise ValueError(
            "no candles in requested test period"
        )

    for index, decision in enumerate(decisions):
        current = snapshot(frames, decision, gc)
        position = st["active"].get(key)

        if position:
            engine.manage(
                st, position, current, decision, a
            )

            if position["state"] in (
                "CLOSED", "INVALIDATED", "EXPIRED"
            ):
                del st["active"][key]

            st["outbox"].clear()
            continue

        if index % STEP:
            continue

        fresh, why = md.freshness(
            current, key, decision
        )

        if not fresh:
            metrics["missing_context"] += 1
            continue

        if (
            st["cool"].get(key)
            and decision - engine.stamp(st["cool"][key])
            < pd.Timedelta(hours=2)
        ):
            continue

        btc_direction = None

        if btc is not None and key in ("ETH", "SOL"):
            b = snapshot(btc, decision)

            if md.freshness(b, "BTC", decision)[0]:
                t4 = a.trend(b["4h"])
                t1 = a.trend(b["1h"])
                btc_direction = t4 if t4 == t1 else 0

        try:
            result = a.analyze(
                key, current, btc_direction
            )
            metrics["analyses"] += 1

            if metrics["first_decision"] is None:
                metrics["first_decision"] = decision.isoformat()

            metrics["last_decision"] = decision.isoformat()

            if (
                result.get("gc_profile")
                or result.get("gc_volume")
            ):
                metrics["gc_available_decisions"] += 1

                if metrics["first_gc_decision"] is None:
                    metrics["first_gc_decision"] = (
                        decision.isoformat()
                    )

                metrics["last_gc_decision"] = (
                    decision.isoformat()
                )

            if result["tier"] in ("A", "ARMED"):
                position = engine.make_setup(
                    st, result, current, decision
                )

                if (
                    position["cost_bps"] < 0
                    or not pd.notna(position["cost_bps"])
                ):
                    raise ValueError(
                        "invalid assumed execution cost"
                    )

                if engine.cost_r(position) > 0.35:
                    metrics["cost_rejections"] += 1
                else:
                    st["active"][key] = position

        except Exception as exc:
            metrics["analysis_errors"] += 1

            if metrics["analysis_errors"] <= 3:
                LOG.error(
                    "BT_ANALYSIS_ERROR %s %s %s",
                    key, decision, md.safe_error(exc),
                )

        st["outbox"].clear()

    if key in st["active"]:
        metrics["open_at_end"] = st["active"][key]

    return st["closed"], metrics


def summary(trades):
    ordered = sorted(
        trades, key=lambda trade: trade["t_close"]
    )
    results = [float(trade["r"]) for trade in ordered]
    gross_profit = sum(r for r in results if r > 0)
    gross_loss = -sum(r for r in results if r < 0)
    equity = peak = drawdown = 0.0

    for result in results:
        equity += result
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)

    return {
        "trades": len(results),
        "win_pct": (
            100 * sum(r > 0 for r in results) / len(results)
            if results else None
        ),
        "gross_R": sum(
            float(trade.get("gross_r", trade["r"]))
            for trade in ordered
        ),
        "assumed_cost_R": sum(
            float(trade.get("gross_r", trade["r"]))
            - float(trade["r"])
            for trade in ordered
        ),
        "net_R": sum(results),
        "mean_R": (
            sum(results) / len(results)
            if results else None
        ),
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss
            else "infinity" if gross_profit else None
        ),
        "max_closed_trade_drawdown_R": drawdown,
        "ambiguous_trades": sum(
            bool(trade.get("ambiguous"))
            for trade in ordered
        ),
    }


def line(trades):
    stats = summary(trades)

    if not stats["trades"]:
        return "0 صفقة مغلقة"

    profit_factor = stats["profit_factor"]

    profit_factor = (
        f"{profit_factor:.2f}"
        if isinstance(profit_factor, (int, float))
        else str(profit_factor)
    )

    return (
        f"{stats['trades']} صفقة | "
        f"نجاح {stats['win_pct']:.0f}% | "
        f"صافي {stats['net_R']:+.2f}R | "
        f"متوسط {stats['mean_R']:+.2f}R | "
        f"PF {profit_factor} | "
        f"هبوط مغلق "
        f"{stats['max_closed_trade_drawdown_R']:.2f}R | "
        f"ملتبسة {stats['ambiguous_trades']}"
    )


def report(mode, results, start, end):
    all_trades = [
        trade
        for value in results.values()
        for trade in value["trades"]
    ]
    cut = start + (end - start) * 2 / 3

    first = [
        trade for trade in all_trades
        if engine.stamp(trade["t_decl"]) < cut
    ]
    last = [
        trade for trade in all_trades
        if engine.stamp(trade["t_decl"]) >= cut
    ]

    lines = [
        f"🧪 باكتيست | ICT للعملات {mode} | "
        "ذهب وناسداك: ترند + مناطق + CISD",
        f"الفترة المطلوبة UTC: "
        f"{start.isoformat()} → {end.isoformat()}",
        "الكل: " + line(all_trades),
        "أول ثلثين حسب إعلان الإعداد: " + line(first),
        "آخر ثلث حسب إعلان الإعداد: " + line(last),
    ]

    for key, value in results.items():
        coverage = value["coverage"]

        lines += [
            key + ": " + line(value["trades"]),
            f"  المصدر {coverage['source']} | "
            f"قرار أول {coverage['first_decision']} | "
            f"أخير {coverage['last_decision']}",
            f"  تحليلات {coverage['analyses']} | "
            f"أخطاء {coverage['analysis_errors']} | "
            f"سياق ناقص {coverage['missing_context']} | "
            f"لقطات GC صالحة "
            f"{coverage['gc_available_decisions']} | "
            f"مفتوحة بالنهاية "
            f"{bool(coverage['open_at_end'])}",
            f"  تغطية GC الفعلية: "
            f"{coverage['first_gc_decision']} → "
            f"{coverage['last_gc_decision']}",
        ]

    totals = summary(all_trades)

    costs = ", ".join(
        f"{key}={engine.assumed_cost_bps(key):g}"
        for key in results
    )

    lines += [
        f"قبل الكلفة {totals['gross_R']:+.2f}R | "
        f"الكلفة المفترضة {totals['assumed_cost_R']:.2f}R",
        "كلفة كلية مفترضة ذهاب/عودة (bps): " + costs,
        "الذهب وناسداك: ترند + مناطق + CISD؛ "
        "الفوليوم العادي للتأييد فقط، بدون بروفايل. "
        "كل هدف ثلث الحجم.",
        f"تأخير الفحص المفترض بعد إغلاق الشمعة: "
        f"{SCAN_DELAY.total_seconds():g} ثانية؛ "
        "تأخير GitHub الحقيقي متغير.",
        "أخبار الماضي غير معاد تشغيلها. "
        "تأخير المصدر التاريخي غير معلوم بالكامل؛ "
        "GC عليه تأخير مفترض محافظ.",
        "تغطية Yahoo أقصر من 90 يوم؛ "
        "راجع التواريخ الفعلية. "
        "الانقسام الزمني وصفي وليس إثبات اختبار خارج العينة.",
        "الهبوط من صفقات مغلقة، ولا يشمل التذبذب العائم. "
        "النتائج لا تثبت تفوقاً أو تضمن المستقبل.",
    ]

    return "\n".join(lines)


def save_results(mode, results, text, start, end):
    OUT.mkdir(exist_ok=True)

    trades = [
        trade
        for value in results.values()
        for trade in value["trades"]
    ]

    payload = {
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "summary": summary(trades),
        "assets": results,
        "assumptions": {
            "roundtrip_cost_bps": os.environ.get(
                "ROUNDTRIP_COST_BPS", "14"
            ),
            "cost_bps_by_asset": {
                key: engine.assumed_cost_bps(key)
                for key in results
            },
            "gc_delay_minutes": os.environ.get(
                "GC_DELAY_MINUTES", "20"
            ),
            "gold_volume_mode": "ordinary_volume_context_only",
            "gold_and_ndx_strategy": "simple_zones_v1",
            "news_replayed": False,
            "scan_every_n_15m_bars": STEP,
            "scan_delay_seconds": SCAN_DELAY.total_seconds(),
        },
    }

    (OUT / f"results_{mode}.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    (OUT / f"report_{mode}.txt").write_text(
        text, encoding="utf-8"
    )

    with (OUT / f"trades_{mode}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "id", "key", "side", "style",
            "entry", "stop0", "t_decl",
            "t0", "t_close", "exit", "r",
            "gross_r", "ambiguous", "source",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(trades)


def send_report(text):
    st = empty_state()
    engine.event(
        st, text, "backtest-" + str(int(time.time()))
    )

    if not engine.flush(
        st,
        lambda _: None,
        os.environ["TG_TOKEN"],
        os.environ["TG_CHAT"],
    ):
        raise RuntimeError(
            "Telegram report failed; "
            "full results retained as artifacts"
        )


def main():
    if (
        DAYS < 1
        or STEP < 1
        or SCAN_DELAY < pd.Timedelta(0)
    ):
        raise ValueError(
            "BT_DAYS and BT_STEP must be positive"
        )

    strategy = load_strategy()
    replay = os.environ.get("BT_INPUT_DIR")

    if replay:
        data, gc, start, end, failures = candle_archive.load(
            Path(replay)
        )

    else:
        end = pd.Timestamp.now(tz="UTC").floor("15min")
        start = end - pd.Timedelta(days=DAYS)
        data = {}
        failures = {}

        for key in strategy.ASSETS:
            try:
                data[key] = load_asset(key, start, end)
            except Exception as exc:
                failures[key] = md.safe_error(exc)
                LOG.error(
                    "BT_LOAD_ERROR %s %s",
                    key, failures[key],
                )

        gc = None

        if "GOLD" in data:
            try:
                gc = gv.download(
                    start - pd.Timedelta(days=7), end
                )
            except Exception as exc:
                LOG.error(
                    "BT_GC_ERROR %s", md.safe_error(exc)
                )

    if (
        os.environ.get("BT_SAVE_CANDLES", "true").lower()
        == "true"
    ):
        candle_archive.save(
            OUT / "input_candles",
            data, gc, start, end, failures,
        )

    OUT.mkdir(exist_ok=True)

    (OUT / "load_errors.json").write_text(
        json.dumps(
            failures, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    if not data:
        raise RuntimeError(
            "no assets loaded; no backtest result claimed"
        )

    modes = (
        [True, False]
        if os.environ.get("COMPARE_ICT", "false").lower() == "true"
        else [True]
    )
    had_errors = bool(failures)

    for use_ict in modes:
        strategy.USE_ICT = use_ict
        results = {}

        for key, frames in data.items():
            trades, coverage = run_asset(
                strategy,
                key,
                frames,
                data.get("BTC"),
                start,
                end,
                gc if key == "GOLD" else None,
            )

            had_errors |= coverage["analysis_errors"] > 0

            results[key] = {
                "trades": trades,
                "coverage": coverage,
                "summary": summary(trades),
            }

        name = "on" if use_ict else "off"
        text = report(name, results, start, end)

        if failures:
            text += (
                "\nأصول لم تختبر بسبب فشل التحميل: "
                + ", ".join(failures)
            )

        print(text)
        save_results(name, results, text, start, end)

        if (
            os.environ.get("BT_SEND_TELEGRAM", "false").lower()
            == "true"
        ):
            brief = (
                "\n".join(text.splitlines()[:6])
                + "\nالتفاصيل والتغطية والأخطاء في ملفات "
                "backtest-results داخل Actions."
            )
            send_report(brief)

    if had_errors:
        raise RuntimeError(
            "partial/invalid run: "
            "see retained load and analysis errors"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()