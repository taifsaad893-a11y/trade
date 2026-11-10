"""ICT Silver Bullet 5-minute paper alerts: GOLD and NDX only.

Rule-based approximation of the 2023 Michael Huddleston lesson:
NY time window + 3-candle FVG + nearby opposing liquidity. A sweep and
MSS are context, not mandatory conditions. No live trade execution.
"""
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

LOG = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
STATE = Path("state.json")
MARKETS = {"GOLD": "الذهب", "NDX": "ناسداك"}
SESSIONS = {3: "London", 10: "New York AM", 14: "New York PM"}
FIVE = pd.Timedelta(minutes=5)
VERSION = "ict_silver_bullet_5m_v1"
MIN_RR = 1.5  # Reject nearby liquidity targets that do not justify the stop.


def utc(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def load():
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        st = {}
    if not isinstance(st, dict):
        raise ValueError("state.json must contain an object")
    sb = st.setdefault("silver_bullet", {})
    for key, default in (("active", {}), ("used", []), ("closed", []),
                         ("outbox", []), ("event_ids", [])):
        sb.setdefault(key, default)
    sb.setdefault("n", 0)
    return st, sb


def save(st):
    fd, tmp = tempfile.mkstemp(prefix=".sb-state-", dir=str(STATE.resolve().parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def emit(sb, identity, message):
    if identity in sb["event_ids"]:
        return
    sb["event_ids"].append(identity)
    sb["event_ids"] = sb["event_ids"][-1500:]
    sb["outbox"].append({"id": identity, "text": message + "\nID: " + identity})


def flush(st, sb):
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if not token or not chat:
        raise RuntimeError("Missing TG_TOKEN or TG_CHAT")
    save(st)
    while sb["outbox"]:
        item = sb["outbox"][0]
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": item["text"][:3900]},
                timeout=(8, 25))
            if not r.ok or r.json().get("ok") is not True:
                raise RuntimeError(f"telegram http {r.status_code}")
        except Exception as exc:
            LOG.error("Telegram delivery deferred: %s", type(exc).__name__)
            return False
        sb["outbox"].pop(0)
        save(st)
    return True


def clean(data, now):
    d = data.copy()
    d.index = pd.to_datetime(d.index, utc=True)
    d = d.sort_index()
    d = d[~d.index.duplicated(keep="last")]
    for c in ("o", "h", "l", "c"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["o", "h", "l", "c"])
    d = d[d.index + FIVE <= now - pd.Timedelta(seconds=5)]
    if len(d) < 40:
        raise ValueError("insufficient completed 5m candles")
    if ((d[["o", "h", "l", "c"]] <= 0).any().any() or
        (d["h"] < d[["o", "l", "c"]].max(axis=1)).any() or
        (d["l"] > d[["o", "h", "c"]].min(axis=1)).any()):
        raise ValueError("invalid OHLC values")
    return d


def yahoo(symbol, now):
    raw = yf.download(symbol, period="5d", interval="5m", progress=False,
                      auto_adjust=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("no Yahoo data for " + symbol)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    d = raw.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c"})
    return clean(d, now)


def twelve(symbol, now):
    key = os.environ.get("TWELVE_KEY", "")
    if not key:
        raise RuntimeError("TWELVE_KEY unavailable")
    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol": symbol, "interval": "5min", "outputsize": 900,
        "timezone": "UTC", "apikey": key}, timeout=(8, 25))
    if not r.ok:
        raise RuntimeError(f"Twelve Data HTTP {r.status_code}")
    j = r.json()
    if "values" not in j:
        raise RuntimeError("Twelve Data 5min unavailable: " + str(j.get("code", "unknown")))
    d = pd.DataFrame(j["values"])
    d.index = pd.to_datetime(d["datetime"], utc=True)
    d = d.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c"})
    return clean(d, now)


def prices(key, now):
    if key == "NDX":
        sym = os.environ.get("NDX_SYMBOL", "NQ=F")
        return yahoo(sym, now), "Yahoo:" + sym
    sym = os.environ.get("GOLD_SYMBOL", "XAU/USD")
    try:
        return twelve(sym, now), "Twelve:" + sym
    except Exception as exc:
        LOG.warning("GOLD 5m Twelve unavailable; trying GC=F proxy: %s", type(exc).__name__)
        return yahoo("GC=F", now), "Yahoo:GC=F (futures proxy)"


def session_of(bar_open):
    n = utc(bar_open).tz_convert(NY)
    if n.weekday() >= 5 or n.hour not in SESSIONS:
        return None
    start = n.replace(minute=0, second=0, microsecond=0)
    end = start + pd.Timedelta(hours=1)
    if n + FIVE >= end:  # last bar closes exactly at end; no new pending entry
        return None
    return {"start": start, "end": end,
            "id": f"{n.date()}:{n.hour:02d}", "name": SESSIONS[n.hour]}


def levels(d, bar, ses):
    past = d[d.index < bar]
    if len(past) < 24:
        return [], []
    ny = past.index.tz_convert(NY)
    today = utc(bar).tz_convert(NY).date()
    old_days = [x for x in sorted(set(ny.date)) if x < today]
    hi, lo = [], []
    if old_days:
        prev = past[ny.date == old_days[-1]]
        hi.append(float(prev["h"].max()))
        lo.append(float(prev["l"].min()))
    current_week = utc(bar).tz_convert(NY).isocalendar()[:2]
    weeks = [x.isocalendar()[:2] for x in ny]
    prior = sorted(set(w for w in weeks if w < current_week))
    if prior:
        wk = past[[w == prior[-1] for w in weeks]]
        hi.append(float(wk["h"].max()))
        lo.append(float(wk["l"].min()))
    start = ses["start"].tz_convert("UTC")
    h = d[(d.index < start) & (d.index >= start - pd.Timedelta(hours=1))]
    if len(h) >= 8:
        hi.append(float(h["h"].max()))
        lo.append(float(h["l"].min()))
    # Only confirmed pivots: two candles on each side, no future information.
    if len(past) >= 5:
        highs, lows = past["h"].to_numpy(), past["l"].to_numpy()
        for i in range(2, len(past) - 2):
            if highs[i] > max(highs[i - 2:i]) and highs[i] >= max(highs[i + 1:i + 3]):
                hi.append(float(highs[i]))
            if lows[i] < min(lows[i - 2:i]) and lows[i] <= min(lows[i + 1:i + 3]):
                lo.append(float(lows[i]))
    return hi, lo


def setup(key, d, source, now):
    bar = d.index[-1]
    ses = session_of(bar)
    if not ses or now - (bar + FIVE) > pd.Timedelta(minutes=8):
        return None
    if any((d.index[-1] - d.index[-i]) != FIVE * (i - 1) for i in (2, 3)):
        return None
    a, b, c = d.iloc[-3], d.iloc[-2], d.iloc[-1]
    side = 1 if float(c["l"]) > float(a["h"]) else (-1 if float(c["h"]) < float(a["l"]) else 0)
    if not side:
        return None
    bot, top = ((float(a["h"]), float(c["l"])) if side == 1
                else (float(c["h"]), float(a["l"])))
    entry = (top + bot) / 2  # consequent encroachment
    recent = d.iloc[-8:]
    tick = 0.01 if key == "GOLD" else 0.25
    stop = (float(recent["l"].min()) - 2 * tick if side == 1
            else float(recent["h"].max()) + 2 * tick)
    highs, lows = levels(d, bar, ses)
    pool = sorted(set(highs if side == 1 else lows))
    targets = [x for x in pool if (x - entry) * side > 0]
    if not targets:
        return None
    target = min(targets) if side == 1 else max(targets)
    risk = (entry - stop) * side
    reward = (target - entry) * side
    if not all(math.isfinite(x) for x in (entry, stop, target)) or risk < 2*tick or reward <= 0:
        return None
    # Keep the nearest opposing liquidity target. Do not skip it for a farther
    # level just to manufacture a better R:R; reject the setup instead.
    if reward < MIN_RR * risk:
        LOG.info("REJECT LOW RR | %s | reward=%.4f | risk=%.4f | rr=%.4f | min=%.2f",
                 key, reward, risk, reward / risk, MIN_RR)
        return None
    return {"key": key, "side": side, "entry": round(entry, 4),
            "stop": round(stop, 4), "target": round(target, 4),
            "gap": [round(bot, 4), round(top, 4)],
            "rr": round(reward/risk, 2), "source": source,
            "session": ses["id"], "session_name": ses["name"],
            "window_end": ses["end"].tz_convert("UTC").isoformat(),
            "created_close": (bar + FIVE).isoformat(), "state": "PENDING"}


def advance(sb, rec, d, source, now):
    if rec["source"] != source:
        LOG.warning("%s open paper setup source changed %s -> %s; no mixed-price tracking",
                    rec["key"], rec["source"], source)
        if rec["state"] == "PENDING" and now >= utc(rec["window_end"]):
            rec["state"] = "EXPIRED"
            emit(sb, rec["id"] + ":expired", f"⌛ {rec['id']} | {MARKETS[rec['key']]}\nانتهت الجلسة والمصدر تغيّر؛ لم يُحتسب دخول")
        return
    since = utc(rec.get("last_end", rec["created_close"]))
    end = utc(rec["window_end"])
    bars = d[d.index + FIVE > since]
    for ts, row in bars.iterrows():
        bar_end = ts + FIVE
        rec["last_end"] = bar_end.isoformat()
        hi, lo = float(row["h"]), float(row["l"])
        if rec["state"] == "PENDING":
            if bar_end > end:
                break
            if lo <= rec["entry"] <= hi:
                rec["state"] = "FILLED"
                rec["filled_at"] = bar_end.isoformat()
                emit(sb, rec["id"] + ":filled", f"📍 {rec['id']} | {MARKETS[rec['key']]}\nتم لمس الدخول الورقي: {rec['entry']:.2f}")
            else:
                continue
        if rec["state"] == "FILLED":
            stop_hit = lo <= rec["stop"] if rec["side"] == 1 else hi >= rec["stop"]
            tp_hit = hi >= rec["target"] if rec["side"] == 1 else lo <= rec["target"]
            if stop_hit or tp_hit:
                rec["state"] = "CLOSED"
                rec["exit"] = "STOP" if stop_hit else "TARGET"
                rec["r"] = -1.0 if stop_hit else rec["rr"]
                rec["closed_at"] = bar_end.isoformat()
                sb["closed"].append(dict(rec))
                result = "🛑 STOP" if stop_hit else "🎯 TARGET (liquidity)"
                emit(sb, rec["id"] + ":close", f"{result} | {rec['id']} | {MARKETS[rec['key']]}\nالنتيجة الورقية: {rec['r']:+.2f}R" + ("\nعند التباس مسار الشمعة، حُسب الوقف أولاً." if stop_hit and tp_hit else ""))
                return
    if rec["state"] == "PENDING" and now >= end:
        rec["state"] = "EXPIRED"
        emit(sb, rec["id"] + ":expired", f"⌛ {rec['id']} | {MARKETS[rec['key']]}\nانتهت ساعة Silver Bullet بدون دخول")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    st, sb = load()
    now = pd.Timestamp.now(tz="UTC")
    LOG.info("SCAN START | %s", now.isoformat())
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if not token or not chat:
        raise RuntimeError("Missing TG_TOKEN or TG_CHAT")
    for method, params in (("getMe", {}), ("getChat", {"chat_id": chat})):
        try:
            response = requests.post(f"https://api.telegram.org/bot{token}/{method}",
                                     json=params, timeout=(8, 25))
            if not response.ok or response.json().get("ok") is not True:
                raise RuntimeError(f"Telegram {method} HTTP {response.status_code}")
        except Exception:
            LOG.error("TELEGRAM CHECK FAILED | %s", method)
            return 1
    LOG.info("TELEGRAM CHECK OK | bot and destination accessible; no test message sent")
    checked = 0
    for key in MARKETS:
        try:
            d, source = prices(key, now)
            age = now - (d.index[-1] + FIVE)
            if age > pd.Timedelta(minutes=9):
                LOG.warning("%s source stale: %s age=%s", key, source, age)
                continue  # Never send a late setup or use stale candles for tracking.
            checked += 1
            LOG.info("DATA OK | %s | %s | last_close=%s | age=%s | bars=%d",
                     key, source, (d.index[-1] + FIVE).isoformat(), age, len(d))
            rec = sb["active"].get(key)
            if rec:
                advance(sb, rec, d, source, now)
                if rec["state"] in ("CLOSED", "EXPIRED"):
                    del sb["active"][key]
                else:
                    continue
            found = setup(key, d, source, now)
            if not found:
                LOG.info("NO NEW SETUP | %s", key)
                continue
            used = f"{key}:{found['session']}"
            if used in sb["used"]:
                continue
            sb["n"] += 1
            found["id"] = f"SB-{key}-{sb['n']:05d}"
            sb["active"][key] = found
            sb["used"].append(used)
            sb["used"] = sb["used"][-500:]
            side = "BUY LIMIT" if found["side"] == 1 else "SELL LIMIT"
            emit(sb, found["id"] + ":setup", (
                f"🟠 ICT SILVER BULLET 5m | {MARKETS[key]} {side}\n"
                f"جلسة: {found['session_name']} (New York)\n"
                f"دخول FVG 50%: {found['entry']:.2f}\n"
                f"ستوب: {found['stop']:.2f} | هدف سيولة: {found['target']:.2f}\n"
                f"R:R = 1:{found['rr']:.2f}\n"
                f"FVG: {found['gap'][0]:.2f} – {found['gap'][1]:.2f}\n"
                f"المصدر: {source}\n"
                "الحالة: انتظار لمس الدخول خلال نفس ساعة Silver Bullet\n"
                "تنبيه ورقي، مو أمر منفّذ"))
        except Exception as exc:
            LOG.error("%s skipped due to data/analysis error: %s", key, type(exc).__name__)
    sb["version"] = VERSION
    delivered = flush(st, sb)
    LOG.info("SCAN COMPLETE | markets_ok=%d/2 | pending_notifications=%d", checked, len(sb["outbox"]))
    return 0 if delivered and checked == len(MARKETS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
