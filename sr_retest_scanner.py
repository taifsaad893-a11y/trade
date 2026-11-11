"""Closed 1h/4h support/resistance, 15m breakout/retest paper scanner."""
import logging
import math
import os
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import silver_bullet_scanner as feed

LOG = logging.getLogger(__name__)
FIFTEEN = pd.Timedelta(minutes=15)
BAGHDAD = ZoneInfo("Asia/Baghdad")
VERSION = "sr_breakout_retest_15m_v1"


def aggregate(d, minutes, now):
    """Reject partial higher-timeframe bars, including session gaps."""
    grouped = d.resample(f"{minutes}min", origin="epoch", closed="left", label="left")
    result = grouped.agg({"o": "first", "h": "max", "l": "min", "c": "last"})
    count = grouped["c"].count()
    return result[(count == minutes // 5) &
                  (result.index + pd.Timedelta(minutes=minutes) <= now)].dropna()


def atr(d):
    prev = d["c"].shift()
    tr = pd.concat([d["h"]-d["l"], (d["h"]-prev).abs(),
                    (d["l"]-prev).abs()], axis=1).max(axis=1)
    return float(tr.tail(14).mean())


def levels(d, before):
    """Only fully closed HTF bars and pivots confirmed before the breakout."""
    out = []
    for minutes in (60, 240):
        h = aggregate(d, minutes, before).tail(100)
        if len(h) < 6:
            continue
        for i in range(2, len(h)-2):
            if h["h"].iloc[i] > h["h"].iloc[i-2:i].max() and h["h"].iloc[i] >= h["h"].iloc[i+1:i+3].max():
                out.append(float(h["h"].iloc[i]))
            if h["l"].iloc[i] < h["l"].iloc[i-2:i].min() and h["l"].iloc[i] <= h["l"].iloc[i+1:i+3].min():
                out.append(float(h["l"].iloc[i]))
        # Recent completed range boundaries are useful even without a pivot.
        out.extend([float(h["h"].tail(20).max()), float(h["l"].tail(20).min())])
    return sorted(set(out))


def setup(key, raw, source, now):
    d = aggregate(raw, 15, now)
    if len(d) < 30 or now-(d.index[-1]+FIFTEEN) > pd.Timedelta(minutes=8):
        return None
    a = atr(d.iloc[:-1])
    if not math.isfinite(a) or a <= 0:
        return None
    last = d.iloc[-1]
    entry = float(last["c"])
    for offset in range(2, min(10, len(d)-1)):
        pos = len(d)-offset
        b, prev = d.iloc[pos], d.iloc[pos-1]
        available = levels(raw, d.index[pos])
        for level in available:
            side = (1 if prev["c"] <= level and b["c"] > level+0.08*a
                    else -1 if prev["c"] >= level and b["c"] < level-0.08*a else 0)
            if not side:
                continue
            interim = d.iloc[pos+1:-1]
            if ((interim["c"]-level)*side < -0.1*a).any():
                continue
            touch = last["l"] <= level+0.15*a and last["h"] >= level-0.15*a
            confirm = (entry-level)*side > 0 and (entry-float(last["o"]))*side > 0
            if not touch or not confirm or abs(entry-level) > 0.75*a:
                continue
            segment = d.iloc[pos+1:]
            structural = (float(segment["l"].min())-0.15*a if side == 1
                          else float(segment["h"].max())+0.15*a)
            risk = max((entry-structural)*side, 0.8*a, entry*0.0005)
            stop = entry-side*risk
            targets = sorted((x for x in available if (x-entry)*side > 0),
                             key=lambda x: abs(x-entry))
            if not targets:
                continue
            target = targets[0]
            rr = abs(target-entry)/risk
            if rr < 1.5 or abs(target-entry) < a:
                continue
            h1 = aggregate(raw, 60, d.index[-1]+FIFTEEN)
            ema = float(h1["c"].ewm(span=50, adjust=False).mean().iloc[-1])
            return {"key": key, "source": source, "side": side,
                    "entry": entry, "stop": stop, "target": target, "rr": rr,
                    "level": level, "atr": a, "ema_aligned": (entry-ema)*side > 0,
                    "breakout": d.index[pos].isoformat(),
                    "created_close": (d.index[-1]+FIFTEEN).isoformat(),
                    "state": "FILLED", "version": VERSION}
    return None


def advance(sb, rec, d, source, now):
    if source != rec["source"]:
        return
    since = feed.utc(rec.get("last_end", rec["created_close"]))
    bars = d[d.index >= since]  # Entry is at the confirmation close, never before.
    for ts, row in bars.iterrows():
        rec["last_end"] = (ts+feed.FIVE).isoformat()
        stop = row["l"] <= rec["stop"] if rec["side"] == 1 else row["h"] >= rec["stop"]
        target = row["h"] >= rec["target"] if rec["side"] == 1 else row["l"] <= rec["target"]
        if stop or target:
            rec.update(state="CLOSED", exit="STOP" if stop else "TARGET",
                       r=-1.0 if stop else rec["rr"], closed_at=(ts+feed.FIVE).isoformat())
            sb["closed"].append(dict(rec))
            feed.emit(sb, rec["id"]+":close", (
                f"{'🛑 ستوب' if stop else '🎯 الهدف'} | {feed.MARKETS[rec['key']]}\n"
                f"{rec['id']} | نتيجة ورقية {rec['r']:+.2f}R\nالمصدر: {source}"
                + ("\nالهدف والستوب بنفس الشمعة: احتُسب الستوب أولاً." if stop and target else "")))
            return


def flush(st, sb):
    """Analyse first; persist and retry Telegram delivery without a blocking preflight."""
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    feed.save(st)
    if not token or not chat:
        LOG.error("Missing Telegram configuration; queued events preserved")
        return False
    while sb["outbox"]:
        item = sb["outbox"][0]
        delivered = False
        for attempt in range(3):
            try:
                r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                  json={"chat_id": chat, "text": item["text"][:3900]},
                                  timeout=(8, 25))
                if r.ok and r.json().get("ok") is True:
                    delivered = True
                    break
                if r.status_code in (400, 401, 403):
                    break
            except (requests.RequestException, ValueError):
                pass
            if attempt < 2:
                time.sleep(2**attempt)
        if not delivered:
            LOG.error("Telegram delivery deferred; queue preserved")
            return False
        sb["outbox"].pop(0)
        feed.save(st)
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    st, legacy = feed.load()
    sb = st.setdefault("sr_retest", {"active": {}, "used": [], "closed": [],
                                    "outbox": [], "event_ids": [], "n": 0})
    now = pd.Timestamp.now(tz="UTC")
    ok = 0
    for key in feed.MARKETS:
        old = legacy["active"].get(key)
        # Retire unfilled old-strategy orders, preserve and manage filled ones.
        if old and old["state"] == "PENDING":
            old["state"] = "EXPIRED"
            feed.emit(legacy, old["id"]+":strategy-change",
                      f"⌛ {old['id']} | أُلغي الانتظار عند تبديل الاستراتيجية، بدون احتساب دخول.")
            del legacy["active"][key]
            old = None
        rec = sb["active"].get(key)
        pinned = rec or old
        try:
            d, source = feed.prices(key, now, pinned["source"] if pinned else None)
            ok += 1
            LOG.info("DATA OK | %s | %s | last_close=%s", key, source, d.index[-1]+feed.FIVE)
            if old:
                feed.advance(legacy, old, d, source, now)
                if old["state"] == "CLOSED":
                    del legacy["active"][key]
                continue
            if rec:
                advance(sb, rec, d, source, now)
                if rec["state"] == "CLOSED":
                    del sb["active"][key]
                continue
            if now.tz_convert("America/New_York").weekday() >= 5:
                continue
            found = setup(key, d, source, now)
            if not found:
                LOG.info("NO NEW SETUP | %s | SR breakout/retest 15m", key)
                continue
            identity = f"{key}:{found['breakout']}:{found['side']}:{found['level']:.4f}"
            if identity in sb["used"]:
                continue
            sb["used"] = (sb["used"]+[identity])[-2000:]
            sb["n"] += 1
            found["id"] = f"SR-{key}-{sb['n']:05d}"
            sb["active"][key] = found
            side = "شراء LONG" if found["side"] == 1 else "بيع SHORT"
            stamp = feed.utc(found["created_close"]).tz_convert(BAGHDAD).strftime("%Y-%m-%d %H:%M")
            feed.emit(sb, found["id"]+":entry", (
                f"🟢 دخول ورقي | {feed.MARKETS[key]} | {side}\n"
                f"الاستراتيجية: كسر وريتيست دعم/مقاومة 1h و4h\n"
                f"تأكيد الدخول: 15m | {stamp} بغداد\n"
                f"الدخول المرجعي (إغلاق التأكيد): {found['entry']:.2f}\n"
                f"الستوب: {found['stop']:.2f}\nالهدف الواحد: {found['target']:.2f}\n"
                f"العائد/المخاطرة: {found['rr']:.2f}R\n"
                f"مستوى الريتيست: {found['level']:.2f}\n"
                f"EMA50 الساعة: {'مؤيد' if found['ema_aligned'] else 'غير مؤيد'} (تقييم فقط)\n"
                f"المصدر: {source}\n"
                "لا تطارد السعر إذا ابتعد عن الدخول المرجعي. تتبع ورقي فقط."))
            LOG.info("PAPER ENTRY | %s | %s", key, found["id"])
        except Exception as exc:
            LOG.error("SCAN ERROR | %s | %s | %s", key, type(exc).__name__, str(exc)[:160])
    sb["version"] = VERSION
    legacy_sent = flush(st, legacy)
    sent = flush(st, sb)
    LOG.info("SCAN COMPLETE | markets_ok=%s/2 | pending_notifications=%s", ok,
             len(sb["outbox"])+len(legacy["outbox"]))
    return 0 if ok == 2 and sent and legacy_sent else 1


if __name__ == "__main__":
    raise SystemExit(main())
