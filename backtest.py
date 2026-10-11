import os
import pandas as pd
import requests

DAYS = 90
STEP = 2
EVENTS = []  # اختياري، بتوقيت UTC مثل: "2026-09-16 18:00"
BEFORE = pd.Timedelta(minutes=30)
AFTER = pd.Timedelta(minutes=45)
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]

src = open("agent.py", encoding="utf-8").read()
src = src.rsplit("\nmain()", 1)[0]
A = {}
exec(compile(src, "agent.py", "exec"), A)
ASSETS = A["ASSETS"]


def bin_range(sym, tf, start, end):
    rows = []
    cur = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cur < end_ms:
        got = None
        for h in HOSTS:
            try:
                r = requests.get(
                    h + "/api/v3/klines",
                    params={"symbol": sym, "interval": tf,
                            "startTime": cur, "endTime": end_ms,
                            "limit": 1000}, timeout=30)
                if r.ok:
                    got = r.json()
                    break
            except Exception:
                pass
        if got is None:
            raise RuntimeError("binance failed " + sym)
        if not got:
            break
        rows += got
        cur = got[-1][0] + 1
        if len(got) < 1000:
            break
    d = pd.DataFrame(rows).iloc[:, :6]
    d.columns = ["t", "o", "h", "l", "c", "v"]
    d = d.astype(float)
    d.index = pd.to_datetime(d["t"], unit="ms", utc=True)
    return d.drop(columns="t").iloc[:-1]


def load_asset(kind, sym, bt_start, end):
    if kind == "bin":
        day = pd.Timedelta(days=1)
        return {
            "15m": bin_range(sym, "15m", bt_start - 4 * day, end),
            "1h": bin_range(sym, "1h", bt_start - 14 * day, end),
            "4h": bin_range(sym, "4h", bt_start - 55 * day, end),
        }
    f15 = A["yf_frame"](sym, "59d", "15m")
    f1h = A["yf_frame"](sym, "700d", "1h")
    f4h = f1h.resample("4h").agg(
        {"o": "first", "h": "max", "l": "min",
         "c": "last", "v": "sum"}).dropna().iloc[:-1]
    return {"15m": f15, "1h": f1h, "4h": f4h}


def slice_at(fr, T):
    d15 = fr["15m"].loc[:T].tail(300)
    i1 = fr["1h"].index
    d1 = fr["1h"][i1 <= T - pd.Timedelta(minutes=45)].tail(300)
    i4 = fr["4h"].index
    d4 = fr["4h"][i4 <= T - pd.Timedelta(minutes=225)].tail(300)
    return d15, d1, d4


def btc_dir_at(btc, T, cache):
    if btc is None:
        return None
    if T not in cache:
        d15, d1, d4 = slice_at(btc, T)
        if len(d1) < 60 or len(d4) < 60:
            cache[T] = None
        else:
            a = A["trend"](d4)
            b = A["trend"](d1)
            cache[T] = a if a == b else 0
    return cache[T]


def run_asset(key, fr, btc, cache, bt_start, blocked):
    st = {"n": 0, "closed": [], "cool": {}}
    sp = None
    idx = fr["15m"].index
    steps = [T for T in idx if T >= bt_start][::STEP]
    for T in steps:
        now = T + pd.Timedelta(minutes=15)
        d15, d1, d4 = slice_at(fr, T)
        if len(d15) < 120 or len(d1) < 60 or len(d4) < 60:
            continue
        price = float(d15["c"].iloc[-1])
        if sp:
            A["manage"](st, sp, d15, d1, price, now)
            if sp["state"] in ("CLOSED", "INVALIDATED", "EXPIRED"):
                sp = None
            continue
        cool = st["cool"].get(key)
        if cool and now - pd.Timestamp(cool) < pd.Timedelta(hours=2):
            continue
        if blocked(now):
            continue
        bd = None
        if key in ("ETH", "SOL"):
            bd = btc_dir_at(btc, T, cache)
        try:
            res = A["analyze"](key, {"15m": d15, "1h": d1, "4h": d4}, bd)
        except Exception:
            continue
        if res["tier"] in ("A", "ARMED"):
            sp = A["open_setup"](st, res, d15.index[-1])
    return st["closed"], sp is not None


def summ(trades):
    if not trades:
        return "0 صفقة"
    rs = [t["r"] for t in sorted(trades, key=lambda x: x["t_close"])]
    n = len(rs)
    wins = sum(1 for r in rs if r > 0)
    gp = sum(r for r in rs if r > 0)
    gl = -sum(r for r in rs if r <= 0)
    pf = f"{gp / gl:.2f}" if gl > 0 else "∞"
    eq = pk = dd = 0.0
    for r in rs:
        eq += r
        pk = max(pk, eq)
        dd = max(dd, pk - eq)
    return (f"{n} صفقة | نجاح {wins / n * 100:.0f}% | "
            f"متوسط R {sum(rs) / n:+.2f} | مجموع {sum(rs):+.1f}R | "
            f"PF {pf} | هبوط أقصى {dd:.1f}R")


def run(data, bt_start, blocked):
    cache = {}
    out = {}
    for key, fr in data.items():
        tr, op = run_asset(key, fr, data.get("BTC"), cache,
                           bt_start, blocked)
        out[key] = tr
        print(key, len(tr), "closed", "| open at end:", op)
    return out


def report(title, out):
    allt = [t for v in out.values() for t in v]
    lines = [title, "الكل: " + summ(allt)]
    for k, v in out.items():
        lines.append(f"{ASSETS[k][2]}: {summ(v)}")
    for sty in ("Breakout", "Retest"):
        sub = [t for t in allt if t["style"] == sty]
        if sub:
            lines.append(f"{sty}: {summ(sub)}")
    return "\n".join(lines)


def send(text):
    print(text)
    requests.post(
        f"https://api.telegram.org/bot{os.environ['TG_TOKEN']}/sendMessage",
        json={"chat_id": os.environ["TG_CHAT"], "text": text[:4000]},
        timeout=30)


def main():
    end = pd.Timestamp.now(tz="UTC").floor("15min")
    bt_start = end - pd.Timedelta(days=DAYS)
    data = {}
    for key, (kind, sym, name) in ASSETS.items():
        try:
            data[key] = load_asset(kind, sym, bt_start, end)
            print(key, "loaded",
                  {k: len(v) for k, v in data[key].items()})
        except Exception as e:
            print(key, "load error:", e)

    base = run(data, bt_start, lambda now: False)
    text = report("🧪 باك تيست بدون فلتر الأخبار (ورقي، قواعد A الحالية)",
                  base)
    if EVENTS:
        ev = [pd.Timestamp(e, tz="UTC") for e in EVENTS]
        filt = run(data, bt_start,
                   lambda now: any(t - BEFORE <= now <= t + AFTER
                                   for t in ev))
        text += "\n\n" + report("🧪 مع فلتر الأخبار (التواريخ اللي حطيتها)",
                                filt)
    text += (
        f"\n\nالفترة: آخر {DAYS} يوم للكريبتو، والذهب وناسداك آخر ~59 يوم "
        "فقط (حد yfinance).\n"
        "تحذير: بدون عمولة أو سبريد أو سلبيج، وفحص كل 30 دقيقة، "
        "والوقف يُفحص قبل الهدف داخل الشمعة الواحدة. "
        "نسبة النجاح تحسب كل صفقة R>0 (حتى لو خرجت على الدخول بعد TP1). "
        "الماضي ما يضمن المستقبل، ولا يشمل ميل الأخبار (Gemini).")
    send(text)


main()
