import os
import json
import math
import tempfile
import numpy as np
import requests
import pandas as pd
import yfinance as yf
import news

GEMINI_KEY = os.environ["GEMINI_KEY"]
TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT = os.environ["TG_CHAT"]
EVENT = os.environ.get("EVENT", "")
STATE_FILE = "state.json"

NDX_OKX = "US100-USDT-SWAP"  # عقد ناسداك على OKX (غيّره إذا اسمه مختلف)

ASSETS = {
    "BTC": ("bin", "BTCUSDT", "بيتكوين"),
    "ETH": ("bin", "ETHUSDT", "إيثريوم"),
    "SOL": ("bin", "SOLUSDT", "سولانا"),
    "GOLD": ("okx", "XAU-USDT", "الذهب"),
    "NDX": ("okx", NDX_OKX, "ناسداك"),
}
TR = {1: "صاعد", -1: "هابط", 0: "عرضي"}
OKX_BARS = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1Dutc"}
TD_BARS = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1day"}
USE_ICT = True  # False = تعطيل نقاط ICT بالكامل


def fmt(x):
    return f"{x:,.2f}" if abs(x) >= 10 else f"{x:.4f}"


def market_closed(key, now):
    if key not in ("GOLD", "NDX"):
        return False
    wd = now.weekday()
    hr = now.hour
    return wd == 5 or (wd == 6 and hr < 22) or (wd == 4 and hr >= 21)


def to_frame(rows):
    d = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 9]]
    d.columns = ["t", "o", "h", "l", "c", "v", "tb"]
    d = d.astype(float)
    d.index = pd.to_datetime(d["t"], unit="ms", utc=True)
    return d.drop(columns="t").iloc[:-1]


def bin_klines(sym, tf, n=300):
    for host in ("https://data-api.binance.vision",
                 "https://api.binance.com"):
        try:
            r = requests.get(
                host + "/api/v3/klines",
                params={"symbol": sym, "interval": tf, "limit": n},
                timeout=20)
            if r.ok:
                return to_frame(r.json())
        except Exception:
            pass
    raise RuntimeError("binance failed " + sym)


_OKX_ID = {}


def okx_resolve(inst):
    # يتأكد إن رمز OKX موجود، وإذا لا يجرب بدائل ويدور بقائمة العقود
    if inst in _OKX_ID:
        return _OKX_ID[inst]
    base = inst.split("-")[0]
    tried = [inst]
    for alt in (inst + "-SWAP", inst.replace("-SWAP", "")):
        if alt not in tried:
            tried.append(alt)
    for cand in tried:
        try:
            r = requests.get(
                "https://www.okx.com/api/v5/market/candles",
                params={"instId": cand, "bar": "1H", "limit": 3},
                timeout=20)
            j = r.json()
            if j.get("code") == "0" and j.get("data"):
                _OKX_ID[inst] = cand
                if cand != inst:
                    print("okx instrument", inst, "->", cand)
                return cand
            print("okx probe", cand, "->", str(j.get("code")),
                  str(j.get("msg"))[:60])
        except Exception as e:
            print("okx probe error:", cand, e)
    try:
        r = requests.get(
            "https://www.okx.com/api/v5/public/instruments",
            params={"instType": "SWAP"}, timeout=30)
        ids = [x["instId"] for x in r.json().get("data", [])]
        m = [i for i in ids if i.startswith(base)
             and i.endswith("-USDT-SWAP")]
        print("okx instruments matching", base, ":", m[:8])
        if m:
            _OKX_ID[inst] = m[0]
            return m[0]
    except Exception as e:
        print("okx discovery error:", e)
    _OKX_ID[inst] = inst
    return inst


def okx_klines(inst, tf, n=300):
    inst = okx_resolve(inst)
    r = requests.get(
        "https://www.okx.com/api/v5/market/candles",
        params={"instId": inst, "bar": OKX_BARS[tf], "limit": n},
        timeout=20)
    j = r.json()
    if j.get("code") != "0" or not j.get("data"):
        raise RuntimeError(f"okx {inst}: " + str(j)[:150])
    rows = [x for x in j["data"] if x[8] == "1"]
    d = pd.DataFrame(rows).iloc[:, :6]
    d.columns = ["t", "o", "h", "l", "c", "v"]
    d = d.astype(float)
    d.index = pd.to_datetime(d["t"], unit="ms", utc=True)
    return d.drop(columns="t").sort_index()


_OKX_ALT = {}


def okx_find(sym):
    # يدور على اسم العقد الصحيح بقائمة OKX إذا الاسم المكتوب مو موجود
    if sym in _OKX_ALT:
        return _OKX_ALT[sym]
    base = sym.split("-")[0]
    order = ("SWAP", "SPOT") if sym.endswith("SWAP") else ("SPOT", "SWAP")
    found = []
    for typ in order:
        try:
            r = requests.get(
                "https://www.okx.com/api/v5/public/instruments",
                params={"instType": typ}, timeout=20)
            for x in r.json().get("data", []):
                iid = x.get("instId", "")
                if iid.split("-")[0] == base and iid != sym:
                    found.append(iid)
        except Exception as e:
            print("okx list failed", typ, ":", e)
        if found:
            break
    found.sort(key=lambda i: ("USDT" not in i, i))
    _OKX_ALT[sym] = found[0] if found else None
    if found:
        print("okx candidates for", base, ":", found[:5])
    return _OKX_ALT[sym]


def td_frame(sym, tf, n=300):
    key = os.environ.get("TWELVE_KEY", "")
    if not key:
        raise RuntimeError("no TWELVE_KEY")
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": sym, "interval": TD_BARS[tf],
                "outputsize": n, "timezone": "UTC", "apikey": key},
        timeout=30)
    j = r.json()
    if "values" not in j:
        raise RuntimeError("twelvedata: " + str(j)[:200])
    d = pd.DataFrame(j["values"])
    d["t"] = pd.to_datetime(d["datetime"], utc=True)
    d = d.set_index("t").sort_index()
    out = d[["open", "high", "low", "close"]].astype(float)
    out.columns = ["o", "h", "l", "c"]
    out["v"] = 0.0
    return out.iloc[:-1]


def yf_frame(sym, period, interval):
    d = yf.download(sym, period=period, interval=interval, progress=False)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    d = d.rename(columns={"Open": "o", "High": "h", "Low": "l",
                          "Close": "c", "Volume": "v"})
    d = d[["o", "h", "l", "c", "v"]].dropna()
    d.index = pd.to_datetime(d.index, utc=True)
    return d.iloc[:-1]


def weekly(d):
    w = d.resample("W-SUN").agg(
        {"o": "first", "h": "max", "l": "min",
         "c": "last", "v": "sum"}).dropna()
    if len(d) and len(w) and d.index[-1].weekday() != 6:
        w = w.iloc[:-1]
    return w


def with_daily(fr, getter):
    try:
        fr["1d"] = getter()
    except Exception as e:
        print("1d failed:", e)
    return fr


def frames(kind, sym):
    if kind == "bin":
        return {tf: bin_klines(sym, tf) for tf in ("15m", "1h", "4h")}
    if kind == "okx":
        fr = None
        use = sym
        try:
            fr = {tf: okx_klines(use, tf) for tf in ("15m", "1h", "4h")}
        except Exception as e:
            print("okx failed", sym, ":", e)
            alt = okx_find(sym)
            if alt:
                try:
                    use = alt
                    fr = {tf: okx_klines(use, tf)
                          for tf in ("15m", "1h", "4h")}
                    print("okx using alternative:", alt)
                except Exception as e2:
                    print("okx alt failed", alt, ":", e2)
                    fr = None
        if fr is None:
            if sym.startswith("XAU"):
                return frames("td", "XAU/USD")
            return frames("yf", "^NDX")
        return with_daily(fr, lambda: okx_klines(use, "1d"))
    if kind == "td":
        fr = {tf: td_frame(sym, tf) for tf in ("15m", "1h", "4h")}
        return with_daily(fr, lambda: td_frame(sym, "1d"))
    f15 = yf_frame(sym, "5d", "15m")
    f1h = yf_frame(sym, "60d", "1h")
    f4h = f1h.resample("4h").agg(
        {"o": "first", "h": "max", "l": "min",
         "c": "last", "v": "sum"}).dropna().iloc[:-1]
    return with_daily({"15m": f15, "1h": f1h, "4h": f4h},
                      lambda: yf_frame(sym, "2y", "1d"))


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up.div(dn.where(dn != 0))
    out = 100 - 100 / (1 + rs)
    out = out.mask((dn == 0) & (up > 0), 100.0)
    out = out.mask((dn == 0) & (up == 0), 50.0)
    return out


def macd_h(s):
    m = ema(s, 12) - ema(s, 26)
    return m - ema(m, 9)


def atr(d, n=14):
    pc = d["c"].shift(1)
    tr = pd.concat([d["h"] - d["l"], (d["h"] - pc).abs(),
                    (d["l"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def trend(d):
    c = d["c"]
    e20 = float(ema(c, 20).iloc[-1])
    e50 = float(ema(c, 50).iloc[-1])
    p = float(c.iloc[-1])
    if p > e20 > e50:
        return 1
    if p < e20 < e50:
        return -1
    return 0


def swings(d, k=2):
    H = d["h"].values
    L = d["l"].values
    hs = []
    ls = []
    for i in range(k, len(d) - k):
        if H[i] == max(H[i - k:i + k + 1]):
            hs.append((i, float(H[i])))
        if L[i] == min(L[i - k:i + k + 1]):
            ls.append((i, float(L[i])))
    return hs, ls


def ict(d15, d1, s, a):
    if not USE_ICT:
        return 0.0, []
    try:
        return ict_calc(d15, d1, s, a)
    except Exception as e:
        print("ict error:", e)
        return 0.0, []


def ict_calc(d15, d1, s, a):
    pts = 0.0
    notes = []
    dd = d15.iloc[-80:]
    o = dd["o"].values
    h = dd["h"].values
    l = dd["l"].values
    c = dd["c"].values
    n = len(dd)
    price = float(c[-1])

    ref_lo = float(min(l[-28:-8]))
    ref_hi = float(max(h[-28:-8]))
    sw_lo = False
    sw_hi = False
    for i in range(n - 8, n):
        if l[i] < ref_lo and c[i] > ref_lo:
            sw_lo = True
        if h[i] > ref_hi and c[i] < ref_hi:
            sw_hi = True
    if (s == 1 and sw_lo) or (s == -1 and sw_hi):
        pts += 1.5
        notes.append("سحب سيولة " + ("قاع" if s == 1 else "قمة"))
    elif (s == 1 and sw_hi) or (s == -1 and sw_lo):
        pts -= 1.0
        notes.append("سحب سيولة عكس الاتجاه")

    hs, ls = swings(dd)
    if len(hs) >= 2 and len(ls) >= 2:
        last3 = c[-3:]
        if s == 1 and max(last3) > hs[-1][1]:
            if hs[-2][1] > hs[-1][1]:
                pts += 1.5
                notes.append("MSS صاعد")
            else:
                pts += 0.75
                notes.append("BOS صاعد")
        if s == -1 and min(last3) < ls[-1][1]:
            if ls[-2][1] < ls[-1][1]:
                pts += 1.5
                notes.append("MSS هابط")
            else:
                pts += 0.75
                notes.append("BOS هابط")

    fvg = None
    for i in range(n - 1, n - 21, -1):
        if s == 1 and l[i] > h[i - 2]:
            bot = float(h[i - 2])
            top = float(l[i])
            if top - bot > 0.1 * a and (
                    i == n - 1 or float(min(l[i + 1:])) > bot):
                fvg = (bot, top)
                break
        if s == -1 and h[i] < l[i - 2]:
            bot = float(h[i])
            top = float(l[i - 2])
            if top - bot > 0.1 * a and (
                    i == n - 1 or float(max(h[i + 1:])) < top):
                fvg = (bot, top)
                break
    if fvg:
        near = fvg[0] - 0.3 * a <= price <= fvg[1] + 0.3 * a
        pts += 1.0 if near else 0.5
        notes.append(
            f"FVG {'صاعد' if s == 1 else 'هابط'} "
            f"{fmt(fvg[0])}-{fmt(fvg[1])}" + (" (السعر عنده)" if near else ""))

    ob = None
    for i in range(n - 4, n - 31, -1):
        if s == 1 and c[i] < o[i]:
            if (max(c[i + 1:i + 4]) > h[i]
                    and max(h[i + 1:i + 4]) - l[i] > 1.5 * a
                    and float(min(l[i + 1:])) >= l[i]):
                ob = (float(l[i]), float(h[i]))
                break
        if s == -1 and c[i] > o[i]:
            if (min(c[i + 1:i + 4]) < l[i]
                    and h[i] - min(l[i + 1:i + 4]) > 1.5 * a
                    and float(max(h[i + 1:])) <= h[i]):
                ob = (float(l[i]), float(h[i]))
                break
    if ob:
        near = ob[0] - 0.3 * a <= price <= ob[1] + 0.3 * a
        pts += 1.0 if near else 0.5
        notes.append(
            f"أوردر بلوك {'صاعد' if s == 1 else 'هابط'} "
            f"{fmt(ob[0])}-{fmt(ob[1])}" + (" (السعر عنده)" if near else ""))

    hi = float(d1["h"].iloc[-48:].max())
    lo = float(d1["l"].iloc[-48:].min())
    if hi > lo:
        pos = (price - lo) / (hi - lo)
        if (s == 1 and pos < 0.5) or (s == -1 and pos > 0.5):
            pts += 0.5
            notes.append("ديسكاونت" if s == 1 else "بريميوم")
        elif (s == 1 and pos > 0.75) or (s == -1 and pos < 0.25):
            pts -= 0.5
            notes.append("دخول متأخر بمنطقة " +
                         ("بريميوم" if s == 1 else "ديسكاونت"))

    hr = dd.index[-1].hour
    if 7 <= hr < 10:
        pts += 0.5
        notes.append("كيل زون لندن")
    elif 12 <= hr < 15:
        pts += 0.5
        notes.append("كيل زون نيويورك")

    pts = max(-1.0, min(2.5, pts))
    return round(pts, 1), notes


A_MIN = 7.5

# =========================================================
# FIRST AGENT STRATEGY:
# Trend Continuation + Breakout / Retest
# =========================================================
TREND_A_MIN = 7.0
TREND_B_MIN = 5.5

# 15m breakout must clear the recent level by a small ATR buffer.
TREND_BREAK_BUFFER_ATR = 0.05

# A strong breakout may enter directly only when momentum/volume is clear.
TREND_STRONG_VOLUME = 1.25
TREND_STRONG_BODY_ATR = 0.55

# Do not chase a breakout that already ran too far from the broken level.
TREND_MAX_DIRECT_EXTENSION_ATR = 0.70

# Retest tolerance around the broken 15m level.
TREND_RETEST_TOUCH_ATR = 0.20
TREND_RETEST_CLOSE_ATR = 0.02

# Early watch distance before the actual breakout.
TREND_WATCH_DISTANCE_ATR = 0.35

# Stop buffer behind breakout/retest structure.
TREND_STOP_BUFFER_ATR = 0.20

USE_MIN_STOP = True
SPOT_ONLY = False  # True = شراء فقط، False = شراء وبيع (Margin)
GOLD_MTF = True    # يُستخدم فقط إذا GOLD_VP = False
GOLD_VP = True     # True = ستراتيجية الذهب: هيكل + بروفايل + دعم ومقاومة
GOLD_VP_MIN_RISK = 0.003  # أقل مسافة وقف للذهب = 0.3% من السعر
TF_ORDER = ["15m", "1h", "4h", "1d"]
MIN_RISK = {"BTC": 0.004, "ETH": 0.004, "SOL": 0.005,
            "GOLD": 0.002, "NDX": 0.0015}


def flow_stats(d15):
    try:
        if "tb" not in d15.columns:
            return None
        x = d15.tail(21)
        v = x["v"].astype(float)
        tb = x["tb"].astype(float)
        v20 = float(v.iloc[-20:].sum())
        if v.isnull().any() or tb.isnull().any() or v20 <= 0:
            return None
        delta = 2 * tb - v
        cvd = float(delta.iloc[-20:].sum()) / v20
        lastd = float(delta.iloc[-1]) / max(float(v.iloc[-1]), 1e-9)
        px = float(x["c"].iloc[-1]) - float(x["c"].iloc[-20])
        div = 0
        if px > 0 and cvd < -0.04:
            div = -1
        elif px < 0 and cvd > 0.04:
            div = 1
        return {"cvd": cvd, "last": lastd, "div": div}
    except Exception:
        return None


def vwap_day(d15):
    try:
        day = d15.index[-1].normalize()
        x = d15[d15.index >= day]
        if len(x) < 4:
            return None
        v = x["v"].astype(float)
        if float(v.sum()) <= 0:
            return None
        tp = (x["h"] + x["l"] + x["c"]) / 3
        return float((tp * v).sum() / v.sum())
    except Exception:
        return None


def sr_levels(d1, a1, price):
    try:
        x = d1.tail(240)
        H = x["h"].values
        L = x["l"].values
        k = 3
        pts = []
        for i in range(k, len(x) - k):
            if H[i] == max(H[i - k:i + k + 1]):
                pts.append(float(H[i]))
            if L[i] == min(L[i - k:i + k + 1]):
                pts.append(float(L[i]))
        pts.sort()
        tol = 0.35 * a1
        cl = []
        for p in pts:
            if cl and abs(p - cl[-1][0] / cl[-1][1]) <= tol:
                cl[-1][0] += p
                cl[-1][1] += 1
            else:
                cl.append([p, 1])
        lv = [(s / n, n) for s, n in cl]
        day = x.index[-1].normalize()
        prev = x[(x.index >= day - pd.Timedelta(days=1)) & (x.index < day)]
        if len(prev) >= 6:
            lv.append((float(prev["h"].max()), 3))
            lv.append((float(prev["l"].min()), 3))
        sup = [t for t in lv if t[0] < price and t[1] >= 2]
        res = [t for t in lv if t[0] > price and t[1] >= 2]
        sup_p = max(sup, key=lambda t: t[0]) if sup else None
        res_p = min(res, key=lambda t: t[0]) if res else None
        return sup_p, res_p
    except Exception:
        return None, None


def analyze_core(key, fr, btc_dir, live=None):
    d15, d1, d4 = fr["15m"], fr["1h"], fr["4h"]
    c = d15["c"]
    price = float(live) if live is not None else float(c.iloc[-1])
    a15 = float(atr(d15).iloc[-1])
    a1 = float(atr(d1).iloc[-1])
    t4, t1 = trend(d4), trend(d1)
    r15 = float(rsi(c).iloc[-1])
    mh = float(macd_h(c).iloc[-1])
    mh1 = float(macd_h(d1["c"]).iloc[-1])
    hi20 = float(d15["h"].iloc[-21:-1].max())
    lo20 = float(d15["l"].iloc[-21:-1].min())
    hi1 = float(d1["h"].iloc[-49:-1].max())
    lo1 = float(d1["l"].iloc[-49:-1].min())
    e200 = float(ema(d1["c"], 200).iloc[-1])
    e20 = float(ema(c, 20).iloc[-1])
    v = d15["v"]
    vr = None
    vm = float(v.iloc[-21:-1].mean())
    if vm > 0:
        vr = float(v.iloc[-1]) / vm
    fl = flow_stats(d15)
    vw = vwap_day(d15)
    sup_p, res_p = sr_levels(d1, a1, price)

    def plan(s, ep):
        if s == 1:
            stop = min(float(d15["l"].tail(12).min()), ep - a15) - 0.2 * a15
            risk = ep - stop
            lv = res_p[0] if (res_p and res_p[0] > ep) else hi1
        else:
            stop = max(float(d15["h"].tail(12).max()), ep + a15) + 0.2 * a15
            risk = stop - ep
            lv = sup_p[0] if (sup_p and sup_p[0] < ep) else lo1
        if (not math.isfinite(risk) or not math.isfinite(a1) or a1 <= 0
                or risk <= 0 or risk > 3 * a1):
            return None
        if USE_MIN_STOP and risk < MIN_RISK.get(key, 0.004) * ep:
            return None
        entry_sup, entry_res = sr_levels(d1, a1, ep)
        if s == 1:
            lv = entry_res[0] if entry_res else hi1
        else:
            lv = entry_sup[0] if entry_sup else lo1
        room = (lv - ep) * s
        if not math.isfinite(room):
            return None
        rr = min(3.0, room / risk) if room > 0 else 3.0
        tps = [ep + s * risk * k for k in (1, 2, 3)]
        return {"entry": ep, "stop": stop, "tps": tps, "rr": rr}

    def score(s):
        sc = 0.0
        sc += 2 if t4 == s else (-1 if t4 == -s else 0)
        sc += 2 if t1 == s else (-1 if t1 == -s else 0)
        brk = (price > hi20) if s == 1 else (price < lo20)
        sc += 2 if brk else 0
        sc += 1 if mh * s > 0 else 0
        sc += 0.5 if mh1 * s > 0 else 0
        ok = (50 < r15 < 72) if s == 1 else (28 < r15 < 50)
        sc += 1 if ok else 0
        if vr is not None:
            sc += 1 if vr >= 1.2 else 0
        # الفوليوم الناقص محايد، ما يضيف نقاط
        sc += 0.5 if (price > e200) == (s == 1) else 0
        if btc_dir is not None and key in ("ETH", "SOL"):
            sc += 1 if btc_dir == s else (-1.5 if btc_dir == -s else 0)
        if fl is not None:
            p = 0.0
            if fl["cvd"] * s > 0.04:
                p += 1.0
            elif fl["cvd"] * s < -0.04:
                p -= 1.0
            if fl["last"] * s > 0.10:
                p += 0.5
            if fl["div"] == -s:
                p -= 1.0
            elif fl["div"] == s:
                p += 0.5
            sc += max(-1.5, min(1.5, p))
        if vw is not None:
            sc += 0.5 if (price > vw) == (s == 1) else -0.5
        sc += ict(d15, d1, s, a15)[0]
        return round(max(0.0, min(10.0, sc)), 1), brk

    sl, bl = score(1)
    ss, bs = score(-1)
    s = 1 if sl >= ss else -1
    sc, brk = (sl, bl) if s == 1 else (ss, bs)
    last = d15.iloc[-1]
    if s == 1:
        rt = (t4 == 1 and t1 == 1 and float(last["l"]) <= e20 + 0.2 * a15
              and price > e20)
        lvl = hi20
        ext = (price - hi20) / a15
    else:
        rt = (t4 == -1 and t1 == -1 and float(last["h"]) >= e20 - 0.2 * a15
              and price < e20)
        lvl = lo20
        ext = (lo20 - price) / a15
    style = "Breakout" if brk else ("Pullback" if rt else None)

    obs = (f"مرصود: 4h {TR[t4]} | 1h {TR[t1]} | RSI15 {r15:.0f} | "
           f"MACD15 {'+' if mh > 0 else '-'}")
    if vr is not None:
        obs += f" | فوليوم {vr:.1f}x"
    obs += f" | امتداد {ext:.1f} ATR"
    if fl is not None:
        who = "شراء" if fl["cvd"] > 0 else "بيع"
        obs += (f"\nOrder flow: ضغط {who} {fl['cvd'] * 100:+.0f}% "
                f"(20 شمعة) | آخر شمعة {fl['last'] * 100:+.0f}%")
        if fl["div"] == -1:
            obs += " | تباعد: السعر صاعد والبيع عدواني"
        elif fl["div"] == 1:
            obs += " | تباعد: السعر نازل والشراء عدواني"
    else:
        obs += "\nOrder flow: غير متوفر لهذا الأصل"
    lv_txt = []
    if sup_p:
        lv_txt.append(f"دعم {fmt(sup_p[0])} ({sup_p[1]} لمسات)")
    if res_p:
        lv_txt.append(f"مقاومة {fmt(res_p[0])} ({res_p[1]} لمسات)")
    if vw is not None:
        lv_txt.append(f"VWAP {fmt(vw)}")
    if lv_txt:
        obs += "\n" + " | ".join(lv_txt)
    notes = ict(d15, d1, s, a15)[1]
    if notes:
        obs += "\nICT: " + " | ".join(notes)
    res = {"key": key, "side": s, "score": sc, "price": price,
           "obs": obs, "tier": None, "style": style, "ext": round(ext, 1)}

    pl = plan(s, price) if style else None
    if style and pl and sc >= A_MIN and pl["rr"] >= 2:
        res.update(tier="A", plan=pl)
        return res

    if brk and sc >= A_MIN and ext > 0.5:
        pa = plan(s, lvl)
        if pa and pa["rr"] >= 2:
            res.update(tier="ARMED", plan=pa, style="BreakRetest",
                       zone=[lvl - 0.3 * a15, lvl + 0.3 * a15])
            return res

    if sc >= 6:
        if s == 1:
            sw = float(d15["l"].tail(12).min())
            cond = (f"ريتست {fmt(lvl)} وثبات فوقه" if brk
                    else f"إغلاق شمعة فوق {fmt(lvl)} مع فوليوم")
            inv = f"كسر {fmt(sw)}"
        else:
            sw = float(d15["h"].tail(12).max())
            cond = (f"ريتست {fmt(lvl)} ورفض تحته" if brk
                    else f"إغلاق شمعة تحت {fmt(lvl)} مع فوليوم")
            inv = f"اختراق {fmt(sw)}"
        res.update(tier="B", cond=cond, inv=inv)
    return res


def gold_profiles(fr):
    out = [("15m", {"15m": fr["15m"], "1h": fr["1h"], "4h": fr["4h"]})]
    d1d = fr.get("1d")
    if d1d is not None and len(d1d) >= 60 and len(fr["4h"]) >= 60:
        out.append(("1h", {"15m": fr["1h"], "1h": fr["4h"], "4h": d1d}))
        wk = weekly(d1d)
        if len(wk) >= 35:
            out.append(("4h", {"15m": fr["4h"], "1h": d1d, "4h": wk}))
    return out


def gold_best(fr, btc_dir):
    live = float(fr["15m"]["c"].iloc[-1])
    cands = []
    for name, f in gold_profiles(fr):
        try:
            r = analyze_core("GOLD", f, btc_dir,
                             None if name == "15m" else live)
        except Exception as e:
            print("gold", name, "error:", e)
            continue
        r["tf"] = name
        cands.append(r)
    if not cands:
        raise RuntimeError("no gold profile")
    rank = {"A": 3, "ARMED": 2, "B": 1, None: 0}
    order = {"15m": 0, "1h": 1, "4h": 2}
    cands.sort(key=lambda r: (rank[r["tier"]], r["score"],
                              order[r["tf"]]), reverse=True)
    best = cands[0]
    others = [f"{r['tf']}: {r['score']}/10" for r in cands
              if r is not best]
    best["obs"] += f"\nفريم الدخول: {best['tf']}"
    if others:
        best["obs"] += " | باقي الفريمات " + ", ".join(others)
    return best


def vol_profile(d, bins=60):
    try:
        lo = float(d["l"].min())
        hi = float(d["h"].max())
        if not hi > lo:
            return None
        step = (hi - lo) / bins
        w = d["v"].astype(float).values
        use_vol = float(w.sum()) > 0
        if not use_vol:
            w = np.ones(len(d))
        L = d["l"].values
        H = d["h"].values
        i0 = np.clip(((L - lo) / step).astype(int), 0, bins - 1)
        i1 = np.clip(((H - lo) / step).astype(int), 0, bins - 1)
        vol = np.zeros(bins)
        for a, b, x in zip(i0, i1, w):
            vol[a:b + 1] += x / (b - a + 1)
        poc_i = int(vol.argmax())
        target = 0.7 * float(vol.sum())
        lo_i = hi_i = poc_i
        acc = float(vol[poc_i])
        while acc < target and (lo_i > 0 or hi_i < bins - 1):
            up = float(vol[hi_i + 1]) if hi_i < bins - 1 else -1.0
            dn = float(vol[lo_i - 1]) if lo_i > 0 else -1.0
            if up >= dn:
                hi_i += 1
                acc += up
            else:
                lo_i -= 1
                acc += dn
        return {"poc": float(lo + (poc_i + 0.5) * step),
                "vah": float(lo + (hi_i + 1) * step),
                "val": float(lo + lo_i * step),
                "vol": use_vol}
    except Exception:
        return None


def mstruct(d, k=3):
    hs, ls = swings(d.tail(200), k)
    if len(hs) < 2 or len(ls) < 2:
        return 0
    if hs[-1][1] > hs[-2][1] and ls[-1][1] > ls[-2][1]:
        return 1
    if hs[-1][1] < hs[-2][1] and ls[-1][1] < ls[-2][1]:
        return -1
    return 0


def sr_multi(fr, tol):
    wts = {"15m": 1, "1h": 2, "4h": 3, "1d": 4}
    pts = []
    for tf, w in wts.items():
        d = fr.get(tf)
        if d is None or len(d) < 20:
            continue
        x = d.tail(300 if tf == "15m" else 240)
        H = x["h"].values
        L = x["l"].values
        for i in range(3, len(x) - 3):
            if H[i] == max(H[i - 3:i + 4]):
                pts.append((float(H[i]), w, tf))
            if L[i] == min(L[i - 3:i + 4]):
                pts.append((float(L[i]), w, tf))
    pts.sort(key=lambda t: t[0])
    cl = []
    for p, w, tf in pts:
        if cl and abs(p - cl[-1]["p"]) <= tol:
            c = cl[-1]
            c["p"] = (c["p"] * c["n"] + p) / (c["n"] + 1)
            c["n"] += 1
            c["w"] += w
            c["tfs"].add(tf)
        else:
            cl.append({"p": p, "n": 1, "w": w, "tfs": {tf}})
    return [c for c in cl if c["w"] >= 3]


def sr_label(c):
    tfs = "+".join(x for x in TF_ORDER if x in c["tfs"])
    return f"{fmt(c['p'])} ({tfs})"


def gold_vp(fr):
    d15 = fr["15m"]
    d1 = fr["1h"]
    price = float(d15["c"].iloc[-1])
    res = {"key": "GOLD", "side": 1, "score": 0.0, "price": price,
           "obs": "", "tier": None, "style": "VProfile", "tf": "15m",
           "ext": 0.0, "miss": []}
    if len(d15) < 150 or len(d1) < 60 or len(fr["4h"]) < 60:
        res["obs"] = "ستراتيجية الذهب: بيانات ناقصة"
        return res
    a15 = float(atr(d15).iloc[-1])
    a1 = float(atr(d1).iloc[-1])
    prof = vol_profile(d15.tail(300))
    if (prof is None or not math.isfinite(a15) or not math.isfinite(a1)
            or a15 <= 0 or a1 <= 0):
        res["obs"] = "ستراتيجية الذهب: تعذر حساب البروفايل"
        return res

    s = mstruct(d1)
    tolz = 0.35 * a1
    srl = sr_multi(fr, tolz)
    sups = sorted([c for c in srl if c["p"] < price], key=lambda c: -c["p"])
    ress = sorted([c for c in srl if c["p"] > price], key=lambda c: c["p"])
    side = s if s != 0 else (1 if price >= prof["poc"] else -1)
    res["side"] = side
    if side == 1:
        zones = [("VAL", prof["val"]), ("POC", prof["poc"])]
    else:
        zones = [("VAH", prof["vah"]), ("POC", prof["poc"])]

    w15 = d15.tail(24)
    n = len(w15)
    Lw = w15["l"].values
    Hw = w15["h"].values
    Cw = w15["c"].values
    t = None
    zname = None
    zlvl = None
    for i in range(n - 1, -1, -1):
        for nm, lv in zones:
            if Lw[i] <= lv + tolz and Hw[i] >= lv - tolz:
                t, zname, zlvl = i, nm, lv
                break
        if t is not None:
            break

    trig = False
    if t is not None and t <= n - 3:
        if side == 1:
            brk_lvl = float(Hw[t:n - 1].max())
            trig = bool(Cw[-1] > brk_lvl)
        else:
            brk_lvl = float(Lw[t:n - 1].min())
            trig = bool(Cw[-1] < brk_lvl)
    ext_ok = True
    if zlvl is not None:
        ext = ((price - zlvl) if side == 1 else (zlvl - price)) / a1
        res["ext"] = round(ext, 1)
        ext_ok = ext <= 1.5

    score = 0.0
    score += 3 if s != 0 else 0
    score += 2 if t is not None else 0
    score += 3 if trig else 0
    if zlvl is not None and any(abs(c["p"] - zlvl) <= tolz for c in srl):
        score += 1
    score += 1 if trend(fr["4h"]) == side else 0
    res["score"] = float(score)

    names = {1: "صاعد (قمم وقيعان أعلى)", -1: "هابط (قمم وقيعان أدنى)",
             0: "عرضي/غير واضح"}
    src = "حجم" if prof["vol"] else "زمن، بدون فوليوم"
    obs = (f"هيكل 1h: {names[s]}\n"
           f"بروفايل ({src}): POC {fmt(prof['poc'])} | "
           f"VAH {fmt(prof['vah'])} | VAL {fmt(prof['val'])}")
    if sups:
        obs += "\nدعم: " + " | ".join(sr_label(c) for c in sups[:2])
    if ress:
        obs += "\nمقاومة: " + " | ".join(sr_label(c) for c in ress[:2])
    if t is not None:
        obs += (f"\nمنطقة {zname} ({fmt(zlvl)}) انلمست قبل "
                f"{n - 1 - t} شمعة 15m")
    obs += "\nفريم الدخول: 15m"

    miss = []
    plan = None
    names_t = []
    if s == 0:
        miss.append("هيكل 1h غير واضح")
    elif t is None:
        miss.append("ما لمس منطقة القيمة بآخر 6 ساعات")
    elif t > n - 3:
        miss.append("ينتظر كسر هيكل 15m")
    elif not trig:
        miss.append("ما صار كسر هيكل 15m")
    elif not ext_ok:
        miss.append("السعر ابتعد عن المنطقة")

    if s != 0 and trig and ext_ok:
        lo_s = max(t - 2, 0)
        if side == 1:
            stop = float(Lw[lo_s:].min()) - 0.25 * a15
            risk = price - stop
        else:
            stop = float(Hw[lo_s:].max()) + 0.25 * a15
            risk = stop - price
        if (not math.isfinite(risk) or risk <= 0 or risk > 3 * a1
                or risk < GOLD_VP_MIN_RISK * price):
            miss.append("الوقف ضيق أو واسع")
        else:
            lv = [(prof["poc"], "POC"), (prof["vah"], "VAH"),
                  (prof["val"], "VAL")]
            for c in srl:
                kind = "مقاومة " if c["p"] > price else "دعم "
                lv.append((c["p"], kind + "+".join(
                    x for x in TF_ORDER if x in c["tfs"])))
            blocked = any(
                c["w"] >= 4 and
                0.2 * risk <= (c["p"] - price) * side < 0.8 * risk
                for c in srl)
            if blocked:
                miss.append("مستوى قوي قريب قدام الدخول")
            else:
                cand = sorted(
                    [x for x in lv if (x[0] - price) * side >= 0.8 * risk],
                    key=lambda x: (x[0] - price) * side)
                tps = []
                for p, nm in cand:
                    if not tps or abs(p - tps[-1]) >= 0.5 * risk:
                        tps.append(p)
                        names_t.append(nm)
                    if len(tps) == 3:
                        break
                while len(tps) < 3:
                    base = tps[-1] if tps else price
                    tps.append(base + side * risk)
                    names_t.append("+1R")
                rr = abs(tps[1] - price) / risk
                if rr >= 2:
                    plan = {"entry": price, "stop": stop, "tps": tps,
                            "rr": round(min(3.0, rr), 1)}
                else:
                    miss.append("R:R أقل من 2")

    if plan is not None:
        obs += ("\nالأهداف: TP1 " + names_t[0] + " | TP2 " + names_t[1]
                + " | TP3 " + names_t[2])
        res.update(tier="A", plan=plan, obs=obs)
        return res

    if s != 0 and t is not None:
        obs += "\nناقص للدخول: " + " | ".join(miss)
    res["obs"] = obs
    res["miss"] = miss
    if s != 0 and t is not None and score >= 5:
        if side == 1:
            lv_prev = float(Hw[t:].max())
            sw = float(Lw[max(t - 2, 0):].min())
            cond = f"إغلاق شمعة 15m فوق {fmt(lv_prev)}"
            inv = "كسر " + fmt(sw)
        else:
            lv_prev = float(Lw[t:].min())
            sw = float(Hw[max(t - 2, 0):].max())
            cond = f"إغلاق شمعة 15m تحت {fmt(lv_prev)}"
            inv = "اختراق " + fmt(sw)
        res.update(tier="B", cond=cond, inv=inv)
    return res


ZONE_MODE = True   # True = الذهب وناسداك يشتغلون بتحليل المناطق
ZONE_ASSETS = ("GOLD", "NDX")
ZONE_MIN_RISK = {"GOLD": 0.0025, "NDX": 0.003}  # أقل مسافة وقف
ZONE_MIN_Q = 4.0        # أقل جودة لصفقة مؤكدة
ZONE_MIN_EARLY = 3.0    # أقل جودة لتنبيه مبكر
RISK_UP = {"منخفضة": "متوسطة", "متوسطة": "عالية", "عالية": "عالية"}


def cisd(d, s):
    # CISD: إغلاق يكسر افتتاح أول شمعة بآخر سلسلة شموع معاكسة (2 فأكثر)
    try:
        x = d.tail(24)
        o = x["o"].values
        c = x["c"].values
        n = len(x)

        def opp(i):
            return c[i] < o[i] if s == 1 else c[i] > o[i]

        e = None
        for i in range(n - 2, 0, -1):
            if opp(i):
                e = i
                break
        if e is None or e < n - 8:
            return False
        b = e
        while b - 1 >= 0 and opp(b - 1):
            b -= 1
        if e - b + 1 < 2:
            return False
        lvl = o[b]
        return bool(c[-1] > lvl) if s == 1 else bool(c[-1] < lvl)
    except Exception:
        return False


def zone_strat(key, fr):
    d15, d1, d4 = fr["15m"], fr["1h"], fr["4h"]
    price = float(d15["c"].iloc[-1])
    res = {"key": key, "side": 1, "score": 0.0, "price": price,
           "obs": "", "tier": None, "style": None, "tf": "15m",
           "ext": 0.0, "miss": [], "risk": None}
    if len(d15) < 120 or len(d1) < 60 or len(d4) < 60:
        res["obs"] = "تحليل المناطق: بيانات ناقصة"
        return res
    a15 = float(atr(d15).iloc[-1])
    a1 = float(atr(d1).iloc[-1])
    if (not math.isfinite(a15) or not math.isfinite(a1)
            or a15 <= 0 or a1 <= 0):
        res["obs"] = "تحليل المناطق: تعذر حساب ATR"
        return res
    res["a1"] = a1
    zw = 0.25 * a1
    srl = sr_multi(fr, 0.35 * a1)
    t1, t4 = trend(d1), trend(d4)
    r15 = float(rsi(d15["c"]).iloc[-1])
    r1 = float(rsi(d1["c"]).iloc[-1])
    e20 = float(ema(d1["c"], 20).iloc[-1])
    e50 = float(ema(d1["c"], 50).iloc[-1])
    vw = vwap_day(d15)
    C = d15["c"].values[-12:]
    L = d15["l"].values[-12:]
    H = d15["h"].values[-12:]
    O = d15["o"].values[-12:]
    names = {1: "صاعد", -1: "هابط", 0: "عرضي"}

    cands = []
    for z in srl:
        bot = z["p"] - zw
        top = z["p"] + zw
        if (C[-1] < bot - 0.1 * a15 and C[-2] < bot - 0.1 * a15
                and (C[-8:-2] >= bot).any() and bot - price <= 1.5 * a1):
            cands.append(("BREAK", -1, z, 3))
            continue
        if (C[-1] > top + 0.1 * a15 and C[-2] > top + 0.1 * a15
                and (C[-8:-2] <= top).any() and price - top <= 1.5 * a1):
            cands.append(("BREAK", 1, z, 3))
            continue
        ov = [i for i in range(4, 12) if L[i] <= top and H[i] >= bot]
        if ov:
            t = ov[-1]
            rng = max(H[t] - L[t], 1e-9)
            lw = (min(O[t], C[t]) - L[t]) / rng
            uw = (H[t] - max(O[t], C[t])) / rng
            if (price > top and C[4:].min() >= bot - 0.1 * a15
                    and C[t] >= bot and (lw >= 0.4 or cisd(d15, 1))):
                cands.append(("BOUNCE", 1, z, 2))
                continue
            if (price < bot and C[4:].max() <= top + 0.1 * a15
                    and C[t] <= top and (uw >= 0.4 or cisd(d15, -1))):
                cands.append(("BOUNCE", -1, z, 2))
                continue
        if bot - 0.15 * a1 <= price <= top + 0.15 * a1:
            sd = 1 if float(C[:-1].mean()) > z["p"] else -1
            cands.append(("EARLY", sd, z, 1))

    def quality(kind, s, z):
        q = 0.0
        q += 2 if t1 == s else (-1 if t1 == -s else 0)
        q += 1 if t4 == s else (-0.5 if t4 == -s else 0)
        q += 1 if (e20 > e50) == (s == 1) else 0
        if kind == "BREAK":
            ok = (50 <= r15 <= 75) if s == 1 else (25 <= r15 <= 50)
        else:
            ok = (30 <= r15 <= 60) if s == 1 else (40 <= r15 <= 70)
        q += 1 if ok else 0
        if vw is not None:
            q += 1 if (price > vw) == (s == 1) else 0
        q += max(-1.0, min(2.0, ict(d15, d1, s, a15)[0]))
        q += 1.5 if cisd(d15, s) else 0
        q += 1 if z["w"] >= 6 else 0
        q += 0.5 if ("4h" in z["tfs"] or "1d" in z["tfs"]) else 0
        return round(max(0.0, min(10.0, q)), 1)

    def build(kind, s, z):
        bot = z["p"] - zw
        top = z["p"] + zw
        if kind == "BREAK":
            stop = (bot - 0.2 * a15) if s == 1 else (top + 0.2 * a15)
        elif s == 1:
            stop = min(bot, float(L[-8:].min())) - 0.25 * a15
        else:
            stop = max(top, float(H[-8:].max())) + 0.25 * a15
        risk = (price - stop) * s
        if (not math.isfinite(risk) or risk <= 0 or risk > 3 * a1
                or risk < ZONE_MIN_RISK.get(key, 0.003) * price):
            return None, None, "الوقف ضيق أو واسع"
        lv = [(e20, "EMA20 1h"), (e50, "EMA50 1h")]
        if vw is not None:
            lv.append((vw, "VWAP"))
        for c in srl:
            if abs(c["p"] - z["p"]) < 1e-9:
                continue
            kd = "مقاومة " if c["p"] > price else "دعم "
            lv.append((c["p"], kd + "+".join(
                x for x in TF_ORDER if x in c["tfs"])))
        blocked = any(
            c["w"] >= 4 and abs(c["p"] - z["p"]) > 1e-9
            and 0.2 * risk <= (c["p"] - price) * s < 0.8 * risk
            for c in srl)
        if blocked:
            return None, None, "مستوى قوي قريب قدام الدخول"
        cand = sorted([x for x in lv
                       if 0.8 * risk <= (x[0] - price) * s <= 4 * risk],
                      key=lambda x: (x[0] - price) * s)
        tps = []
        nm = []
        for p_, n_ in cand:
            if not tps or abs(p_ - tps[-1]) >= 0.5 * risk:
                tps.append(p_)
                nm.append(n_)
            if len(tps) == 3:
                break
        while len(tps) < 3:
            base = tps[-1] if tps else price
            tps.append(base + s * risk)
            nm.append("+1R")
        rr = abs(tps[1] - price) / risk
        if rr < 2:
            return None, None, "R:R أقل من 2"
        return ({"entry": price, "stop": stop, "tps": tps,
                 "rr": round(min(3.0, rr), 1)}, nm, None)

    scored = [(rk, quality(kd, sd, z), kd, sd, z) for kd, sd, z, rk in cands]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    miss = []
    for rk, q, kind, s, z in scored:
        if kind == "EARLY" and q < ZONE_MIN_EARLY:
            miss.append("تنبيه مبكر: الجودة منخفضة")
            continue
        if kind != "EARLY" and q < ZONE_MIN_Q:
            miss.append("جودة الإشارة منخفضة")
            continue
        plan, tnames, why = build(kind, s, z)
        if plan is None:
            miss.append(why)
            continue
        bot = z["p"] - zw
        top = z["p"] + zw
        if kind == "EARLY":
            risk_g = "عالية"
        else:
            risk_g = "منخفضة" if q >= 7 else ("متوسطة" if q >= 5 else "عالية")
        kind_txt = {"BREAK": "كسر بثبات", "BOUNCE": "ارتداد مؤكد",
                    "EARLY": "تنبيه مبكر (بدون تأكيد)"}[kind]
        if kind == "BREAK":
            ztxt = ("مقاومة مكسورة " if s == 1 else "دعم مكسور ")
        else:
            ztxt = ("دعم " if s == 1 else "مقاومة ")
        ztxt += zlabel(z)
        vtxt = ("غير متوفر" if vw is None
                else ("السعر فوق VWAP" if price > vw else "السعر تحت VWAP"))
        obs = (f"المنطقة: {ztxt} | {fmt(bot)} - {fmt(top)}\n"
               f"نوع الإشارة: {kind_txt}\n"
               f"RSI15 {r15:.0f} | RSI1h {r1:.0f} | EMA20/50 1h "
               f"{'صاعد' if e20 > e50 else 'هابط'} | VWAP: {vtxt}\n"
               f"ترند 1h {names[t1]} | 4h {names[t4]} | "
               f"CISD {'نعم' if cisd(d15, s) else 'لا'}")
        nt = ict(d15, d1, s, a15)[1]
        if nt:
            obs += "\nICT: " + " | ".join(nt)
        obs += ("\nالأهداف: TP1 " + tnames[0] + " | TP2 " + tnames[1]
                + " | TP3 " + tnames[2])
        res.update(
            side=s, score=q, plan=plan, risk=risk_g, obs=obs,
            style={"BREAK": "ZBreak", "BOUNCE": "ZBounce",
                   "EARLY": "ZEarly"}[kind],
            tier="EARLY" if kind == "EARLY" else "A",
            zone_p=float(z["p"]), zone_txt=ztxt,
            ext=round(abs(price - z["p"]) / a1, 1))
        if kind == "EARLY":
            if s == 1:
                res["cond"] = (f"إغلاق شمعة 15m فوق {fmt(top)} "
                               "مع شمعة رفض أو CISD")
                res["inv"] = "كسر " + fmt(plan["stop"])
            else:
                res["cond"] = (f"إغلاق شمعة 15m تحت {fmt(bot)} "
                               "مع شمعة رفض أو CISD")
                res["inv"] = "اختراق " + fmt(plan["stop"])
        return res

    side = t1 if t1 != 0 else (1 if price >= e20 else -1)
    res["side"] = side
    ups = sorted([c for c in srl if c["p"] > price], key=lambda c: c["p"])
    dns = sorted([c for c in srl if c["p"] < price], key=lambda c: -c["p"])
    obs = (f"تحليل المناطق: ترند 1h {names[t1]} | 4h {names[t4]} | "
           f"RSI15 {r15:.0f}")
    if dns:
        obs += "\nدعم: " + " | ".join(sr_label(c) for c in dns[:2])
    if ups:
        obs += "\nمقاومة: " + " | ".join(sr_label(c) for c in ups[:2])
    if vw is not None:
        obs += f"\nVWAP {fmt(vw)}"
    if miss:
        obs += "\nناقص: " + " | ".join(dict.fromkeys(miss))
    res["obs"] = obs
    return res


def zlabel(z):
    return sr_label(z)



# =========================================================
# TREND CONTINUATION + BREAKOUT / RETEST
# =========================================================

def trend_breakout_strat(key, fr, btc_dir):
    """
    Simple first-agent strategy.

    1H:
      - decides LONG / SHORT using EMA20 + EMA50 trend.
      - if 1H is range, no entry.

    15m:
      - fresh breakout of a clear recent level, OR
      - clean retest after a breakout.

    Direct A entry:
      - only for a fresh strong breakout with clear momentum/volume,
        or a confirmed clean retest.

    ARMED:
      - valid breakout but direct entry is not ideal;
        wait for a retest of the broken level.

    4H, volume, 15m EMA and BTC direction for ETH/SOL:
      - quality/score only; they are not a pile of mandatory conditions.
    """
    d15, d1, d4 = fr["15m"], fr["1h"], fr["4h"]

    price = float(d15["c"].iloc[-1])
    res = {
        "key": key,
        "side": 1,
        "score": 0.0,
        "price": price,
        "obs": "",
        "tier": None,
        "style": None,
        "tf": "15m",
        "ext": 0.0,
        "risk": None,
        "miss": [],
    }

    if len(d15) < 60 or len(d1) < 60 or len(d4) < 30:
        res["obs"] = "Trend Breakout: بيانات غير كافية"
        return res

    a15 = float(atr(d15).iloc[-1])
    a1 = float(atr(d1).iloc[-1])

    if (not math.isfinite(a15) or not math.isfinite(a1)
            or a15 <= 0 or a1 <= 0):
        res["obs"] = "Trend Breakout: تعذر حساب ATR"
        return res

    # -----------------------------------------------------
    # 1H decides the trading direction.
    # -----------------------------------------------------
    t1 = trend(d1)
    t4 = trend(d4)

    if t1 == 0:
        res["side"] = t4 if t4 != 0 else 1
        res["obs"] = (
            f"Trend Breakout: 1h {TR[t1]} | 4h {TR[t4]}\n"
            "الحالة: 1H رينج — لا دخول، ننتظر اتجاه أو كسر أوضح."
        )
        return res

    s = t1
    res["side"] = s

    # -----------------------------------------------------
    # 15m breakout level.
    #
    # Use 20 candles BEFORE the previous candle. This lets
    # us identify:
    #   1) previous-candle breakout + current retest
    #   2) current fresh breakout
    # without moving the level with the breakout candle.
    # -----------------------------------------------------
    base = d15.iloc[-22:-2]
    prev = d15.iloc[-2]
    last = d15.iloc[-1]

    if len(base) < 15:
        res["obs"] = "Trend Breakout: تاريخ 15m غير كافي"
        return res

    hi = float(base["h"].max())
    lo = float(base["l"].min())
    level = hi if s == 1 else lo

    break_buffer = TREND_BREAK_BUFFER_ATR * a15
    touch_buffer = TREND_RETEST_TOUCH_ATR * a15
    close_buffer = TREND_RETEST_CLOSE_ATR * a15

    prev_close = float(prev["c"])
    last_close = float(last["c"])
    last_open = float(last["o"])
    last_high = float(last["h"])
    last_low = float(last["l"])

    if s == 1:
        prev_broke = prev_close > hi + break_buffer
        fresh_break = (
            last_close > hi + break_buffer
            and prev_close <= hi + break_buffer
        )
        clean_retest = (
            prev_broke
            and last_low <= hi + touch_buffer
            and last_close >= hi + close_buffer
        )
        approaching = (
            0 <= hi - last_close <= TREND_WATCH_DISTANCE_ATR * a15
        )
        extension = max(0.0, (last_close - hi) / a15)
    else:
        prev_broke = prev_close < lo - break_buffer
        fresh_break = (
            last_close < lo - break_buffer
            and prev_close >= lo - break_buffer
        )
        clean_retest = (
            prev_broke
            and last_high >= lo - touch_buffer
            and last_close <= lo - close_buffer
        )
        approaching = (
            0 <= last_close - lo <= TREND_WATCH_DISTANCE_ATR * a15
        )
        extension = max(0.0, (lo - last_close) / a15)

    res["ext"] = round(extension, 2)

    # -----------------------------------------------------
    # Light momentum / volume context.
    # -----------------------------------------------------
    e20_15 = float(ema(d15["c"], 20).iloc[-1])
    e50_15 = float(ema(d15["c"], 50).iloc[-1])
    ema15_ok = (
        (last_close > e20_15 > e50_15)
        if s == 1
        else (last_close < e20_15 < e50_15)
    )

    body = abs(last_close - last_open)
    body_atr = body / a15

    v = d15["v"].astype(float)
    vm = float(v.iloc[-21:-1].mean())
    vr = (float(v.iloc[-1]) / vm) if vm > 0 else None

    strong_volume = vr is not None and vr >= TREND_STRONG_VOLUME
    strong_body = body_atr >= TREND_STRONG_BODY_ATR

    # A strong direct breakout needs real momentum, not only a tiny close
    # a few points outside the level.
    strong_breakout = fresh_break and strong_volume and strong_body

    # -----------------------------------------------------
    # Simple score: no indicator soup.
    # -----------------------------------------------------
    score = 3.0  # 1H trend is present and decides direction.

    if t4 == s:
        score += 1.0
    elif t4 == -s:
        score -= 0.5

    if clean_retest:
        score += 3.0
    elif fresh_break:
        score += 2.5
    elif prev_broke:
        score += 1.5
    elif approaching:
        score += 1.0

    if ema15_ok:
        score += 1.0

    if vr is not None:
        if vr >= 1.50:
            score += 1.0
        elif vr >= 1.00:
            score += 0.5
        elif vr < 0.65:
            score -= 0.5

    if strong_body:
        score += 0.5

    if btc_dir is not None and key in ("ETH", "SOL"):
        if btc_dir == s:
            score += 0.5
        elif btc_dir == -s:
            score -= 0.5

    score = round(max(0.0, min(10.0, score)), 1)
    res["score"] = score

    # -----------------------------------------------------
    # Trade-plan builder.
    # Targets remain simple 1R / 2R / 3R.
    # We require enough room for TP2 before an observed 1H
    # support/resistance when one is available.
    # -----------------------------------------------------
    def make_plan(entry, mode):
        entry = float(entry)

        if s == 1:
            if mode == "RETEST":
                structure = min(
                    float(last["l"]) - 0.15 * a15,
                    level - 0.50 * a15,
                )
            else:
                structure = min(
                    level - 0.35 * a15,
                    float(last["l"]) - 0.10 * a15,
                )
            stop = structure
            risk = entry - stop
        else:
            if mode == "RETEST":
                structure = max(
                    float(last["h"]) + 0.15 * a15,
                    level + 0.50 * a15,
                )
            else:
                structure = max(
                    level + 0.35 * a15,
                    float(last["h"]) + 0.10 * a15,
                )
            stop = structure
            risk = stop - entry

        if not math.isfinite(risk) or risk <= 0:
            return None

        # Avoid absurdly tight/wide stops while keeping ATR as the main guide.
        if risk < 0.25 * a15 or risk > 2.5 * a1:
            return None

        sup_p, res_p = sr_levels(d1, a1, entry)
        obstacle = None

        if s == 1 and res_p and res_p[0] > entry:
            obstacle = float(res_p[0])
        elif s == -1 and sup_p and sup_p[0] < entry:
            obstacle = float(sup_p[0])

        if obstacle is not None:
            room_r = ((obstacle - entry) * s) / risk
            if room_r < 2.0:
                return None
            rr = min(3.0, room_r)
        else:
            rr = 3.0

        tps = [
            entry + s * risk,
            entry + s * 2.0 * risk,
            entry + s * 3.0 * risk,
        ]

        return {
            "entry": entry,
            "stop": float(stop),
            "tps": [float(x) for x in tps],
            "rr": round(float(rr), 2),
        }

    # -----------------------------------------------------
    # Human-readable observations.
    # -----------------------------------------------------
    vr_text = "غير متوفر" if vr is None else f"{vr:.2f}x"
    setup_text = (
        "Retest"
        if clean_retest
        else ("Fresh Breakout" if fresh_break
              else ("Broken — waiting retest" if prev_broke
                    else "Approaching breakout"))
    )

    res["obs"] = (
        f"Trend Continuation: 1h {TR[t1]} | 4h {TR[t4]}\n"
        f"15m: {setup_text} | المستوى {fmt(level)} | "
        f"EMA20/50 {'متوافق' if ema15_ok else 'غير متوافق'}\n"
        f"فوليوم {vr_text} | جسم الشمعة {body_atr:.2f} ATR | "
        f"امتداد {extension:.2f} ATR"
    )

    # -----------------------------------------------------
    # 1) CLEAN RETEST -> A ENTRY
    # -----------------------------------------------------
    if clean_retest and score >= TREND_A_MIN:
        pl = make_plan(last_close, "RETEST")
        if pl and pl["rr"] >= 2.0:
            res.update(
                tier="A",
                style="TrendRetest",
                plan=pl,
                risk="منخفضة" if score >= 8.0 else "متوسطة",
            )
            return res

    # -----------------------------------------------------
    # 2) STRONG FRESH BREAKOUT -> A ENTRY
    # -----------------------------------------------------
    if (strong_breakout
            and extension <= TREND_MAX_DIRECT_EXTENSION_ATR
            and score >= TREND_A_MIN):
        pl = make_plan(last_close, "BREAKOUT")
        if pl and pl["rr"] >= 2.0:
            res.update(
                tier="A",
                style="TrendBreakout",
                plan=pl,
                risk="متوسطة",
            )
            return res

    # -----------------------------------------------------
    # 3) VALID BREAKOUT -> ARMED FOR RETEST
    #
    # If breakout is valid but not ideal for chasing, wait
    # at the broken level rather than entering late.
    # -----------------------------------------------------
    if (fresh_break or prev_broke) and score >= TREND_B_MIN:
        pl = make_plan(level, "RETEST")
        if pl and pl["rr"] >= 2.0:
            zone = [
                level - TREND_RETEST_TOUCH_ATR * a15,
                level + TREND_RETEST_TOUCH_ATR * a15,
            ]
            res.update(
                tier="ARMED",
                style="TrendBreakRetest",
                plan=pl,
                zone=zone,
                risk="منخفضة" if score >= 7.5 else "متوسطة",
            )
            return res

    # -----------------------------------------------------
    # 4) EARLY WATCH NEAR THE LEVEL
    # -----------------------------------------------------
    if approaching and score >= TREND_B_MIN:
        if s == 1:
            cond = (
                f"إغلاق 15m فوق {fmt(level + break_buffer)}؛ "
                f"إذا الفوليوم قوي يدخل Breakout، وإذا مو قوي ننتظر Retest"
            )
            inv = f"ضعف الاتجاه أو رجوع 1h إلى رينج"
        else:
            cond = (
                f"إغلاق 15m تحت {fmt(level - break_buffer)}؛ "
                f"إذا الفوليوم قوي يدخل Breakout، وإذا مو قوي ننتظر Retest"
            )
            inv = f"ضعف الاتجاه أو رجوع 1h إلى رينج"

        res.update(
            tier="B",
            style="TrendWatch",
            cond=cond,
            inv=inv,
        )
        return res

    return res


def analyze(key, fr, btc_dir):
    # First agent now uses one clear strategy for every market:
    # 1H trend -> 15m breakout/retest.
    # Existing data sources, Telegram, state and paper tracking stay unchanged.
    res = trend_breakout_strat(key, fr, btc_dir)

    if SPOT_ONLY and res["side"] == -1:
        res["tier"] = None

    return res


def load():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            st = json.load(f)
    except FileNotFoundError:
        st = {}
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise RuntimeError(
            f"{STATE_FILE} is unreadable; preserving it instead of "
            "resetting trade state") from e

    if not isinstance(st, dict):
        raise RuntimeError(f"{STATE_FILE} must contain a JSON object")

    defaults = (("active", {}), ("watch", {}), ("cool", {}),
                ("closed", []), ("n", 0))
    for key, value in defaults:
        st.setdefault(key, value)
    for key in ("active", "watch", "cool"):
        if not isinstance(st[key], dict):
            raise RuntimeError(f"{STATE_FILE}: '{key}' must be an object")
    if not isinstance(st["closed"], list):
        raise RuntimeError(f"{STATE_FILE}: 'closed' must be a list")
    if not isinstance(st["n"], int) or isinstance(st["n"], bool):
        raise RuntimeError(f"{STATE_FILE}: 'n' must be an integer")
    return st


def save(st):
    directory = os.path.dirname(os.path.abspath(STATE_FILE))
    fd, tmp_path = tempfile.mkstemp(prefix=".state-", suffix=".tmp",
                                    dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, STATE_FILE)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def realized(sp, exit_r):
    risk0 = abs(sp["entry"] - sp["stop0"]) or 1e-9
    done = sum(abs(sp["tps"][k] - sp["entry"]) / risk0
               for k in range(3) if sp["hits"][k]) / 3
    left = (3 - sum(sp["hits"])) / 3
    return round(done + left * exit_r, 2)


def finish(st, sp, state, exit_r, label, now):
    sp["state"] = state
    sp["r"] = realized(sp, exit_r)
    sp["exit"] = label
    st["closed"].append({
        "id": sp["id"], "key": sp["key"], "side": sp["side"],
        "style": sp["style"], "entry": sp["entry"],
        "stop0": sp["stop0"], "tps": sp["tps"],
        "hits": list(sp["hits"]), "r": sp["r"], "exit": label,
        "tf": sp.get("tf", "15m"), "risk": sp.get("risk"),
        "t_decl": sp["t_decl"], "t_close": now.isoformat()})
    st["cool"][sp["key"]] = now.isoformat()


def trade_line(c):
    h = c.get("hits") or [False, False, False]
    n = sum(h)
    tps = c.get("tps") or []
    reach = ""
    if n and len(tps) >= n:
        mv = abs(tps[n - 1] - c["entry"])
        reach = f"وصل TP{n} ({fmt(tps[n - 1])}) بمسافة {fmt(mv)}"
    if c["exit"] == "TP3":
        what = "حقق TP3 كامل"
    elif c["exit"] == "STOP":
        what = ("ضرب الوقف قبل أي هدف" if n == 0
                else reach + " ثم رجع وضرب الوقف")
    else:
        what = "خروج مبكر" + ("، " + reach if reach else "")
    sd = "LONG" if c["side"] == 1 else "SHORT"
    nm = ASSETS[c["key"]][2]
    if c["r"] <= 0:
        ico = "❌"
    elif c["exit"] == "STOP":
        ico = "🟡"
    else:
        ico = "✅"
    return (f"{ico} {c['id']} | {nm} {sd} | {c['style']} | "
            f"فريم {c.get('tf', '15m')}\n"
            f"   دخول {fmt(c['entry'])} | {what}\n"
            f"   النتيجة: {c['r']:+.2f}R")


def weekly_report(st, now):
    try:
        back = (now.weekday() - 4) % 7
        due = (now.normalize() - pd.Timedelta(days=back)
               + pd.Timedelta(hours=21))
        if due > now:
            due -= pd.Timedelta(days=7)
        last = st.get("last_report")
        if last and pd.Timestamp(last) >= due:
            return
        start = due - pd.Timedelta(days=7)
        wk = [c for c in st["closed"]
              if c.get("t_close")
              and start < pd.Timestamp(c["t_close"]) <= due]
        if not wk:
            st["last_report"] = now.isoformat()
            return
        total = sum(c["r"] for c in wk)
        win = sum(1 for c in wk if c["r"] > 0)
        text = (
            "📋 تقرير الأسبوع (صفقات A المغلقة، تتبع ورقي)\n"
            f"رابحة: {win} | خاسرة: {len(wk) - win} | "
            f"المجموع {total:+.2f}R\n\n"
            + "\n\n".join(trade_line(c) for c in wk)
            + "\n\n⚠️ ورقي افتراضي، مو توصية مالية.")
        t = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text[:4000]}, timeout=30)
        if t.ok:
            st["last_report"] = now.isoformat()
        else:
            print("report telegram error:", t.status_code, t.text)
    except Exception as e:
        print("report error:", e)


def open_setup(st, res, t_last):
    st["n"] += 1
    pl = res["plan"]
    return {
        "id": f"{res['key']}-{st['n']:03d}", "key": res["key"],
        "side": res["side"], "style": res["style"],
        "tf": res.get("tf", "15m"), "risk": res.get("risk"),
        "entry": pl["entry"], "stop": pl["stop"], "stop0": pl["stop"],
        "tps": pl["tps"], "hits": [False, False, False],
        "rr": pl["rr"], "score": res["score"],
        "t0": t_last.isoformat(), "t_decl": t_last.isoformat(),
        "state": "ENTERED" if res["tier"] == "A" else "ARMED",
        "zone": res.get("zone"),
    }


def manage(st, sp, d15, d1, price, now):
    ev = []
    s = sp["side"]
    name = ASSETS[sp["key"]][2]
    pid = sp["id"]
    tf = sp.get("tf", "15m")
    arm_h = {"15m": 4, "1h": 12, "4h": 48}.get(tf, 4)
    hold_h = {"15m": 24, "1h": 72, "4h": 240}.get(tf, 24)
    risk0 = abs(sp["entry"] - sp["stop0"])
    cs = d15[d15.index > pd.Timestamp(sp.get("seen", sp["t0"]))]
    for ts, k in cs.iterrows():
        sp["seen"] = ts.isoformat()
        if market_closed(sp["key"], ts):
            continue
        hi, lo = float(k["h"]), float(k["l"])
        if sp["state"] == "ARMED":
            if (lo <= sp["stop"]) if s == 1 else (hi >= sp["stop"]):
                sp["state"] = "INVALIDATED"
                ev.append(f"❌ {pid} {name}: انلغت، السعر كسر الوقف قبل الدخول")
                return ev
            if lo <= sp["zone"][1] and hi >= sp["zone"][0]:
                sp["state"] = "ENTERED"
                sp["t0"] = ts.isoformat()
                ev.append(f"📍 {pid} {name}: لمس منطقة الدخول، "
                          f"دخول ورقي @ {fmt(sp['entry'])}")
            continue
        if (lo <= sp["stop"]) if s == 1 else (hi >= sp["stop"]):
            er = (sp["stop"] - sp["entry"]) * s / risk0
            finish(st, sp, "CLOSED", er, "STOP", now)
            ev.append(f"🛑 {pid} {name}: STOP HIT | R ورقي {sp['r']}")
            return ev
        for i in range(3):
            if sp["hits"][i]:
                continue
            target_hit = ((hi >= sp["tps"][i]) if s == 1
                          else (lo <= sp["tps"][i]))
            if not target_hit:
                break

            sp["hits"][i] = True
            ev.append(f"🎯 {pid} {name}: TP{i + 1} HIT "
                      f"({fmt(sp['tps'][i])})")
            if i == 2:
                finish(st, sp, "CLOSED", 0, "TP3", now)
                ev.append(f"✅ {pid}: إغلاق كامل | R ورقي {sp['r']}")
                return ev

            sp["stop"] = sp["entry"] if i == 0 else sp["tps"][0]
            ev.append(f"🔒 {pid}: MOVE SL → {fmt(sp['stop'])}")

            # شموع OHLC ما تبين ترتيب الحركة داخل الشمعة، فنحسب هدف واحد
            # لكل شمعة، وإذا الوقف المتحرك انلمس بنفس الشمعة نقفل الباقي.
            moved_stop_hit = ((lo <= sp["stop"]) if s == 1
                              else (hi >= sp["stop"]))
            if moved_stop_hit:
                er = (sp["stop"] - sp["entry"]) * s / risk0
                finish(st, sp, "CLOSED", er, "STOP", now)
                ev.append(f"🛑 {pid} {name}: الوقف المتحرك انلمس بنفس الشمعة "
                          f"| R ورقي {sp['r']}")
                return ev
            break
    if sp["state"] == "ARMED":
        if now - pd.Timestamp(sp["t_decl"]) > pd.Timedelta(hours=arm_h):
            sp["state"] = "EXPIRED"
            ev.append(f"⌛ {pid} {name}: انتهت بدون دخول")
        return ev
    if sp["state"] == "ENTERED":
        age = now - pd.Timestamp(sp["t0"])
        lim = pd.Timedelta(hours=hold_h * (2 if sp["hits"][0] else 1))
        flip = (tf == "15m" and not str(sp["style"]).startswith("Z")
                and trend(d1) == -s and not sp["hits"][0])
        if age > lim or flip:
            er = (price - sp["entry"]) * s / risk0
            finish(st, sp, "CLOSED", er, "EARLY EXIT", now)
            why = "الترند انقلب" if flip else "انتهى الوقت"
            ev.append(f"🚪 {pid} {name}: EARLY EXIT @ {fmt(price)} "
                      f"({why}) | R ورقي {sp['r']}")
    return ev


def stats(cl):
    def line(x, nm):
        rs = [c["r"] for c in x]
        n = len(rs)
        w = sum(r for r in rs if r > 0)
        lo = -sum(r for r in rs if r <= 0)
        pf = f"{w / lo:.2f}" if lo > 0 else "∞"
        eq = pk = dd = 0.0
        for r in rs:
            eq += r
            pk = max(pk, eq)
            dd = max(dd, pk - eq)
        return (f"{nm}: {n} صفقة | نجاح {sum(r > 0 for r in rs) / n * 100:.0f}%"
                f" | متوسط R {sum(rs) / n:.2f} | PF {pf} | هبوط أقصى {dd:.1f}R")

    if len(cl) < 10:
        return f"📈 الأداء الورقي: {len(cl)} صفقة فقط، عينة قليلة للحكم."
    out = [line(cl, "📈 الكل")]
    for f in ("style", "key"):
        for g in sorted({c[f] for c in cl}):
            sub = [c for c in cl if c[f] == g]
            if len(sub) >= 10:
                out.append(line(sub, g))
    return "\n".join(out) + "\nعينة صغيرة، لا تعتبرها دليل تفوق."


def msg_setup(res, sp, name):
    side = "LONG" if sp["side"] == 1 else "SHORT"
    head = ("🟢 A ENTRY (ورقي)" if sp["state"] == "ENTERED"
            else "🟠 A ARMED (ينتظر ريتست)")
    zone = ""
    if sp["zone"]:
        zone = (f"منطقة الدخول: {fmt(sp['zone'][0])} - "
                f"{fmt(sp['zone'][1])}\n")
    return (
        f"{head} | {name} {side}\n"
        f"ID: {sp['id']} | نوع: {sp['style']} | "
        f"فريم الدخول: {sp.get('tf', '15m')}\n"
        f"السعر الآن: {fmt(res['price'])}\n{zone}"
        f"دخول: {fmt(sp['entry'])} | وقف: {fmt(sp['stop'])}\n"
        f"TP1 {fmt(sp['tps'][0])} | TP2 {fmt(sp['tps'][1])} | "
        f"TP3 {fmt(sp['tps'][2])}\n"
        f"R:R تقريباً {sp['rr']} | سكور {sp['score']}/10\n"
        + (f"درجة المخاطرة: {res['risk']}\n" if res.get("risk") else "")
        + f"{res['obs']}\n"
        f"الإلغاء: {'تحت' if sp['side'] == 1 else 'فوق'} {fmt(sp['stop'])}\n"
        f"(تتبع ورقي افتراضي، مو دخول فعلي)")


def msg_watch(res, name):
    side = "LONG" if res["side"] == 1 else "SHORT"
    return (f"🟡 B+ مراقبة | {name} {side} (مو دخول لسا)\n"
            f"السعر: {fmt(res['price'])} | سكور {res['score']}/10\n"
            f"التأكيد المطلوب: {res['cond']}\n"
            f"الإلغاء: {res['inv']}\n{res['obs']}")


def msg_early(res, name):
    side = "LONG" if res["side"] == 1 else "SHORT"
    pl = res["plan"]
    return (
        f"🟣 تنبيه مبكر (بدون تأكيد) | {name} {side}\n"
        f"السعر: {fmt(res['price'])} | وصل {res.get('zone_txt', '')}\n"
        f"درجة المخاطرة: {res['risk']} (جودة {res['score']}/10)\n"
        f"دخول مقترح: {fmt(pl['entry'])} | وقف: {fmt(pl['stop'])}\n"
        f"TP1 {fmt(pl['tps'][0])} | TP2 {fmt(pl['tps'][1])} | "
        f"TP3 {fmt(pl['tps'][2])}\n"
        f"للتأكيد: {res['cond']}\n"
        f"{res['obs']}\n"
        "القرار لك. ما يتتبع كصفقة ورقية إلا بعد التأكيد.")


def news_risk(res, nw):
    try:
        if res.get("risk") and nw.get("blackout"):
            res["risk"] = RISK_UP.get(res["risk"], res["risk"])
            res["obs"] += "\nدرجة المخاطرة ارتفعت بسبب خبر قوي قريب"
    except Exception as e:
        print("news risk error:", e)
    return res


def main():
    st = load()
    now = pd.Timestamp.now(tz="UTC")
    data = {}
    for key, (kind, sym, name) in ASSETS.items():
        try:
            fr = frames(kind, sym)
            if now - fr["15m"].index[-1] > pd.Timedelta(hours=3):
                print(key, "market closed")
                continue
            data[key] = fr
        except Exception as e:
            print(key, "data error:", e)

    btc_dir = None
    if "BTC" in data:
        a = trend(data["BTC"]["4h"])
        b = trend(data["BTC"]["1h"])
        btc_dir = a if a == b else 0

    nw = news.get_news(st, now, GEMINI_KEY)
    events = []
    summary = []
    for key, fr in data.items():
        name = ASSETS[key][2]
        d15, d1 = fr["15m"], fr["1h"]
        price = float(d15["c"].iloc[-1])
        wk = f"{key}:E"
        w = st["watch"].get(wk)
        if w:
            if ((price <= w["stop"]) if w["side"] == 1
                    else (price >= w["stop"])):
                events.append(f"⚠️ {name}: انلغى التنبيه المبكر، السعر "
                              f"تجاوز {fmt(w['stop'])}")
                st["watch"].pop(wk, None)
            elif now - pd.Timestamp(w["ts"]) > pd.Timedelta(hours=12):
                st["watch"].pop(wk, None)
        sp = st["active"].get(key)
        if sp:
            events += manage(st, sp, d15, d1, price, now)
            if sp["state"] in ("CLOSED", "INVALIDATED", "EXPIRED"):
                if sp["state"] == "CLOSED":
                    events.append(stats(st["closed"]))
                del st["active"][key]
            else:
                summary.append(f"{name}: صفقة ورقية {sp['state']}")
            continue
        if market_closed(key, now):
            summary.append(f"{name}: السوق مسكر")
            continue
        cool = st["cool"].get(key)
        if cool and now - pd.Timestamp(cool) < pd.Timedelta(hours=2):
            summary.append(f"{name}: فترة انتظار بعد إغلاق")
            continue
        try:
            res = analyze(key, fr, btc_dir)
        except Exception as e:
            print(key, "analyze error:", e)
            continue
        res = news.apply(res, nw)
        res = news_risk(res, nw)
        sd = "LONG" if res["side"] == 1 else "SHORT"
        summary.append(f"{name}: {sd} سكور {res['score']}/10 "
                       f"| فريم {res.get('tf', '15m')} "
                       f"| سعر {fmt(res['price'])}")
        if res["tier"] in ("A", "ARMED"):
            pe = st["watch"].pop(f"{key}:E", None)
            if (pe and pe.get("side") == res["side"] and
                    now - pd.Timestamp(pe["ts"]) < pd.Timedelta(hours=12)):
                res["obs"] += "\n✅ تأكيد للتنبيه المبكر السابق"
            sp = open_setup(st, res, d15.index[-1])
            st["active"][key] = sp
            st["watch"].pop(key, None)
            events.append(msg_setup(res, sp, name))
        elif res["tier"] == "EARLY":
            wk = f"{key}:E"
            pw = st["watch"].get(wk)
            zp = res.get("zone_p", 0.0)
            fresh = (pw and pw.get("side") == res["side"]
                     and abs(pw.get("zone", 0.0) - zp)
                     <= 0.5 * res.get("a1", 0.0)
                     and now - pd.Timestamp(pw["ts"])
                     < pd.Timedelta(hours=6))
            if not fresh:
                st["watch"][wk] = {"side": res["side"], "zone": zp,
                                   "stop": res["plan"]["stop"],
                                   "ts": now.isoformat()}
                events.append(msg_early(res, name))
        elif res["tier"] == "B":
            w = st["watch"].get(key)
            fresh = (w and w["side"] == res["side"] and
                     now - pd.Timestamp(w["ts"]) < pd.Timedelta(hours=6))
            if not fresh:
                st["watch"][key] = {"side": res["side"],
                                    "ts": now.isoformat()}
                events.append(msg_watch(res, name))
        elif res["score"] < 5.5:
            st["watch"].pop(key, None)

    events += news.alerts(nw, st, now)
    summary.append(nw["status"])
    if not events and EVENT == "workflow_dispatch":
        events.append("ℹ️ فحص يدوي، ما في حدث جديد:\n" + "\n".join(summary))

    if events:
        text = "\n\n".join(events)
        try:
            url = ("https://generativelanguage.googleapis.com/v1beta/"
                   "models/gemini-flash-latest:generateContent")
            pr = ("علّق بالعراقي بجملتين بالكثير على هالتنبيهات. "
                  "لا تغيّر الأرقام ولا تخترع أرقام ولا تضمن ربح:\n" + text)
            r = requests.post(
                url,
                headers={"x-goog-api-key": GEMINI_KEY},
                json={"contents": [{"parts": [{"text": pr}]}]},
                timeout=60,
            )
            j = r.json()
            if "candidates" in j:
                text += ("\n\n💬 " +
                         j["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:
            print("gemini error:", e)
        text += "\n\n⚠️ تحليل آلي وتتبع ورقي، مو توصية مالية."
        t = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text[:4000]}, timeout=30)
        if not t.ok:
            print("Telegram error:", t.status_code, t.text)
            raise SystemExit(1)
        print("sent")
    else:
        print("no events")
    weekly_report(st, now)
    save(st)


main()
