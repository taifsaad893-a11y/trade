import os
import json
import requests
import pandas as pd
import yfinance as yf

GEMINI_KEY = os.environ["GEMINI_KEY"]
TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT = os.environ["TG_CHAT"]
EVENT = os.environ.get("EVENT", "")
STATE_FILE = "state.json"

ASSETS = {
    "BTC": ("bin", "BTCUSDT", "بيتكوين"),
    "ETH": ("bin", "ETHUSDT", "إيثريوم"),
    "SOL": ("bin", "SOLUSDT", "سولانا"),
    "GOLD": ("yf", "GC=F", "الذهب"),
    "NDX": ("yf", "^NDX", "ناسداك"),
}
TR = {1: "صاعد", -1: "هابط", 0: "عرضي"}


def fmt(x):
    return f"{x:,.2f}" if abs(x) >= 10 else f"{x:.4f}"


def bin_klines(sym, tf, n=300):
    for host in ("https://data-api.binance.vision",
                 "https://api.binance.com"):
        try:
            r = requests.get(
                host + "/api/v3/klines",
                params={"symbol": sym, "interval": tf, "limit": n},
                timeout=20)
            if r.ok:
                d = pd.DataFrame(r.json()).iloc[:, :6]
                d.columns = ["t", "o", "h", "l", "c", "v"]
                d = d.astype(float)
                d.index = pd.to_datetime(d["t"], unit="ms", utc=True)
                return d.drop(columns="t").iloc[:-1]
        except Exception:
            pass
    raise RuntimeError("binance failed " + sym)


def yf_frame(sym, period, interval):
    d = yf.download(sym, period=period, interval=interval, progress=False)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    d = d.rename(columns={"Open": "o", "High": "h", "Low": "l",
                          "Close": "c", "Volume": "v"})
    d = d[["o", "h", "l", "c", "v"]].dropna()
    d.index = pd.to_datetime(d.index, utc=True)
    return d.iloc[:-1]


def frames(kind, sym):
    if kind == "bin":
        return {tf: bin_klines(sym, tf) for tf in ("15m", "1h", "4h")}
    f15 = yf_frame(sym, "5d", "15m")
    f1h = yf_frame(sym, "60d", "1h")
    f4h = f1h.resample("4h").agg(
        {"o": "first", "h": "max", "l": "min",
         "c": "last", "v": "sum"}).dropna().iloc[:-1]
    return {"15m": f15, "1h": f1h, "4h": f4h}


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


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
def analyze(key, fr, btc_dir):
    d15, d1, d4 = fr["15m"], fr["1h"], fr["4h"]
    c = d15["c"]
    price = float(c.iloc[-1])
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

    def plan(s, ep):
        if s == 1:
            stop = min(float(d15["l"].tail(12).min()), ep - a15) - 0.2 * a15
            risk = ep - stop
            lvl = hi1
        else:
            stop = max(float(d15["h"].tail(12).max()), ep + a15) + 0.2 * a15
            risk = stop - ep
            lvl = lo1
        if risk <= 0 or risk > 3 * a1:
            return None
        room = (lvl - ep) * s
        rr = min(3.0, room / risk) if room > 0 else 3.0
        tps = [ep + s * risk * k for k in (1, 2, 3)]
        return {"entry": ep, "stop": stop, "tps": tps, "rr": round(rr, 1)}

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
        else:
            sc += 0.5
        sc += 0.5 if (price > e200) == (s == 1) else 0
        if btc_dir is not None and key in ("ETH", "SOL"):
            sc += 1 if btc_dir == s else (-1.5 if btc_dir == -s else 0)
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
    style = "Breakout" if brk else ("Retest" if rt else None)
    obs = (f"مرصود: 4h {TR[t4]} | 1h {TR[t1]} | RSI15 {r15:.0f} | "
           f"MACD15 {'+' if mh > 0 else '-'}")
    if vr is not None:
        obs += f" | فوليوم {vr:.1f}x"
    obs += f" | امتداد {ext:.1f} ATR"
    notes = ict(d15, d1, s, a15)[1]
    if notes:
        obs += "\nICT: " + " | ".join(notes)
    res = {"key": key, "side": s, "score": sc, "price": price,
           "obs": obs, "tier": None, "style": style, "ext": round(ext, 1)}

    pl = plan(s, price) if style else None
    if style and pl and sc >= 7.5 and pl["rr"] >= 2:
        res.update(tier="A", plan=pl)
        return res

    if brk and sc >= 7.5 and ext > 0.5:
        pa = plan(s, lvl)
        if pa and pa["rr"] >= 2:
            res.update(tier="ARMED", plan=pa, style="Retest",
                       zone=[lvl - 0.3 * a15, lvl + 0.3 * a15])
            return res

    if sc >= 6:
        if s == 1:
            sw = float(d15["l"].tail(12).min())
            cond = (f"ريتست {fmt(lvl)} وثبات فوقه" if brk
                    else f"إغلاق شمعة 15m فوق {fmt(lvl)} مع فوليوم")
            inv = f"كسر {fmt(sw)}"
        else:
            sw = float(d15["h"].tail(12).max())
            cond = (f"ريتست {fmt(lvl)} ورفض تحته" if brk
                    else f"إغلاق شمعة 15m تحت {fmt(lvl)} مع فوليوم")
            inv = f"اختراق {fmt(sw)}"
        res.update(tier="B", cond=cond, inv=inv)
    return res
def load():
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
    except Exception:
        st = {}
    for k, v in (("active", {}), ("watch", {}), ("cool", {}),
                 ("closed", []), ("n", 0)):
        st.setdefault(k, v)
    return st


def save(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, ensure_ascii=False, indent=1)


def realized(sp, exit_r):
    done = sum(k + 1 for k in range(3) if sp["hits"][k]) / 3
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
    return (f"{ico} {c['id']} | {nm} {sd} | {c['style']}\n"
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
    risk0 = abs(sp["entry"] - sp["stop0"])
    cs = d15[d15.index > pd.Timestamp(sp["t0"])]
    for ts, k in cs.iterrows():
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
            if (hi >= sp["tps"][i]) if s == 1 else (lo <= sp["tps"][i]):
                sp["hits"][i] = True
                ev.append(f"🎯 {pid} {name}: TP{i + 1} HIT "
                          f"({fmt(sp['tps'][i])})")
                if i == 2:
                    finish(st, sp, "CLOSED", 0, "TP3", now)
                    ev.append(f"✅ {pid}: إغلاق كامل | R ورقي {sp['r']}")
                    return ev
                sp["stop"] = sp["entry"] if i == 0 else sp["tps"][0]
                ev.append(f"🔒 {pid}: MOVE SL → {fmt(sp['stop'])}")
            else:
                break
    if sp["state"] == "ARMED":
        if now - pd.Timestamp(sp["t_decl"]) > pd.Timedelta(hours=4):
            sp["state"] = "EXPIRED"
            ev.append(f"⌛ {pid} {name}: انتهت بدون دخول")
        return ev
    if sp["state"] == "ENTERED":
        age = now - pd.Timestamp(sp["t0"])
        lim = pd.Timedelta(hours=48 if sp["hits"][0] else 24)
        flip = trend(d1) == -s and not sp["hits"][0]
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
        f"ID: {sp['id']} | نوع: {sp['style']}\n"
        f"السعر الآن: {fmt(res['price'])}\n{zone}"
        f"دخول: {fmt(sp['entry'])} | وقف: {fmt(sp['stop'])}\n"
        f"TP1 {fmt(sp['tps'][0])} | TP2 {fmt(sp['tps'][1])} | "
        f"TP3 {fmt(sp['tps'][2])}\n"
        f"R:R تقريباً {sp['rr']} | سكور {sp['score']}/10\n"
        f"{res['obs']}\n"
        f"الإلغاء: {'تحت' if sp['side'] == 1 else 'فوق'} {fmt(sp['stop'])}\n"
        f"(تتبع ورقي افتراضي، مو دخول فعلي)")


def msg_watch(res, name):
    side = "LONG" if res["side"] == 1 else "SHORT"
    return (f"🟡 B+ مراقبة | {name} {side} (مو دخول لسا)\n"
            f"السعر: {fmt(res['price'])} | سكور {res['score']}/10\n"
            f"التأكيد المطلوب: {res['cond']}\n"
            f"الإلغاء: {res['inv']}\n{res['obs']}")


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

    events = []
    summary = []
    for key, fr in data.items():
        name = ASSETS[key][2]
        d15, d1 = fr["15m"], fr["1h"]
        price = float(d15["c"].iloc[-1])
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
        cool = st["cool"].get(key)
        if cool and now - pd.Timestamp(cool) < pd.Timedelta(hours=2):
            summary.append(f"{name}: فترة انتظار بعد إغلاق")
            continue
        try:
            res = analyze(key, fr, btc_dir)
        except Exception as e:
            print(key, "analyze error:", e)
            continue
        sd = "LONG" if res["side"] == 1 else "SHORT"
        summary.append(f"{name}: {sd} سكور {res['score']}/10")
        if res["tier"] in ("A", "ARMED"):
            sp = open_setup(st, res, d15.index[-1])
            st["active"][key] = sp
            st["watch"].pop(key, None)
            events.append(msg_setup(res, sp, name))
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
                url, params={"key": GEMINI_KEY},
                json={"contents": [{"parts": [{"text": pr}]}]}, timeout=60)
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
