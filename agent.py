import os
import yfinance as yf
import requests

SYMBOLS = {
    "GC=F": "الذهب",
    "BTC-USD": "بيتكوين",
    "SOL-USD": "سولانا",
    "^NDX": "ناسداك 100",
}
GEMINI_KEY = os.environ["GEMINI_KEY"]
TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT = os.environ["TG_CHAT"]


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn)


def analyze(sym, name):
    df = yf.download(sym, period="6mo", interval="1d", progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    v = df["Volume"]

    prev = c.shift(1)
    tr = (h - l).combine((h - prev).abs(), max)
    tr = tr.combine((l - prev).abs(), max)
    atr = float(tr.rolling(14).mean().iloc[-1])

    price = float(c.iloc[-1])
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma50 = float(c.rolling(50).mean().iloc[-1])
    r = float(rsi(c).iloc[-1])

    sup = float(l.tail(20).min())
    res = float(h.tail(20).max())
    sup_major = float(l.tail(60).min())
    res_major = float(h.tail(60).max())

    if price > ma20 > ma50:
        trend = "صاعد"
    elif price < ma20 < ma50:
        trend = "هابط"
    else:
        trend = "عرضي/غير واضح"
    rng = (h - l).replace(0, float("nan"))
    pos = ((c - l) - (h - c)) / rng
    vol_ok = float(v.tail(20).sum()) > 0
    if vol_ok:
        delta = float((pos * v).tail(10).sum())
        vol_rel = float(v.iloc[-1] / v.tail(20).mean())
        flow_side = "شراء" if delta > 0 else "بيع"
        flow_txt = f"ضغط {flow_side} | فوليوم {vol_rel:.1f}x من المتوسط"
    else:
        flow_side = None
        flow_txt = "الفوليوم غير متوفر"

    head = (
        f"{name} ({sym}) | سعر {price:.2f}\n"
        f"ترند: {trend} | RSI {r:.0f}\n"
        f"دعم {sup:.2f} (أقوى {sup_major:.2f}) | "
        f"مقاومة {res:.2f} (أقوى {res_major:.2f})\n"
        f"Order flow (تقدير): {flow_txt}"
    )

    long_ok = trend == "صاعد" and r < 70 and flow_side in ("شراء", None)
    short_ok = trend == "هابط" and r > 30 and flow_side in ("بيع", None)

    if long_ok:
        stop = max(price - 1.5 * atr, sup - 0.2 * atr)
        target = price + 2 * (price - stop)
        side = "شراء"
    elif short_ok:
        stop = min(price + 1.5 * atr, res + 0.2 * atr)
        target = price - 2 * (stop - price)
        side = "بيع"
    else:
        return head + "\nالقرار: لا توجد صفقة واضحة، انتظر"

    rr = abs(target - price) / abs(price - stop)
    return head + (
        f"\nسيناريو {side}: دخول {price:.2f} | "
        f"وقف {stop:.2f} | هدف {target:.2f} | R:R {rr:.1f}"
    )
parts = []
for s, n in SYMBOLS.items():
    try:
        parts.append(analyze(s, n))
    except Exception as e:
        parts.append(f"{n} ({s}): تعذر التحليل ({e})")
data = "\n\n".join(parts)

prompt = (
    "هذي مستويات وتحليل محسوب آلياً:\n"
    + data
    + "\n\nعلّق باختصار على كل أصل: قوة الترند، وهل الـ order flow يدعمه، "
    "وأهم مستوى يراقبه المتداول. لا تغيّر الأرقام ولا تخترع أرقام جديدة، "
    "ولا تضمن نجاح أي صفقة."
)

url = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-flash-latest:generateContent"
)
resp = requests.post(
    url,
    params={"key": GEMINI_KEY},
    json={"contents": [{"parts": [{"text": prompt}]}]},
    timeout=60,
)

j = resp.json()
comment = ""
if "candidates" in j:
    comment = j["candidates"][0]["content"]["parts"][0]["text"]
else:
    print("Gemini error:", resp.status_code, j)

msg = (
    "📊 تحليل السوق\n\n"
    + data
    + "\n\n"
    + comment
    + "\n\n⚠️ تحليل آلي للمتابعة فقط، مو توصية مالية."
)

t = requests.post(
    f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
    json={"chat_id": TG_CHAT, "text": msg[:4000]},
    timeout=30,
)
if not t.ok:
    print("Telegram error:", t.status_code, t.text)
    raise SystemExit(1)
print("OK, sent to Telegram")
