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
    o, h, l, c, v = (df["Open"], df["High"], df["Low"],
                     df["Close"], df["Volume"])

    prev = c.shift(1)
    tr = (h - l).combine((h - prev).abs(), max).combine((l - prev).abs(), max)
    atr = float(tr.rolling(14).mean().iloc[-1])

    price = float(c.iloc[-1])
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma50 = float(c.rolling(50).mean().iloc[-1])
    r = float(rsi(c).iloc[-1])

    # دعم ومقاومة: قريب (20 يوم) وأقوى (60 يوم)
    sup = float(l.tail(20).min())
    res = float(h.tail(20).max())
    sup_major = float(l.tail(60).min())
    res_major = float(h.tail
