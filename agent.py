import os
import yfinance as yf
import requests

SYMBOLS = ["AAPL", "TSLA", "BTC-USD", "^GSPC"]
GEMINI_KEY = os.environ["GEMINI_KEY"]
TG_TOKEN = os.environ["TG_TOKEN"]
TG_CHAT = os.environ["TG_CHAT"]

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn)

def analyze(sym):
    df = yf.download(sym, period="3mo", interval="1d", progress=False)
    c = df["Close"].squeeze()
    ma = c.rolling(20).mean().iloc[-1]
    return (f"{sym}: سعر {c.iloc[-1]:.2f}, "
            f"تغير {c.pct_change().iloc[-1]*100:.2f}%, "
            f"RSI {rsi(c).iloc[-1]:.0f}, "
            f"{'فوق' if c.iloc[-1] > ma else 'تحت'} MA20")

data = "\n".join(analyze(s) for s in SYMBOLS)

prompt = f"""بيانات السوق الحالية:
{data}

حللها باختصار وأعطني 3-5 تاسكات عملية للساعة الجاية (مراجعة، مراقبة، انتظار).
لا تعطِ نصائح شراء أو بيع مباشرة."""

r = requests.post(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
    params={"key": GEMINI_KEY},
    json={"contents": [{"parts": [{"text": prompt}]}]},
    timeout=60)
text = r.json()["candidates"][0]["content"]["parts"][0]["text"]

requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
              json={"chat_id": TG_CHAT, "text": text}, timeout=30)
