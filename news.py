import json
import re
import requests
import xml.etree.ElementTree as ET
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0"}
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://news.google.com/rss/search?q=Federal+Reserve+OR+inflation"
    "+OR+gold+price+OR+Nasdaq+OR+tariffs&hl=en-US&gl=US&ceid=US:en",
]
GEM_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "gemini-flash-latest:generateContent")
BEFORE = pd.Timedelta(minutes=30)
AFTER = pd.Timedelta(minutes=45)
CAL_TTL = pd.Timedelta(hours=6)
BIAS_TTL = pd.Timedelta(hours=3)
BIAS_MAX_AGE = pd.Timedelta(hours=9)
RETRY = pd.Timedelta(minutes=30)
BIAS_FILTER = True
GROUP = {"BTC": "crypto", "ETH": "crypto", "SOL": "crypto",
         "GOLD": "gold", "NDX": "nasdaq"}
LAB = {1: "صاعد", -1: "هابط", 0: "محايد"}


def due(c, k, ttl, now):
    ts = c.get(k)
    return (not ts) or now - pd.Timestamp(ts) > ttl


def hm(m):
    m = int(m)
    if m >= 60:
        return f"{m // 60} س {m % 60} د"
    return f"{m} د"


def fetch_calendar():
    r = requests.get(CAL_URL, headers=UA, timeout=20)
    r.raise_for_status()
    out = []
    for e in r.json():
        if e.get("country") != "USD" or e.get("impact") != "High":
            continue
        t = pd.Timestamp(e["date"])
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        out.append({"t": t.isoformat(), "title": e.get("title", ""),
                    "fc": e.get("forecast") or "",
                    "prev": e.get("previous") or ""})
    return out


def fetch_headlines():
    titles = []
    for u in FEEDS:
        try:
            r = requests.get(u, headers=UA, timeout=15)
            root = ET.fromstring(r.content)
            n = 0
            for it in root.iter("item"):
                t = it.findtext("title")
                if t:
                    titles.append(t.strip())
                    n += 1
                if n >= 15:
                    break
        except Exception as e:
            print("feed error:", u[:40], e)
    return titles[:45]


def ask_bias(titles, gem_key):
    pr = ("هذي عناوين أخبار حديثة:\n"
          + "\n".join("- " + t for t in titles)
          + "\n\nحدد ميل الأخبار للساعات القادمة لكل مجموعة: "
          "crypto (بيتكوين وإيثريوم وسولانا) و gold (الذهب) و "
          "nasdaq (ناسداك). القيم: 1 صاعد، -1 هابط، 0 محايد أو غير واضح. "
          "لا تعطِ 1 أو -1 إلا إذا العناوين تدل بوضوح. "
          "أرجع JSON فقط بهذا الشكل: "
          '{"crypto":0,"gold":0,"nasdaq":0,"why":"جملة قصيرة بالعراقي"}')
    r = requests.post(
        GEM_URL, params={"key": gem_key},
        json={"contents": [{"parts": [{"text": pr}]}],
              "generationConfig": {"responseMimeType": "application/json"}},
        timeout=60)
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    d = json.loads(re.sub(r"```json|```", "", txt).strip())
    bias = {}
    for g in ("crypto", "gold", "nasdaq"):
        v = int(d.get(g, 0))
        bias[g] = 1 if v > 0 else (-1 if v < 0 else 0)
    return bias, str(d.get("why", ""))[:160]


def get_news(st, now, gem_key):
    nw = {"blackout": False, "bias": {}, "events": [],
          "status": "الأخبار: غير متوفرة"}
    try:
        c = st.setdefault("news", {})
        if due(c, "cal_try", CAL_TTL, now):
            try:
                c["cal"] = fetch_calendar()
                c["cal_try"] = now.isoformat()
            except Exception as e:
                print("calendar error:", e)
                c["cal_try"] = (now - CAL_TTL + RETRY).isoformat()
        if due(c, "bias_try", BIAS_TTL, now):
            try:
                tl = fetch_headlines()
                if not tl:
                    raise RuntimeError("no headlines")
                c["bias"], c["why"] = ask_bias(tl, gem_key)
                c["bias_ts"] = now.isoformat()
                c["bias_try"] = now.isoformat()
            except Exception as e:
                print("bias error:", e)
                c["bias_try"] = (now - BIAS_TTL + RETRY).isoformat()

        evs = sorted(((pd.Timestamp(e["t"]), e)
                      for e in c.get("cal") or []),
                     key=lambda x: x[0])
        nw["events"] = evs
        cur = None
        for t, e in evs:
            if t - BEFORE <= now <= t + AFTER:
                nw["blackout"] = True
                cur = e
        nxt = next(((t, e) for t, e in evs if t > now), None)

        parts = []
        if not c.get("cal"):
            parts.append("التقويم غير متوفر (ما في تنبيه أخبار)")
        elif cur:
            parts.append(f"⚠️ خبر قوي الحين ({cur['title']}): "
                         "الإشارات شغالة، انتبه للتذبذب")
        elif nxt and nxt[0] - now < pd.Timedelta(hours=24):
            m = (nxt[0] - now).total_seconds() / 60
            parts.append(f"أقرب خبر قوي: {nxt[1]['title']} بعد {hm(m)}")
        else:
            parts.append("ما في خبر دولار قوي خلال 24 ساعة")

        bts = c.get("bias_ts")
        if bts and c.get("bias") and now - pd.Timestamp(bts) <= BIAS_MAX_AGE:
            b = c["bias"]
            nw["bias"] = b
            parts.append("ميل الأخبار (تقديري): كريبتو "
                         + LAB[b.get("crypto", 0)]
                         + " | ذهب " + LAB[b.get("gold", 0)]
                         + " | ناسداك " + LAB[b.get("nasdaq", 0)])
            if c.get("why"):
                parts.append(c["why"])
        nw["status"] = "الأخبار: " + " | ".join(parts)
    except Exception as e:
        print("news error:", e)
    return nw


def apply(res, nw):
    try:
        if res.get("tier") is None:
            return res
        if nw["blackout"]:
            res["obs"] += "\n⚠️ خبر قوي قريب أو جاري، توقع تذبذب"
        b = nw["bias"].get(GROUP.get(res["key"]), 0)
        s = res["side"]
        if b == s:
            res["obs"] += "\nالأخبار: ميلها يدعم الاتجاه (تقديري)"
        elif b == -s:
            res["obs"] += "\nالأخبار: ميلها عكس الاتجاه (تقديري)"
            if BIAS_FILTER and res["tier"] in ("A", "ARMED"):
                stp = res["plan"]["stop"]
                res["tier"] = "B"
                res["cond"] = "تأكيد من الأخبار أو انقلاب ميلها لصالح الاتجاه"
                res["inv"] = ("كسر " if s == 1 else "اختراق ") + f"{stp:,.2f}"
    except Exception as e:
        print("news apply error:", e)
    return res


def alerts(nw, st, now):
    out = []
    try:
        c = st.setdefault("news", {})
        sent = c.setdefault("sent", [])
        for t, e in nw["events"]:
            mins = (t - now).total_seconds() / 60
            eid = e["title"] + e["t"]
            if 0 < mins <= 60 and eid not in sent:
                sent.append(eid)
                extra = ""
                if e["fc"] or e["prev"]:
                    extra = (f"\nالتوقع: {e['fc'] or '-'} | "
                             f"السابق: {e['prev'] or '-'}")
                out.append(
                    f"⚠️ خبر قوي (USD) بعد {int(mins)} دقيقة: "
                    f"{e['title']}{extra}\n"
                    "الإشارات والصفقات تستمر عادي، بس توقع تذبذب قوي "
                    "وحركة سريعة بالوقف.")
        c["sent"] = sent[-30:]
    except Exception as e:
        print("news alerts error:", e)
    return out
