import math
import os

import pandas as pd


STRATEGY_VERSION = "sweep_mss_idm_demand_supply_v2"
LOOKBACK = 48
SWING = 2
SWEEP_BUFFER_ATR = 0.05
MSS_BUFFER_ATR = 0.03
ZONE_PAD_ATR = 0.08
STOP_BUFFER_ATR = 0.12
MAX_ZONE_DISTANCE_ATR = 1.25
MIN_RR = 2.0


def _direction_name(side):
    return "LONG" if side == 1 else "SHORT"


def _trend_label(value):
    if value == 1:
        return "BULLISH"
    if value == -1:
        return "BEARISH"
    return "RANGE"


def _safe_trend(a, frame):
    try:
        if frame is None or len(frame) < 20:
            return 0
        value = int(a.trend(frame))
        return value if value in (-1, 0, 1) else 0
    except Exception:
        return 0


def _base_result(key, side, price, score, style, obs):
    return {
        "key": key, "side": int(side), "price": float(price),
        "score": round(float(score), 1), "style": style, "tf": "15m",
        "tier": None, "ext": 0.0, "miss": [], "obs": obs,
        "strategy_version": STRATEGY_VERSION,
    }


def _cost_reason(key, plan):
    try:
        bps = float(os.environ.get("ROUNDTRIP_COST_BPS_" + key, os.environ.get("ROUNDTRIP_COST_BPS", "14")))
    except (TypeError, ValueError):
        return "افتراض الكلفة غير صالح"
    risk = abs(plan["entry"] - plan["stop"])
    if not math.isfinite(bps) or bps < 0 or risk <= 0:
        return "خطة المخاطرة غير صالحة"
    cost_r = plan["entry"] * bps / 10000 / risk
    if cost_r > 0.35:
        return f"الكلفة المفترضة {cost_r:.2f}R تتجاوز 0.35R"
    return None


def _swings(d):
    highs, lows = [], []
    start = max(SWING, len(d) - LOOKBACK)
    for i in range(start, len(d) - SWING):
        h = float(d["h"].iloc[i]); l = float(d["l"].iloc[i])
        if h >= float(d["h"].iloc[i-SWING:i+SWING+1].max()): highs.append((i, h))
        if l <= float(d["l"].iloc[i-SWING:i+SWING+1].min()): lows.append((i, l))
    return highs, lows


def _find_setup(d15, a15):
    highs, lows = _swings(d15)
    if not highs or not lows:
        return None
    start = max(3, len(d15) - 14)
    for sweep_i in range(len(d15) - 2, start - 1, -1):
        sh = [(i, p) for i, p in highs if i < sweep_i]
        sl = [(i, p) for i, p in lows if i < sweep_i]
        if not sh or not sl: continue
        prior_high_i, prior_high = sh[-1]; prior_low_i, prior_low = sl[-1]
        row = d15.iloc[sweep_i]
        sweep_high = float(row["h"]); sweep_low = float(row["l"]); sweep_close = float(row["c"])
        if sweep_low < prior_low - SWEEP_BUFFER_ATR * a15 and sweep_close > prior_low:
            internal = [(i, p) for i, p in highs if prior_low_i < i < sweep_i]
            mss = internal[-1][1] if internal else float(d15["h"].iloc[max(0, sweep_i-6):sweep_i].max())
            for mss_i in range(sweep_i + 1, len(d15)):
                if float(d15["c"].iloc[mss_i]) > mss + MSS_BUFFER_ATR * a15:
                    ids = [(i, p) for i, p in lows if sweep_i < i < mss_i]
                    idm = ids[-1][1] if ids else sweep_low
                    zone_i = sweep_i
                    for j in range(mss_i - 1, sweep_i - 1, -1):
                        if float(d15["c"].iloc[j]) <= float(d15["o"].iloc[j]): zone_i = j; break
                    z = d15.iloc[zone_i]
                    zl = min(float(z["o"]), float(z["l"])) - ZONE_PAD_ATR * a15
                    zh = max(float(z["o"]), float(z["c"])) + ZONE_PAD_ATR * a15
                    return {"side":1,"sweep":prior_low,"sweep_extreme":sweep_low,"mss":mss,"idm":idm,"zone":[zl,zh]}
        if sweep_high > prior_high + SWEEP_BUFFER_ATR * a15 and sweep_close < prior_high:
            internal = [(i, p) for i, p in lows if prior_high_i < i < sweep_i]
            mss = internal[-1][1] if internal else float(d15["l"].iloc[max(0, sweep_i-6):sweep_i].min())
            for mss_i in range(sweep_i + 1, len(d15)):
                if float(d15["c"].iloc[mss_i]) < mss - MSS_BUFFER_ATR * a15:
                    ids = [(i, p) for i, p in highs if sweep_i < i < mss_i]
                    idm = ids[-1][1] if ids else sweep_high
                    zone_i = sweep_i
                    for j in range(mss_i - 1, sweep_i - 1, -1):
                        if float(d15["c"].iloc[j]) >= float(d15["o"].iloc[j]): zone_i = j; break
                    z = d15.iloc[zone_i]
                    zl = min(float(z["o"]), float(z["c"])) - ZONE_PAD_ATR * a15
                    zh = max(float(z["o"]), float(z["h"])) + ZONE_PAD_ATR * a15
                    return {"side":-1,"sweep":prior_high,"sweep_extreme":sweep_high,"mss":mss,"idm":idm,"zone":[zl,zh]}
    return None


def _build_plan(key, setup, entry, a15):
    side = setup["side"]; zl, zh = setup["zone"]; entry = float(entry)
    if side == 1:
        stop = min(setup["sweep_extreme"], zl) - STOP_BUFFER_ATR * a15; risk = entry - stop
    else:
        stop = max(setup["sweep_extreme"], zh) + STOP_BUFFER_ATR * a15; risk = stop - entry
    if not math.isfinite(risk) or risk <= 0 or risk < 0.18 * a15:
        return None, "مسافة الوقف غير صالحة"
    plan = {"entry":entry,"stop":float(stop),"tps":[float(entry + side*risk*2.0),float(entry + side*risk*3.0),float(entry + side*risk*4.0)],"rr":MIN_RR,"arithmetic_extensions":True}
    reason = _cost_reason(key, plan)
    return (None, reason) if reason else (plan, None)


def analyze_sequence(a, st, key, fr, btc_dir):
    st.setdefault("confirmation_setups", {}).pop(key, None)
    d15=fr.get("15m"); d1=fr.get("1h"); d4=fr.get("4h")
    if d15 is None or len(d15)<30:
        price=float(d15["c"].iloc[-1]) if d15 is not None and len(d15) else 0.0
        r=_base_result(key,1,price,0.0,"SweepMSS","بيانات 15m غير كافية لاستراتيجية Sweep + MSS + IDM."); r["miss"]=["بيانات 15m غير كافية"]; return r
    price=float(d15["c"].iloc[-1])
    try: a15=float(a.atr(d15).iloc[-1])
    except Exception: a15=float("nan")
    if not math.isfinite(a15) or a15<=0:
        r=_base_result(key,1,price,0.0,"SweepMSS","تعذر حساب ATR للـ15m."); r["miss"]=["ATR غير متوفر"]; return r
    t1=_safe_trend(a,d1); t4=_safe_trend(a,d4); setup=_find_setup(d15,a15)
    if setup is None:
        side=t1 if t1 in (-1,1) else (t4 if t4 in (-1,1) else 1)
        r=_base_result(key,side,price,3.5,"SweepMSSWatch","Sweep + MSS + IDM | لا يوجد تسلسل مكتمل حالياً.\nننتظر Sweep للسيولة ثم MSS مؤكد، وبعدها الرجوع إلى Demand/Supply."); r["miss"]=["لا يوجد Sweep + MSS مكتمل"]; return r
    side=setup["side"]; zl,zh=setup["zone"]; in_zone=zl<=price<=zh
    distance=0.0 if in_zone else (zl-price if price<zl else price-zh)/a15
    near_zone=in_zone or distance<=MAX_ZONE_DISTANCE_ATR
    context=(1.0 if t1==side else (-0.4 if t1==-side else 0.0))+(0.6 if t4==side else (-0.2 if t4==-side else 0.0))
    score=max(0.0,min(10.0,8.0+context+(0.4 if in_zone else 0.0))); zone_name="Demand" if side==1 else "Supply"
    obs=("Sweep + MSS + IDM | 15m\n"+f"الاتجاه: {_direction_name(side)} | Sweep: {setup['sweep']:,.2f}\n"+f"MSS: {setup['mss']:,.2f} | IDM: {setup['idm']:,.2f}\n"+f"{zone_name}: {zl:,.2f} - {zh:,.2f}\n"+f"1H: {_trend_label(t1)} | 4H: {_trend_label(t4)} (سياق فقط)")
    r=_base_result(key,side,price,score,"SweepMSSIDM",obs); r.update(zone=[float(zl),float(zh)],zone_p=float((zl+zh)/2),a1=float(a15*4),ext=round(float(distance),2))
    if not near_zone:
        # ARMED setups need a complete limit-entry plan for scanner tracking.
        armed_entry = min(max(price, zl), zh)
        plan, reason = _build_plan(key, setup, armed_entry, a15)
        if plan is None:
            r["miss"]=[reason or "تعذر بناء خطة انتظار الريتست"]
            return r
        r.update(tier="ARMED",plan=plan,cond=f"انتظار رجوع السعر إلى {zone_name} {zl:,.2f}-{zh:,.2f}",inv=f"إبطال إذا كسر السعر Sweep extreme {setup['sweep_extreme']:,.2f} بعكس الصفقة.")
        r["miss"]=[f"السعر بعيد عن {zone_name}; ننتظر retrace"]
        return r
    plan,reason=_build_plan(key,setup,price,a15)
    if plan and reason is None:
        r.update(tier="A",plan=plan); r["obs"] += "\n✅ Sweep + MSS مكتمل وIDM محدد والسعر داخل/قريب من الزون؛ R:R الأدنى 1:2."
    else: r["miss"]=[reason or "تعذر بناء خطة الدخول"]
    if getattr(a,"SPOT_ONLY",False) and side==-1:
        r["tier"]=None; r["miss"]=["SHORT disabled by SPOT_ONLY"]
    return r
