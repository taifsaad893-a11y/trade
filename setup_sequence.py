import math
import os
import pandas as pd

from simple_zones import analyze_zones, build_plan, utc


def confirmation_level(d, side):
    x = d.tail(24)
    o = x["o"].to_numpy()
    c = x["c"].to_numpy()

    def opposite(i):
        return c[i] < o[i] if side == 1 else c[i] > o[i]

    for end in range(len(x) - 1, max(-1, len(x) - 9), -1):
        if not opposite(end):
            continue

        start = end
        while start > 0 and opposite(start - 1):
            start -= 1

        if end - start + 1 >= 2:
            return float(o[start])

    return None


def terminal(st, key, p, price, reason):
    st["confirmation_setups"].pop(key, None)

    return {
        "key": key,
        "side": p["side"],
        "price": price,
        "score": 0.0,
        "style": "SimpleZones",
        "tf": "15m",
        "tier": None,
        "ext": 0.0,
        "obs": reason,
        "miss": [reason],
        "strategy_version": "sequence_v1",
        "_sequence_notice": (
            f"⚠️ {key} | الإعداد {p['id']}\n{reason}"
        ),
    }


def waiting(p, price):
    side = p["side"]
    direction = "فوق" if side == 1 else "تحت"
    invalid_direction = "تحت" if side == 1 else "فوق"

    return {
        "key": p["key"],
        "side": side,
        "price": price,
        "score": p["score"],
        "style": "SimpleZones",
        "tf": "15m",
        "tier": "B",
        "ext": 0.0,
        "zone_p": p["zone"]["p"],
        "a1": p["a1"],
        "miss": [],
        "strategy_version": "sequence_v1",
        "cond": (
            f"إغلاق شمعة 15m جديدة {direction} "
            f"{p['trigger']:,.2f}؛ لا يشترط ريتست بعدها. "
            "عند التأكيد: ترند الساعة موافق، وقف ضمن الحدود، "
            "الهدف الثاني 2R على الأقل، والكلفة حتى 0.35R."
        ),
        "inv": (
            f"إغلاق {invalid_direction} {p['invalid']:,.2f}"
            " أو انتهاء ساعتين من إعلان الإعداد"
        ),
        "obs": (
            p["obs"]
            + f"\nالإعداد المحفوظ: {p['id']}"
            + "\nالمنطقة ومستوى التأكيد ثابتان لهذا الإعداد."
            + "\nالفوليوم للسياق فقط."
        ),
    }


def breakout_plan(key, side, price, level, d, a15, a1, levels):
    minimum = {
        "BTC": 0.004,
        "ETH": 0.004,
        "SOL": 0.005,
        "GOLD": 0.0025,
        "NDX": 0.003,
    }

    recent = d.tail(2)
    half = 0.25 * a1

    if side == 1:
        stop = min(
            level - half, float(recent["l"].min())
        ) - 0.2 * a15
    else:
        stop = max(
            level + half, float(recent["h"].max())
        ) + 0.2 * a15

    risk = (price - stop) * side

    if (
        not math.isfinite(risk)
        or risk <= 0
        or risk > 3 * a1
        or risk < minimum[key] * price
    ):
        return None, "مسافة الوقف خارج الحدود المعتمدة"

    blocked = any(
        z["w"] >= 4
        and abs(z["p"] - level) > 1e-9
        and 0.2 * risk <= (z["p"] - price) * side < 0.8 * risk
        for z in levels
    )

    if blocked:
        return None, "مستوى قوي قريب يعترض الدخول"

    ahead = sorted(
        (z["p"] - price) * side
        for z in levels
        if abs(z["p"] - level) > 1e-9
        and (z["p"] - price) * side > 0
    )

    distances = []

    for distance in ahead:
        if (
            0.8 * risk <= distance <= 4 * risk
            and (
                not distances
                or distance - distances[-1] >= 0.5 * risk
            )
        ):
            distances.append(distance)

        if len(distances) == 3:
            break

    arithmetic = len(distances) < 3

    while len(distances) < 3:
        previous = distances[-1] if distances else 0
        distances.append(previous + risk)

    if distances[1] / risk < 2:
        return None, "الهدف الثاني لا يوفر 2R"

    return {
        "entry": price,
        "stop": stop,
        "tps": [price + side * v for v in distances],
        "rr": distances[1] / risk,
        "arithmetic_extensions": arithmetic,
    }, None


def cost_reason(key, plan):
    bps = float(os.environ.get(
        "ROUNDTRIP_COST_BPS_" + key,
        os.environ.get("ROUNDTRIP_COST_BPS", "14"),
    ))

    if not math.isfinite(bps) or bps < 0:
        raise ValueError("invalid cost assumption")

    risk = abs(plan["entry"] - plan["stop"])
    cost = plan["entry"] * bps / 10000 / risk

    if cost > 0.35:
        return f"الكلفة {cost:.2f}R تتجاوز 0.35R"

    return None


def breakout(a, key, fr):
    d = fr["15m"]

    if len(d) < 120 or len(fr["1h"]) < 60:
        return None

    price = float(d["c"].iloc[-1])
    previous = float(d["c"].iloc[-2])

    # نستبعد شمعة الإشارة من نطاق الكسر.
    high = float(d["h"].iloc[-21:-1].max())
    low = float(d["l"].iloc[-21:-1].min())

    a15 = float(a.atr(d).iloc[-1])
    a1 = float(a.atr(fr["1h"]).iloc[-1])

    if not all(math.isfinite(v) and v > 0 for v in (a15, a1)):
        return None

    if previous <= high and price > high + 0.1 * a15:
        side, level = 1, high
    elif previous >= low and price < low - 0.1 * a15:
        side, level = -1, low
    else:
        return None

    res = {
        "key": key,
        "side": side,
        "price": price,
        "score": 5.0,
        "style": "Breakout",
        "tf": "15m",
        "tier": "B",
        "ext": abs(price - level) / a15,
        "zone_p": level,
        "a1": a1,
        "miss": [],
        "strategy_version": "sequence_v1",
        "inv": (
            ("إغلاق تحت " if side == 1 else "إغلاق فوق ")
            + f"{level:,.2f}"
        ),
        "obs": (
            "بريك أوت بإغلاق 15m خارج نطاق 20 شمعة سابقة."
            "\nلا يشترط CISD أو ريتست أو فوليوم."
            "\nالسكور تصنيف للقواعد وليس احتمال نجاح."
            "\nالأهداف من مستويات السعر، وتُستكمل "
            "بامتدادات حسابية عند الحاجة."
        ),
    }

    reason = None
    plan = None

    if a.trend(fr["1h"]) != side:
        reason = "ترند الساعة غير موافق لاتجاه البريك أوت"
    elif abs(price - level) > a15:
        reason = "شمعة الكسر ابتعدت أكثر من ATR 15m؛ لا مطاردة"

    if reason is None:
        levels = a.sr_multi(fr, 0.35 * a1)
        plan, reason = breakout_plan(
            key, side, price, level, d, a15, a1, levels
        )

    if plan and reason is None:
        reason = cost_reason(key, plan)

    if reason:
        res["cond"] = "لا دخول بهذا الكسر: " + reason
        res["miss"] = [reason]
        res["obs"] += "\nسبب عدم الدخول: " + reason
    else:
        res.update(tier="A", plan=plan, score=7.5)

    return res


def analyze_sequence(a, st, key, fr, btc_dir):
    br = breakout(a, key, fr)

    if getattr(a, "SPOT_ONLY", False) and br:
        if br["side"] == -1:
            br = None

    # البريك أوت مسار إضافي للكريبتو.
    if key not in ("GOLD", "NDX"):
        return br if br else a.analyze(key, fr, btc_dir)

    pending = st.setdefault("confirmation_setups", {})
    d = fr["15m"]

    close = utc(d["end"].iloc[-1])
    price = float(d["c"].iloc[-1])
    now = utc(fr.get(
        "_decision_at", pd.Timestamp.now(tz="UTC")
    ))
    p = pending.get(key)

    # البريك أوت الصالح لا ينتظر إعداد المنطقة.
    if br and (br["tier"] == "A" or p is None):
        if br["tier"] == "A":
            pending.pop(key, None)
        return br

    if p:
        if now > utc(p["expires"]):
            return terminal(
                st, key, p, price,
                "انتهى الإعداد بعد ساعتين بدون دخول",
            )

        fresh = d[d["end"].map(utc) > utc(p["created"])]

        if len(fresh):
            broken = (
                fresh["c"] <= p["invalid"]
                if p["side"] == 1
                else fresh["c"] >= p["invalid"]
            )

            if bool(broken.any()):
                return terminal(
                    st, key, p, price,
                    "إغلاق شمعة تجاوز حد إلغاء المنطقة",
                )

        if close <= utc(p["created"]):
            return waiting(p, price)

        confirmed = (
            price > p["trigger"]
            if p["side"] == 1
            else price < p["trigger"]
        )

        if not confirmed:
            return waiting(p, price)

        reason = None
        plan = None

        if a.trend(fr["1h"]) != p["side"]:
            reason = (
                "تحقق الإغلاق، لكن ترند الساعة "
                "لم يعد موافقاً للاتجاه"
            )
        elif abs(price - p["zone"]["p"]) > 1.5 * p["a1"]:
            reason = (
                "تحقق الإغلاق، لكن السعر ابتعد "
                "أكثر من 1.5 ATR ساعة عن المنطقة"
            )

        if reason is None:
            initial = pd.DataFrame(p["bars"])
            plan, reason = build_plan(
                key, p["side"], price, p["zone"],
                p["touch"], initial,
                p["a15"], p["a1"], p["levels"],
            )

        if plan and reason is None:
            reason = cost_reason(key, plan)

        if reason:
            return terminal(
                st, key, p, price,
                "لم يُفتح دخول ورقي: " + reason,
            )

        res = waiting(p, price)
        res.update(
            tier="A",
            plan=plan,
            score=min(10.0, p["score"] + 2.5),
        )
        res["obs"] += (
            "\n✅ تحقق شرط الإعداد المحفوظ؛ "
            "لا ريتست إضافي."
        )
        pending.pop(key, None)
        return res

    res = analyze_zones(key, fr, vars(a))

    if getattr(a, "SPOT_ONLY", False) and res["side"] == -1:
        res["tier"] = None
        return res

    if res["tier"] != "B":
        return res

    side = res["side"]
    trigger = confirmation_level(d, side)

    if trigger is None:
        res["cond"] = (
            "تكوّن سلسلة من شمعتين معاكستين "
            "لتحديد مستوى CISD؛ "
            "لا يوجد مستوى تأكيد محفوظ بعد"
        )
        return res

    fulfilled = (
        price > trigger if side == 1 else price < trigger
    )

    if fulfilled:
        res["cond"] = (
            "التأكيد السعري تحقق بالفعل؛ "
            + " | ".join(
                res.get("miss")
                or ["الخطة لم تستوفِ شروط الدخول"]
            )
        )
        return res

    a1 = float(res["a1"])
    a15 = float(a.atr(d).iloc[-1])
    z = {"p": float(res["zone_p"]), "w": 3}
    x = d.tail(8)

    touches = (
        (x["l"] <= z["p"] + 0.25 * a1)
        & (x["h"] >= z["p"] - 0.25 * a1)
    )
    positions = [
        i for i, touched in enumerate(touches) if touched
    ]

    if not positions:
        return res

    levels = [
        {"p": float(v["p"]), "w": int(v["w"])}
        for v in a.sr_multi(fr, 0.35 * a1)
    ]

    p = {
        "id": key + "-" + close.strftime("%Y%m%d%H%M"),
        "key": key,
        "side": side,
        "created": close.isoformat(),
        "expires": (
            now + pd.Timedelta(hours=2)
        ).isoformat(),
        "trigger": trigger,
        "invalid": z["p"] - side * 0.25 * a1,
        "zone": z,
        "a1": a1,
        "a15": a15,
        "touch": positions[-1],
        "levels": levels,
        "bars": (
            x[["o", "h", "l", "c"]]
            .astype(float)
            .to_dict("records")
        ),
        "obs": res["obs"],
        "score": float(res["score"]),
    }

    pending[key] = p
    return waiting(p, price)