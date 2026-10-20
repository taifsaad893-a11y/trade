"""Gold and Nasdaq: trend + support/resistance + CISD.
Ordinary volume is context only.
"""
import logging
import math
import os

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
NAMES = {1: "صاعد", -1: "هابط", 0: "عرضي"}


def utc(value):
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def ordinary_volume(history, now):
    delay = float(os.environ.get("GC_DELAY_MINUTES", "20"))
    maximum_age = float(os.environ.get("GC_MAX_AGE_MINUTES", "75"))

    if (
        not math.isfinite(delay)
        or delay < 0
        or not math.isfinite(maximum_age)
        or maximum_age <= 0
    ):
        raise ValueError("invalid GC delay/age assumptions")

    cutoff = utc(now) - pd.Timedelta(minutes=delay)
    d = history[history["end"] <= cutoff].tail(21)

    if len(d) < 21:
        raise ValueError("حجم GC يحتاج 21 شمعة متاحة")

    age = (
        utc(now) - utc(d["end"].iloc[-1])
    ).total_seconds() / 60

    if age > maximum_age:
        raise ValueError("لقطة حجم GC قديمة")

    if (
        utc(d["end"].iloc[-1]) - utc(d.index[0])
        > pd.Timedelta(hours=12)
    ):
        raise ValueError("شموع حجم GC متباعدة حول إغلاق السوق")

    v = d["v"].to_numpy(dtype=float)

    if not np.isfinite(v).all() or (v < 0).any():
        raise ValueError("حجم GC غير صالح")

    average = float(v[:-1].mean())

    if average <= 0:
        raise ValueError("متوسط حجم GC غير متوفر")

    return {
        "last": float(v[-1]),
        "average20": average,
        "ratio": float(v[-1] / average),
        "age_minutes": round(age, 1),
        "last_close": utc(d["end"].iloc[-1]).isoformat(),
        "source": "yahoo:" + history.attrs.get(
            "symbol", os.environ.get("GC_SYMBOL", "GC=F")
        ),
        "role": "context_only",
        "assumed_delay_minutes": delay,
    }


def recent_cisd(d, side):
    """A fresh close crossing the opening of a 2+ opposite candle run."""
    x = d.tail(24)
    o = x["o"].to_numpy()
    c = x["c"].to_numpy()

    for j in range(len(x) - 1, max(1, len(x) - 3), -1):
        def opposite(i):
            return c[i] < o[i] if side == 1 else c[i] > o[i]

        e = next(
            (
                i for i in range(j - 1, max(-1, j - 9), -1)
                if opposite(i)
            ),
            None,
        )

        if e is None:
            continue

        b = e
        while b > 0 and opposite(b - 1):
            b -= 1

        if e - b + 1 < 2:
            continue

        level = float(o[b])

        crossed = (
            c[j - 1] <= level < c[j]
            if side == 1
            else c[j - 1] >= level > c[j]
        )
        holds = c[-1] > level if side == 1 else c[-1] < level

        if crossed and holds:
            return {
                "level": level,
                "time": x.index[j].isoformat(),
            }

    return None


def build_plan(key, side, price, z, touch, d, a15, a1, levels):
    half = 0.25 * a1
    recent = d.iloc[max(touch - 1, 0):]

    if side == 1:
        stop = (
            min(z["p"] - half, float(recent["l"].min()))
            - 0.2 * a15
        )
    else:
        stop = (
            max(z["p"] + half, float(recent["h"].max()))
            + 0.2 * a15
        )

    risk = (price - stop) * side
    minimum_risk = {"GOLD": 0.0025, "NDX": 0.003}[key]

    if (
        not math.isfinite(risk)
        or risk <= 0
        or risk > 3 * a1
        or risk < minimum_risk * price
    ):
        return None, "مسافة الوقف خارج الحدود المعتمدة"

    ahead = sorted(
        (c["p"] - price) * side
        for c in levels
        if abs(c["p"] - z["p"]) > 1e-9
        and (c["p"] - price) * side > 0
    )

    blocked = any(
        c["w"] >= 4
        and abs(c["p"] - z["p"]) > 1e-9
        and 0.2 * risk <= (c["p"] - price) * side < 0.8 * risk
        for c in levels
    )

    if blocked:
        return None, "مستوى قوي قريب يعترض الدخول"

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
        distances.append(
            (distances[-1] if distances else 0) + risk
        )

    if distances[1] / risk < 2:
        return None, "الهدف الثاني لا يوفر 2R"

    return {
        "entry": price,
        "stop": stop,
        "tps": [price + side * x for x in distances],
        "rr": distances[1] / risk,
        "arithmetic_extensions": arithmetic,
    }, None


def analyze_zones(key, fr, strategy):
    d15 = fr["15m"]
    d1 = fr["1h"]
    d4 = fr["4h"]
    price = float(d15["c"].iloc[-1])

    res = {
        "key": key,
        "side": 1,
        "price": price,
        "score": 0.0,
        "style": "SimpleZones",
        "tf": "15m",
        "tier": None,
        "ext": 0.0,
        "obs": "",
        "miss": [],
        "strategy_version": "simple_zones_v1",
    }

    if min(len(d15), len(d1), len(d4)) < 60 or len(d15) < 120:
        res["obs"] = "تحليل المناطق: بيانات ناقصة"
        return res

    a15 = float(strategy["atr"](d15).iloc[-1])
    a1 = float(strategy["atr"](d1).iloc[-1])

    if not all(math.isfinite(x) and x > 0 for x in (a15, a1)):
        res["obs"] = "تحليل المناطق: تعذر حساب مسافة الوقف"
        return res

    t1 = strategy["trend"](d1)
    t4 = strategy["trend"](d4)
    levels = strategy["sr_multi"](fr, 0.35 * a1)
    f = strategy["fmt"]

    sup = sorted(
        [z for z in levels if z["p"] < price],
        key=lambda z: -z["p"],
    )
    resistance = sorted(
        [z for z in levels if z["p"] > price],
        key=lambda z: z["p"],
    )

    obs = [
        f"ترند 1h: {NAMES[t1]} | 4h للسياق: {NAMES[t4]}"
    ]

    if sup:
        obs.append(
            "دعم: " + " | ".join(f(z["p"]) for z in sup[:2])
        )

    if resistance:
        obs.append(
            "مقاومة: "
            + " | ".join(f(z["p"]) for z in resistance[:2])
        )

    volume = None

    try:
        if key == "GOLD":
            history = fr.get("_gc15")
            decision = utc(
                fr.get("_decision_at", pd.Timestamp.now(tz="UTC"))
            )

            if history is None:
                if fr.get("_backtest"):
                    raise ValueError(
                        "لا توجد لقطة GC تاريخية؛ "
                        "لا يُجلب حجم حالي للاختبار"
                    )

                from gold_volume import live_history
                history = live_history(decision)

            volume = ordinary_volume(history, decision)
            res["gc_volume"] = volume

            obs.append(
                f"حجم GC العادي: {volume['last']:,.0f} عقد | "
                f"متوسط 20 شمعة {volume['average20']:,.0f} | "
                f"{volume['ratio']:.2f}x"
            )
            obs.append(
                f"مصدر الحجم {volume['source']} | "
                f"آخر إغلاق GC {volume['last_close']} | "
                f"عمر {volume['age_minutes']} دقيقة"
            )

        else:
            v = d15["v"].tail(21).to_numpy(dtype=float)

            if (
                len(v) < 21
                or not np.isfinite(v).all()
                or (v < 0).any()
                or float(v[:-1].mean()) <= 0
            ):
                raise ValueError("حجم المؤشر غير متوفر")

            volume = {
                "last": float(v[-1]),
                "average20": float(v[:-1].mean()),
                "ratio": float(v[-1] / v[:-1].mean()),
                "source": fr["_source"],
                "role": "context_only",
            }
            res["source_volume"] = volume

            obs.append(
                f"حجم المؤشر حسب المصدر: {volume['last']:,.0f} | "
                f"متوسط 20 شمعة {volume['average20']:,.0f} | "
                f"{volume['ratio']:.2f}x"
            )

    except Exception as exc:
        from scanner_data import safe_error
        obs.append("الفوليوم غير متوفر: " + safe_error(exc))

    obs.append(
        "الفوليوم سياق إضافي وليس شرط دخول إجباري. "
        "بدون فوليوم بروفايل أو أوردر فلو."
    )

    if key == "GOLD":
        obs.append(
            "الحجم يخص عقود GC بتأخير المصدر، وليس سبوت الذهب."
        )

    x = d15.tail(8)
    candidates = []

    for z in levels:
        side = 1 if z["p"] <= price else -1

        if t1 and side != t1:
            continue

        bot = z["p"] - 0.25 * a1
        top = z["p"] + 0.25 * a1

        touches = np.flatnonzero(
            ((x["l"] <= top) & (x["h"] >= bot)).to_numpy()
        )

        if not len(touches):
            continue

        touch = int(touches[-1])
        near = (
            abs(price - z["p"]) <= 0.4 * a1
            or touch == len(x) - 1
        )
        trigger = recent_cisd(d15, side)
        confirmed = bool(
            t1 == side
            and trigger
            and abs(price - z["p"]) <= 1.5 * a1
        )

        if near or confirmed:
            candidates.append(
                (
                    confirmed,
                    -abs(price - z["p"]),
                    side,
                    z,
                    touch,
                    trigger,
                )
            )

    if not candidates:
        res["side"] = t1 or 1
        res["miss"] = [
            "انتظار وصول السعر لمنطقة دعم أو مقاومة معتمدة"
        ]
        res["obs"] = "\n".join(obs)
        return res

    candidates.sort(
        key=lambda candidate: (candidate[0], candidate[1]),
        reverse=True,
    )
    _, _, side, z, touch, trigger = candidates[0]

    res.update(
        side=side,
        zone_p=float(z["p"]),
        a1=a1,
        score=2.5 * sum([
            t1 == side,
            True,
            bool(trigger),
            bool(volume and volume["ratio"] >= 1.2),
        ]),
    )

    zone_name = (
        "دعم " if side == 1 else "مقاومة "
    ) + f(z["p"])

    res["zone_txt"] = zone_name
    obs.append("المنطقة المرصودة: " + zone_name)

    if trigger:
        obs.append(
            f"CISD: مؤكد عند {f(trigger['level'])} | "
            f"شمعة {trigger['time']}"
        )
    else:
        obs.append("CISD: بانتظار إغلاق يؤكد تغير التسليم")

    res["tier"] = "B"
    res["cond"] = (
        "بقاء السعر عند المنطقة مع ترند الساعة "
        "وظهور CISD على شمعة 15m مغلقة، ووقف وأهداف صالحين"
    )
    res["inv"] = (
        "إغلاق أسفل " + f(z["p"] - 0.25 * a1)
        if side == 1
        else "إغلاق أعلى " + f(z["p"] + 0.25 * a1)
    )

    if t1 != side:
        res["miss"].append("ترند الساعة غير واضح")

    if not trigger:
        res["miss"].append("انتظار CISD مؤكد")

    if abs(price - z["p"]) > 1.5 * a1:
        res["miss"].append("السعر ابتعد عن المنطقة")

    if not res["miss"]:
        plan, why = build_plan(
            key, side, price, z, touch, x, a15, a1, levels
        )

        if plan:
            cost_bps = float(os.environ.get(
                "ROUNDTRIP_COST_BPS_" + key,
                os.environ.get("ROUNDTRIP_COST_BPS", "14"),
            ))

            if not math.isfinite(cost_bps) or cost_bps < 0:
                raise ValueError("invalid cost assumption")

            cost = (
                price * cost_bps
                / 10000
                / abs(price - plan["stop"])
            )

            if cost > 0.35:
                res["miss"].append(
                    f"الكلفة المفترضة {cost:.2f}R تتجاوز 0.35R"
                )
            else:
                res.update(tier="A", plan=plan)

            obs.append(
                "الأهداف من مستويات السعر، وتُستكمل "
                "بامتدادات حسابية عند الحاجة. كل هدف ثلث الحجم."
            )
        else:
            res["miss"].append(why)

    if res["miss"]:
        obs.append("ناقص: " + " | ".join(res["miss"]))

    obs.append(
        "السكور يصف توافق العناصر فقط، "
        "وليس احتمال نجاح مقاساً."
    )

    res["obs"] = "\n".join(obs)

    LOG.info(
        "SIMPLE_ZONES asset=%s tier=%s side=%s missing=%s",
        key, res["tier"], side, res["miss"],
    )

    return res


def analyze_gold(fr, strategy):
    return analyze_zones("GOLD", fr, strategy)