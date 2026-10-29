import math
import os

import pandas as pd


STRATEGY_VERSION = "taifmarket_structure_v2"

BREAK_BUFFER_ATR = 0.05
BREAK_BODY_ATR = 0.35
BREAK_CLOSE_EDGE = 0.35

RETEST_TOUCH_ATR = 0.20
RETEST_CLOSE_ATR = 0.03
RETEST_BODY_ATR = 0.10

BOUNCE_TOUCH_ATR = 0.25
BOUNCE_MAX_PENETRATION_ATR = 0.35
BOUNCE_CLOSE_ATR = 0.08
BOUNCE_BODY_ATR = 0.15

WATCH_DISTANCE_ATR = 0.35
MAX_DIRECT_EXTENSION_ATR = 0.80
STOP_BUFFER_ATR = 0.20


def _utc(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


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


def _volume_ratio(d15):
    if len(d15) < 22 or "v" not in d15:
        return None

    v = pd.to_numeric(d15["v"], errors="coerce")
    previous = float(v.iloc[-21:-1].mean())
    current = float(v.iloc[-1])

    if (
        not math.isfinite(previous)
        or previous <= 0
        or not math.isfinite(current)
        or current < 0
    ):
        return None

    return current / previous


def _base_result(key, side, price, score, style, obs):
    return {
        "key": key,
        "side": int(side),
        "price": float(price),
        "score": round(float(score), 1),
        "style": style,
        "tf": "15m",
        "tier": None,
        "ext": 0.0,
        "miss": [],
        "obs": obs,
        "strategy_version": STRATEGY_VERSION,
    }


def _cost_reason(key, plan):
    try:
        bps = float(
            os.environ.get(
                "ROUNDTRIP_COST_BPS_" + key,
                os.environ.get("ROUNDTRIP_COST_BPS", "14"),
            )
        )
    except (TypeError, ValueError):
        return "افتراض الكلفة غير صالح"

    risk = abs(plan["entry"] - plan["stop"])
    if not math.isfinite(bps) or bps < 0 or risk <= 0:
        return "خطة المخاطرة غير صالحة"

    cost_r = plan["entry"] * bps / 10000 / risk
    if cost_r > 0.35:
        return f"الكلفة المفترضة {cost_r:.2f}R تتجاوز 0.35R"

    return None


def _build_plan(key, side, entry, level, last, a15, a1, mode):
    entry = float(entry)
    level = float(level)
    a1 = float(a1) if math.isfinite(float(a1)) and float(a1) > 0 else 4.0 * a15

    if side == 1:
        if mode in ("RETEST", "BOUNCE"):
            stop = min(
                float(last["l"]) - 0.05 * a15,
                level - STOP_BUFFER_ATR * a15,
            )
        else:
            stop = min(
                float(last["l"]) - 0.10 * a15,
                level - STOP_BUFFER_ATR * a15,
            )
        risk = entry - stop
    else:
        if mode in ("RETEST", "BOUNCE"):
            stop = max(
                float(last["h"]) + 0.05 * a15,
                level + STOP_BUFFER_ATR * a15,
            )
        else:
            stop = max(
                float(last["h"]) + 0.10 * a15,
                level + STOP_BUFFER_ATR * a15,
            )
        risk = stop - entry

    if not math.isfinite(risk) or risk <= 0:
        return None, "مسافة الوقف غير صالحة"
    if risk < 0.20 * a15:
        return None, "الوقف قريب جداً"
    if risk > 2.5 * a1:
        return None, "الوقف واسع جداً"

    plan = {
        "entry": entry,
        "stop": float(stop),
        "tps": [
            float(entry + side * risk),
            float(entry + side * risk * 2.0),
            float(entry + side * risk * 3.0),
        ],
        "rr": 2.0,
        "arithmetic_extensions": True,
    }

    reason = _cost_reason(key, plan)
    if reason:
        return None, reason

    return plan, None


def _ema_context(a, d15, side, price):
    try:
        if len(d15) < 50:
            return None
        ema20 = float(a.ema(d15["c"], 20).iloc[-1])
        ema50 = float(a.ema(d15["c"], 50).iloc[-1])
    except Exception:
        return None

    if not all(math.isfinite(x) for x in (ema20, ema50)):
        return None

    if side == 1:
        return price > ema20 > ema50
    return price < ema20 < ema50


def _score_quality(structure_score, side, trend_1h, trend_4h, ema_ok, vr):
    score = float(structure_score)

    if trend_1h == side:
        score += 0.8
    elif trend_1h == -side:
        score -= 0.5

    if trend_4h == side:
        score += 0.6
    elif trend_4h == -side:
        score -= 0.3

    if ema_ok is True:
        score += 0.5
    elif ema_ok is False:
        score -= 0.2

    if vr is not None:
        if vr >= 1.50:
            score += 0.8
        elif vr >= 1.00:
            score += 0.4
        elif vr < 0.65:
            score -= 0.3

    return round(max(0.0, min(10.0, score)), 1)


def _sr_levels(a, fr, a15, a1):
    levels = []

    try:
        for zone in a.sr_multi(fr, 0.35 * a1):
            p = float(zone["p"])
            if math.isfinite(p):
                levels.append((p, int(zone.get("w", 1)), "multi-TF"))
    except Exception:
        pass

    d15 = fr["15m"]
    d1 = fr.get("1h")

    if len(d15) >= 22:
        base = d15.iloc[-22:-2]
        levels.extend(
            [
                (float(base["l"].min()), 3, "15m support"),
                (float(base["h"].max()), 3, "15m resistance"),
            ]
        )

    if d1 is not None and len(d1) >= 12:
        base1 = d1.iloc[-12:-1]
        levels.extend(
            [
                (float(base1["l"].min()), 4, "1h support"),
                (float(base1["h"].max()), 4, "1h resistance"),
            ]
        )

    deduped = []
    for p, weight, source in sorted(levels, key=lambda item: (-item[1], item[0])):
        if any(abs(p - existing[0]) <= 0.08 * a15 for existing in deduped):
            continue
        deduped.append((p, weight, source))

    return deduped


def _find_bounce(d15, levels, a15):
    last = d15.iloc[-1]
    price = float(last["c"])
    opening = float(last["o"])
    high = float(last["h"])
    low = float(last["l"])
    span = max(high - low, 1e-12)
    body = abs(price - opening)

    long_candidates = []
    short_candidates = []

    for level, weight, source in levels:
        # Support bounce: level is at/below current close, candle probes it,
        # then closes clearly back above it with a bullish 15m confirmation.
        if level <= price + BOUNCE_TOUCH_ATR * a15:
            touched = low <= level + BOUNCE_TOUCH_ATR * a15
            held = low >= level - BOUNCE_MAX_PENETRATION_ATR * a15
            confirmed = (
                price >= level + BOUNCE_CLOSE_ATR * a15
                and price > opening
                and body >= BOUNCE_BODY_ATR * a15
                and (price - low) / span >= 0.60
            )
            if touched and held and confirmed:
                distance = abs(price - level) / a15
                long_candidates.append((weight - distance, level, source))

        # Resistance bounce: mirror image of support bounce.
        if level >= price - BOUNCE_TOUCH_ATR * a15:
            touched = high >= level - BOUNCE_TOUCH_ATR * a15
            held = high <= level + BOUNCE_MAX_PENETRATION_ATR * a15
            confirmed = (
                price <= level - BOUNCE_CLOSE_ATR * a15
                and price < opening
                and body >= BOUNCE_BODY_ATR * a15
                and (high - price) / span >= 0.60
            )
            if touched and held and confirmed:
                distance = abs(price - level) / a15
                short_candidates.append((weight - distance, level, source))

    best_long = max(long_candidates, default=None)
    best_short = max(short_candidates, default=None)

    if best_long and best_short:
        return (1, best_long[1], best_long[2]) if best_long[0] >= best_short[0] else (-1, best_short[1], best_short[2])
    if best_long:
        return 1, best_long[1], best_long[2]
    if best_short:
        return -1, best_short[1], best_short[2]
    return None


def analyze_sequence(a, st, key, fr, btc_dir):
    """Runtime strategy used by scanner.py.

    Entry is structure-first:
      * clear 15m breakout,
      * breakout + retest,
      * clear S/R bounce with closed-15m confirmation.

    1H, 4H, EMA and volume are quality/score context only.  They never veto
    an otherwise valid structural entry, and 1H RANGE is never a no-trade gate.
    """

    st.setdefault("confirmation_setups", {}).pop(key, None)

    d15 = fr["15m"]
    d1 = fr.get("1h")
    d4 = fr.get("4h")

    if d15 is None or len(d15) < 25:
        price = float(d15["c"].iloc[-1]) if d15 is not None and len(d15) else 0.0
        return _base_result(
            key,
            1,
            price,
            0.0,
            "Structure15m",
            "بيانات 15m غير كافية للتحليل.",
        )

    last = d15.iloc[-1]
    prev = d15.iloc[-2]
    price = float(last["c"])

    try:
        a15 = float(a.atr(d15).iloc[-1])
    except Exception:
        a15 = float("nan")

    if not math.isfinite(a15) or a15 <= 0:
        return _base_result(
            key,
            1,
            price,
            0.0,
            "Structure15m",
            "تعذر حساب ATR للـ15m.",
        )

    try:
        a1 = float(a.atr(d1).iloc[-1]) if d1 is not None and len(d1) >= 15 else 4.0 * a15
    except Exception:
        a1 = 4.0 * a15

    trend_1h = _safe_trend(a, d1)
    trend_4h = _safe_trend(a, d4)
    vr = _volume_ratio(d15)

    # Breakout levels are derived entirely from closed 15m structure.
    base = d15.iloc[-22:-2]
    high = float(base["h"].max())
    low = float(base["l"].min())

    break_buffer = BREAK_BUFFER_ATR * a15
    touch_buffer = RETEST_TOUCH_ATR * a15
    close_buffer = RETEST_CLOSE_ATR * a15

    previous_close = float(prev["c"])
    opening = float(last["o"])
    high_now = float(last["h"])
    low_now = float(last["l"])
    candle_span = max(high_now - low_now, 1e-12)
    body = abs(price - opening)
    body_atr = body / a15

    prev_long_break = previous_close > high + break_buffer
    prev_short_break = previous_close < low - break_buffer

    fresh_long = price > high + break_buffer and previous_close <= high + break_buffer
    fresh_short = price < low - break_buffer and previous_close >= low - break_buffer

    clear_long_breakout = (
        fresh_long
        and price > opening
        and body_atr >= BREAK_BODY_ATR
        and (high_now - price) / candle_span <= BREAK_CLOSE_EDGE
    )
    clear_short_breakout = (
        fresh_short
        and price < opening
        and body_atr >= BREAK_BODY_ATR
        and (price - low_now) / candle_span <= BREAK_CLOSE_EDGE
    )

    clean_long_retest = (
        prev_long_break
        and low_now <= high + touch_buffer
        and price >= high + close_buffer
        and price > opening
        and body_atr >= RETEST_BODY_ATR
    )
    clean_short_retest = (
        prev_short_break
        and high_now >= low - touch_buffer
        and price <= low - close_buffer
        and price < opening
        and body_atr >= RETEST_BODY_ATR
    )

    levels = _sr_levels(a, fr, a15, a1)
    bounce = _find_bounce(d15, levels, a15)

    candidates = []

    if clean_long_retest:
        candidates.append((3, 1, high, "BreakoutRetest", "RETEST", 7.4, "Breakout + Retest"))
    if clean_short_retest:
        candidates.append((3, -1, low, "BreakoutRetest", "RETEST", 7.4, "Breakout + Retest"))
    if clear_long_breakout:
        candidates.append((2, 1, high, "ClearBreakout", "BREAKOUT", 7.2, "Clear Breakout"))
    if clear_short_breakout:
        candidates.append((2, -1, low, "ClearBreakout", "BREAKOUT", 7.2, "Clear Breakout"))
    if bounce is not None:
        side, level, source = bounce
        candidates.append((1, side, level, "SupportResistanceBounce", "BOUNCE", 7.0, f"S/R Bounce ({source})"))

    if candidates:
        _, side, level, style, mode, structure_score, setup_name = max(candidates, key=lambda item: item[0])

        extension = max(0.0, (price - level) * side / a15)
        ema_ok = _ema_context(a, d15, side, price)
        score = _score_quality(
            structure_score,
            side,
            trend_1h,
            trend_4h,
            ema_ok,
            vr,
        )

        volume_text = "غير متوفر" if vr is None else f"{vr:.2f}x"
        ema_text = "غير متوفر" if ema_ok is None else ("متوافق" if ema_ok else "غير متوافق")
        obs = (
            "Taifmarket | Structure-first 15m\n"
            f"الإشارة: {setup_name} | الاتجاه: {_direction_name(side)}\n"
            f"1H: {_trend_label(trend_1h)} | 4H: {_trend_label(trend_4h)}\n"
            f"EMA: {ema_text} | Volume: {volume_text}\n"
            f"المستوى: {level:,.2f} | جسم 15m: {body_atr:.2f} ATR\n"
            "1H/4H/EMA/Volume عوامل جودة وScore فقط؛ لا تمنع الدخول."
        )

        res = _base_result(key, side, price, score, style, obs)
        res.update(
            ext=round(extension, 2),
            zone_p=float(level),
            a1=float(a1),
        )

        if mode == "BREAKOUT" and extension > MAX_DIRECT_EXTENSION_ATR:
            # The breakout is structurally valid but too extended for a direct
            # fill. Arm the broken level for a retest instead of chasing.
            plan, reason = _build_plan(
                key,
                side,
                level,
                level,
                last,
                a15,
                a1,
                "RETEST",
            )
            if plan and reason is None:
                res.update(
                    tier="ARMED",
                    style="BreakoutRetest",
                    plan=plan,
                    zone=[
                        float(level - RETEST_TOUCH_ATR * a15),
                        float(level + RETEST_TOUCH_ATR * a15),
                    ],
                )
                res["obs"] += "\n🟠 الكسر واضح لكنه ممتد؛ تم تسليح Retest بدل مطاردة السعر."
                return res
            res["miss"] = [reason or "تعذر بناء خطة Retest"]
            return res

        plan, reason = _build_plan(
            key,
            side,
            price,
            level,
            last,
            a15,
            a1,
            mode,
        )
        if plan and reason is None:
            res.update(tier="A", plan=plan)
            res["obs"] += "\n✅ تأكيد 15m مكتمل؛ الإشارة صالحة للدخول."
        else:
            res["miss"] = [reason or "تعذر بناء خطة الدخول"]
            res["obs"] += "\nسبب عدم الدخول: " + res["miss"][0]

        if getattr(a, "SPOT_ONLY", False) and res["side"] == -1:
            res["tier"] = None
            res["miss"] = ["SHORT disabled by SPOT_ONLY"]
            res["obs"] += "\nSHORT ملغي لأن SPOT_ONLY مفعّل."

        return res

    # Breakout exists but the 15m confirmation is not strong enough for a
    # direct entry: arm a retest.  Higher-timeframe trend still does not gate it.
    arm_options = []
    if fresh_long or prev_long_break:
        arm_options.append((1, high))
    if fresh_short or prev_short_break:
        arm_options.append((-1, low))

    if arm_options:
        side, level = min(
            arm_options,
            key=lambda item: abs(price - item[1]),
        )
        ema_ok = _ema_context(a, d15, side, price)
        score = _score_quality(
            5.8,
            side,
            trend_1h,
            trend_4h,
            ema_ok,
            vr,
        )
        plan, reason = _build_plan(
            key,
            side,
            level,
            level,
            last,
            a15,
            a1,
            "RETEST",
        )

        obs = (
            "Taifmarket | Breakout detected, waiting 15m retest confirmation\n"
            f"الاتجاه: {_direction_name(side)} | المستوى: {level:,.2f}\n"
            f"1H: {_trend_label(trend_1h)} | 4H: {_trend_label(trend_4h)}\n"
            "1H/4H/EMA/Volume جودة فقط وليست شروط دخول."
        )
        res = _base_result(key, side, price, score, "BreakoutRetest", obs)

        if plan and reason is None:
            res.update(
                tier="ARMED",
                plan=plan,
                zone=[
                    float(level - RETEST_TOUCH_ATR * a15),
                    float(level + RETEST_TOUCH_ATR * a15),
                ],
            )
            return res

        res["miss"] = [reason or "تعذر تسليح Retest"]
        return res

    # Early watch is informational only; it does not prevent future entries.
    long_distance = high - price
    short_distance = price - low
    approaching_long = 0 <= long_distance <= WATCH_DISTANCE_ATR * a15
    approaching_short = 0 <= short_distance <= WATCH_DISTANCE_ATR * a15

    if approaching_long or approaching_short:
        if approaching_long and approaching_short:
            side = 1 if long_distance <= short_distance else -1
        else:
            side = 1 if approaching_long else -1

        level = high if side == 1 else low
        trigger = level + side * break_buffer
        ema_ok = _ema_context(a, d15, side, price)
        score = _score_quality(
            4.8,
            side,
            trend_1h,
            trend_4h,
            ema_ok,
            vr,
        )

        res = _base_result(
            key,
            side,
            price,
            score,
            "StructureWatch",
            (
                "Taifmarket | السعر قريب من كسر 15m\n"
                f"الاتجاه المحتمل: {_direction_name(side)} | المستوى: {level:,.2f}\n"
                f"1H: {_trend_label(trend_1h)} | 4H: {_trend_label(trend_4h)}\n"
                "الساعة لا تمنع الإشارة؛ ننتظر تأكيد 15m فقط."
            ),
        )
        res.update(
            tier="B",
            cond=f"إغلاق 15m {'فوق' if side == 1 else 'تحت'} {trigger:,.2f} أو Retest/Bounce مؤكد.",
            inv="فشل تأكيد 15m أو كسر واضح بالاتجاه المعاكس.",
        )
        return res

    # No structural 15m setup right now.  RANGE is reported as context only.
    side = 1
    if trend_1h in (-1, 1):
        side = trend_1h
    elif trend_4h in (-1, 1):
        side = trend_4h

    ema_ok = _ema_context(a, d15, side, price)
    score = _score_quality(
        3.5,
        side,
        trend_1h,
        trend_4h,
        ema_ok,
        vr,
    )

    res = _base_result(
        key,
        side,
        price,
        score,
        "Structure15m",
        (
            "Taifmarket | لا توجد إشارة دخول 15m مكتملة حالياً.\n"
            f"1H: {_trend_label(trend_1h)} | 4H: {_trend_label(trend_4h)}\n"
            "المطلوب للدخول: Breakout واضح، أو Breakout+Retest، "
            "أو Bounce واضح من Support/Resistance مع تأكيد 15m.\n"
            "1H/4H/EMA/Volume عوامل جودة وScore فقط."
        ),
    )
    res["miss"] = ["لا يوجد تأكيد هيكلي 15m مؤهل حالياً"]

    return res
