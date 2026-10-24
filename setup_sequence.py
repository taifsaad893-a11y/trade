import math
import os
import pandas as pd


# =========================================================
# STRATEGY CONFIG
# Trend Continuation + Breakout / Retest
# =========================================================

STRATEGY_VERSION = "trend_breakout_retest_v1"

# Breakout must close beyond the level by this ATR amount.
BREAK_BUFFER_ATR = 0.05

# Strong breakout requirements for direct A setup.
STRONG_VOLUME_RATIO = 1.25
STRONG_BODY_ATR = 0.55

# Don't chase an already extended breakout.
MAX_DIRECT_EXTENSION_ATR = 0.70

# Retest zone around the broken level.
RETEST_TOUCH_ATR = 0.20

# Retest candle must close back in trend direction.
RETEST_CLOSE_ATR = 0.02

# How close price can be before breakout to send a B watch.
WATCH_DISTANCE_ATR = 0.35

# Stop buffer behind structure.
STOP_BUFFER_ATR = 0.20

# Minimum quality scores.
A_MIN_SCORE = 7.0
B_MIN_SCORE = 5.5

# Minimum R:R to TP2.
MIN_TP2_RR = 2.0


# =========================================================
# HELPERS
# =========================================================

def utc(value):
    t = pd.Timestamp(value)

    if t.tzinfo is None:
        return t.tz_localize("UTC")

    return t.tz_convert("UTC")


def volume_ratio(d):
    """
    Current closed 15m candle volume vs previous 20 candles.

    Volume is a quality bonus, not a mandatory condition
    except for direct strong-breakout entry.
    """

    if len(d) < 22:
        return None

    v = d["v"].astype(float)

    previous = float(
        v.iloc[-21:-1].mean()
    )

    if (
        not math.isfinite(previous)
        or previous <= 0
    ):
        return None

    current = float(v.iloc[-1])

    return current / previous


def direction_name(side):
    return "LONG" if side == 1 else "SHORT"


# =========================================================
# COST CHECK
# =========================================================

def cost_reason(key, plan):
    bps = float(
        os.environ.get(
            "ROUNDTRIP_COST_BPS_" + key,
            os.environ.get(
                "ROUNDTRIP_COST_BPS",
                "14",
            ),
        )
    )

    if (
        not math.isfinite(bps)
        or bps < 0
    ):
        raise ValueError(
            "invalid cost assumption"
        )

    risk = abs(
        plan["entry"]
        - plan["stop"]
    )

    if risk <= 0:
        return "مسافة الوقف غير صالحة"

    cost = (
        plan["entry"]
        * bps
        / 10000
        / risk
    )

    if cost > 0.35:
        return (
            f"الكلفة {cost:.2f}R "
            "تتجاوز 0.35R"
        )

    return None


# =========================================================
# RESULT TEMPLATE
# =========================================================

def base_result(
    key,
    side,
    price,
    score,
    style,
    obs,
):
    return {
        "key": key,
        "side": side,
        "price": float(price),
        "score": round(
            float(score),
            1,
        ),
        "style": style,
        "tf": "15m",
        "tier": None,
        "ext": 0.0,
        "miss": [],
        "obs": obs,
        "strategy_version": (
            STRATEGY_VERSION
        ),
    }


# =========================================================
# TARGET / STOP PLAN
# =========================================================

def build_trade_plan(
    a,
    key,
    fr,
    side,
    entry,
    level,
    a15,
    a1,
    mode,
):
    """
    Build simple structure + ATR plan.

    TP1 = 1R
    TP2 = 2R
    TP3 = 3R

    We also check whether a strong observed 1H level
    blocks the path before TP2.
    """

    d = fr["15m"]

    last = d.iloc[-1]

    entry = float(entry)
    level = float(level)

    # =====================================================
    # STOP
    # =====================================================

    if side == 1:

        if mode == "RETEST":

            stop = min(
                float(last["l"])
                - 0.05 * a15,

                level
                - STOP_BUFFER_ATR
                * a15,
            )

        else:

            stop = min(
                level
                - STOP_BUFFER_ATR
                * a15,

                float(last["l"])
                - 0.10
                * a15,
            )

        risk = entry - stop

    else:

        if mode == "RETEST":

            stop = max(
                float(last["h"])
                + 0.05 * a15,

                level
                + STOP_BUFFER_ATR
                * a15,
            )

        else:

            stop = max(
                level
                + STOP_BUFFER_ATR
                * a15,

                float(last["h"])
                + 0.10
                * a15,
            )

        risk = stop - entry

    # =====================================================
    # VALIDATE RISK
    # =====================================================

    if (
        not math.isfinite(risk)
        or risk <= 0
    ):
        return (
            None,
            "مسافة الوقف غير صالحة",
        )

    # Avoid unrealistically tiny stop.
    if risk < 0.20 * a15:
        return (
            None,
            "الوقف قريب جداً",
        )

    # Avoid excessively wide stop.
    if risk > 2.5 * a1:
        return (
            None,
            "الوقف واسع جداً",
        )

    # =====================================================
    # TARGETS
    # =====================================================

    tp1 = (
        entry
        + side
        * risk
    )

    tp2 = (
        entry
        + side
        * risk
        * 2.0
    )

    tp3 = (
        entry
        + side
        * risk
        * 3.0
    )

    # =====================================================
    # CHECK STRONG LEVEL BEFORE TP2
    # =====================================================

    try:

        levels = a.sr_multi(
            fr,
            0.35 * a1,
        )

    except Exception:
        levels = []

    blocked = False

    for zone in levels:

        try:
            p = float(
                zone["p"]
            )

            w = int(
                zone.get(
                    "w",
                    0,
                )
            )

        except Exception:
            continue

        # Ignore the breakout level itself.
        if abs(
            p - level
        ) <= 0.10 * a15:
            continue

        distance = (
            p - entry
        ) * side

        # Strong level before TP2.
        if (
            w >= 4
            and 0 < distance < 2.0 * risk
        ):
            blocked = True
            break

    if blocked:
        return (
            None,
            "مستوى قوي يعترض الطريق قبل TP2",
        )

    plan = {
        "entry": float(entry),
        "stop": float(stop),

        "tps": [
            float(tp1),
            float(tp2),
            float(tp3),
        ],

        "rr": 2.0,

        "arithmetic_extensions": True,
    }

    reason = cost_reason(
        key,
        plan,
    )

    if reason:
        return None, reason

    return plan, None


# =========================================================
# TREND BREAKOUT / RETEST ANALYSIS
# =========================================================

def analyze_trend_breakout(
    a,
    key,
    fr,
    btc_dir,
):
    d15 = fr["15m"]
    d1 = fr["1h"]
    d4 = fr["4h"]

    # =====================================================
    # DATA CHECK
    # =====================================================

    if (
        len(d15) < 60
        or len(d1) < 60
        or len(d4) < 25
    ):

        price = (
            float(d15["c"].iloc[-1])
            if len(d15)
            else 0.0
        )

        return base_result(
            key,
            1,
            price,
            0.0,
            "TrendBreakout",
            "بيانات غير كافية للتحليل.",
        )

    last = d15.iloc[-1]
    prev = d15.iloc[-2]

    price = float(
        last["c"]
    )

    # =====================================================
    # ATR
    # =====================================================

    a15 = float(
        a.atr(d15).iloc[-1]
    )

    a1 = float(
        a.atr(d1).iloc[-1]
    )

    if not all(
        math.isfinite(x)
        and x > 0
        for x in (
            a15,
            a1,
        )
    ):

        return base_result(
            key,
            1,
            price,
            0.0,
            "TrendBreakout",
            "تعذر حساب ATR.",
        )

    # =====================================================
    # 1H = MAIN DIRECTION
    # =====================================================

    trend_1h = a.trend(
        d1
    )

    trend_4h = a.trend(
        d4
    )

    # No trade while 1H is range.
    if trend_1h == 0:

        return base_result(
            key,
            (
                trend_4h
                if trend_4h != 0
                else 1
            ),
            price,
            0.0,
            "TrendBreakout",
            (
                "Trend Continuation\n"
                "1H: RANGE\n"
                "الحالة: لا دخول حالياً؛ "
                "ننتظر اتجاه واضح على الساعة."
            ),
        )

    side = trend_1h

    # =====================================================
    # FIXED 15M BREAKOUT LEVEL
    # =====================================================

    # We exclude current + previous candle.
    # This lets the previous candle be the breakout
    # and current candle be the retest.
    base = d15.iloc[-22:-2]

    if len(base) < 15:

        return base_result(
            key,
            side,
            price,
            0.0,
            "TrendBreakout",
            "بيانات 15m غير كافية.",
        )

    high = float(
        base["h"].max()
    )

    low = float(
        base["l"].min()
    )

    level = (
        high
        if side == 1
        else low
    )

    break_buffer = (
        BREAK_BUFFER_ATR
        * a15
    )

    touch_buffer = (
        RETEST_TOUCH_ATR
        * a15
    )

    close_buffer = (
        RETEST_CLOSE_ATR
        * a15
    )

    previous_close = float(
        prev["c"]
    )

    previous_open = float(
        prev["o"]
    )

    last_open = float(
        last["o"]
    )

    last_high = float(
        last["h"]
    )

    last_low = float(
        last["l"]
    )

    # =====================================================
    # LONG CONDITIONS
    # =====================================================

    if side == 1:

        previous_broke = (
            previous_close
            > high
            + break_buffer
        )

        fresh_breakout = (
            price
            > high
            + break_buffer
            and previous_close
            <= high
            + break_buffer
        )

        clean_retest = (
            previous_broke
            and last_low
            <= high
            + touch_buffer
            and price
            >= high
            + close_buffer
        )

        approaching = (
            0
            <= high - price
            <= WATCH_DISTANCE_ATR
            * a15
        )

        extension = max(
            0.0,
            (
                price
                - high
            )
            / a15,
        )

    # =====================================================
    # SHORT CONDITIONS
    # =====================================================

    else:

        previous_broke = (
            previous_close
            < low
            - break_buffer
        )

        fresh_breakout = (
            price
            < low
            - break_buffer
            and previous_close
            >= low
            - break_buffer
        )

        clean_retest = (
            previous_broke
            and last_high
            >= low
            - touch_buffer
            and price
            <= low
            - close_buffer
        )

        approaching = (
            0
            <= price - low
            <= WATCH_DISTANCE_ATR
            * a15
        )

        extension = max(
            0.0,
            (
                low
                - price
            )
            / a15,
        )

    # =====================================================
    # 15M EMA CONTEXT
    # =====================================================

    ema20 = float(
        a.ema(
            d15["c"],
            20,
        ).iloc[-1]
    )

    ema50 = float(
        a.ema(
            d15["c"],
            50,
        ).iloc[-1]
    )

    if side == 1:

        ema_ok = (
            price
            > ema20
            > ema50
        )

    else:

        ema_ok = (
            price
            < ema20
            < ema50
        )

    # =====================================================
    # VOLUME
    # =====================================================

    vr = volume_ratio(
        d15
    )

    strong_volume = (
        vr is not None
        and vr
        >= STRONG_VOLUME_RATIO
    )

    # =====================================================
    # CANDLE MOMENTUM
    # =====================================================

    body = abs(
        price
        - last_open
    )

    body_atr = (
        body
        / a15
    )

    strong_body = (
        body_atr
        >= STRONG_BODY_ATR
    )

    # Direct breakout requires BOTH:
    # clear body + clear volume.
    strong_breakout = (
        fresh_breakout
        and strong_volume
        and strong_body
    )

    # =====================================================
    # SCORE
    # =====================================================

    # 1H trend exists.
    score = 3.0

    # 4H is context only.
    if trend_4h == side:
        score += 1.0

    elif trend_4h == -side:
        score -= 0.5

    # Setup quality.
    if clean_retest:
        score += 3.0

    elif fresh_breakout:
        score += 2.5

    elif previous_broke:
        score += 1.5

    elif approaching:
        score += 1.0

    # 15m EMA bonus.
    if ema_ok:
        score += 1.0

    # Volume bonus.
    if vr is not None:

        if vr >= 1.50:
            score += 1.0

        elif vr >= 1.00:
            score += 0.5

        elif vr < 0.65:
            score -= 0.5

    # Candle momentum bonus.
    if strong_body:
        score += 0.5

    # BTC context for ETH/SOL only.
    # Bonus / penalty, NOT mandatory.
    if (
        key in ("ETH", "SOL")
        and btc_dir is not None
    ):

        if btc_dir == side:
            score += 0.5

        elif btc_dir == -side:
            score -= 0.5

    score = round(
        max(
            0.0,
            min(
                10.0,
                score,
            ),
        ),
        1,
    )

    # =====================================================
    # OBSERVATION TEXT
    # =====================================================

    if clean_retest:
        setup_name = (
            "Clean Retest"
        )

    elif fresh_breakout:
        setup_name = (
            "Fresh Breakout"
        )

    elif previous_broke:
        setup_name = (
            "Breakout — waiting retest"
        )

    elif approaching:
        setup_name = (
            "Approaching Breakout"
        )

    else:
        setup_name = (
            "No valid setup"
        )

    volume_text = (
        "غير متوفر"
        if vr is None
        else f"{vr:.2f}x"
    )

    obs = (
        "Trend Continuation + Breakout/Retest\n"
        f"الاتجاه: {direction_name(side)}\n"
        f"1H: {'BULLISH' if trend_1h == 1 else 'BEARISH'}"
        f" | 4H: "
        f"{'BULLISH' if trend_4h == 1 else ('BEARISH' if trend_4h == -1 else 'RANGE')}\n"
        f"15m: {setup_name}\n"
        f"المستوى: {level:,.2f}\n"
        f"EMA20/50: "
        f"{'متوافق' if ema_ok else 'غير متوافق'}\n"
        f"الفوليوم: {volume_text}\n"
        f"جسم الشمعة: {body_atr:.2f} ATR\n"
        f"الامتداد: {extension:.2f} ATR\n"
        "4H والفوليوم وEMA عوامل جودة؛ "
        "مو مجموعة شروط إجبارية."
    )

    res = base_result(
        key,
        side,
        price,
        score,
        "TrendBreakout",
        obs,
    )

    res.update(
        ext=round(
            extension,
            2,
        ),
        zone_p=level,
        a1=a1,
    )

    # =====================================================
    # 1. CLEAN RETEST -> A
    # =====================================================

    if (
        clean_retest
        and score >= A_MIN_SCORE
    ):

        plan, reason = (
            build_trade_plan(
                a,
                key,
                fr,
                side,
                price,
                level,
                a15,
                a1,
                "RETEST",
            )
        )

        if (
            plan
            and reason is None
        ):

            res.update(
                tier="A",
                style="TrendRetest",
                plan=plan,
            )

            res["obs"] += (
                "\n✅ ريتست نظيف للمستوى "
                "المكسور وتأكيد باتجاه الترند."
            )

            return res

        if reason:
            res["miss"] = [
                reason
            ]

            res["obs"] += (
                "\nسبب عدم الدخول: "
                + reason
            )

    # =====================================================
    # 2. STRONG FRESH BREAKOUT -> A
    # =====================================================

    if (
        strong_breakout
        and extension
        <= MAX_DIRECT_EXTENSION_ATR
        and score >= A_MIN_SCORE
    ):

        plan, reason = (
            build_trade_plan(
                a,
                key,
                fr,
                side,
                price,
                level,
                a15,
                a1,
                "BREAKOUT",
            )
        )

        if (
            plan
            and reason is None
        ):

            res.update(
                tier="A",
                style="TrendBreakout",
                plan=plan,
            )

            res["obs"] += (
                "\n✅ بريك أوت قوي: "
                "Momentum + Volume واضحين؛ "
                "دخول بدون انتظار ريتست."
            )

            return res

        if reason:
            res["miss"] = [
                reason
            ]

            res["obs"] += (
                "\nسبب عدم الدخول: "
                + reason
            )

    # =====================================================
    # 3. BREAKOUT -> ARMED RETEST
    # =====================================================

    if (
        fresh_breakout
        or previous_broke
    ):

        # If it ran too far, do not chase.
        if (
            extension
            > MAX_DIRECT_EXTENSION_ATR
        ):

            res["obs"] += (
                "\n⚠️ السعر ابتعد عن مستوى الكسر؛ "
                "ممنوع المطاردة. ننتظر ريتست."
            )

        plan, reason = (
            build_trade_plan(
                a,
                key,
                fr,
                side,
                level,
                level,
                a15,
                a1,
                "RETEST",
            )
        )

        if (
            plan
            and reason is None
            and score >= B_MIN_SCORE
        ):

            res.update(
                tier="ARMED",
                style="TrendBreakRetest",
                plan=plan,

                zone=[
                    float(
                        level
                        - RETEST_TOUCH_ATR
                        * a15
                    ),
                    float(
                        level
                        + RETEST_TOUCH_ATR
                        * a15
                    ),
                ],
            )

            res["obs"] += (
                "\n🟠 الكسر صالح، لكن ما نطارد السعر؛ "
                "الدخول فقط إذا رجع للمستوى."
            )

            return res

        if reason:

            res["miss"] = [
                reason
            ]

            res["obs"] += (
                "\nسبب عدم تسليح الريتست: "
                + reason
            )

    # =====================================================
    # 4. EARLY WATCH
    # =====================================================

    if (
        approaching
        and score >= B_MIN_SCORE
    ):

        if side == 1:

            trigger = (
                level
                + break_buffer
            )

            res["cond"] = (
                f"إغلاق 15m فوق "
                f"{trigger:,.2f}. "
                "إذا الكسر قوي بفوليوم ومومنتوم "
                "واضحين ممكن دخول مباشر؛ "
                "وإلا ننتظر Retest."
            )

            res["inv"] = (
                "فقدان اتجاه 1H الصاعد "
                "أو تحول الساعة إلى Range."
            )

        else:

            trigger = (
                level
                - break_buffer
            )

            res["cond"] = (
                f"إغلاق 15m تحت "
                f"{trigger:,.2f}. "
                "إذا الكسر قوي بفوليوم ومومنتوم "
                "واضحين ممكن دخول مباشر؛ "
                "وإلا ننتظر Retest."
            )

            res["inv"] = (
                "فقدان اتجاه 1H الهابط "
                "أو تحول الساعة إلى Range."
            )

        res.update(
            tier="B",
            style="TrendWatch",
        )

        return res

    # =====================================================
    # NO SETUP
    # =====================================================

    if not res["miss"]:

        if trend_1h == side:

            res["miss"] = [
                "لا يوجد Breakout أو Retest مؤهل حالياً"
            ]

    return res


# =========================================================
# MAIN ENTRY POINT USED BY scanner.py
# =========================================================

def analyze_sequence(
    a,
    st,
    key,
    fr,
    btc_dir,
):
    """
    This is the function scanner.py calls.

    Every asset now uses the SAME first-agent strategy:

        1H trend
             ↓
        15m breakout
             ↓
        strong breakout -> A
             OR
        normal breakout -> ARMED retest
             ↓
        clean retest -> A

    No SimpleZones / CISD / FVG / ICT requirement.
    """

    # Old confirmation setups belong to the previous strategy.
    # Clear them so they cannot interfere with the new logic.
    st.setdefault(
        "confirmation_setups",
        {}
    ).pop(
        key,
        None,
    )

    res = analyze_trend_breakout(
        a,
        key,
        fr,
        btc_dir,
    )

    # Respect spot-only mode if enabled.
    if (
        getattr(
            a,
            "SPOT_ONLY",
            False,
        )
        and res["side"] == -1
    ):

        res["tier"] = None

        res["miss"] = [
            "SHORT disabled by SPOT_ONLY"
        ]

        res["obs"] += (
            "\nSHORT ملغي لأن SPOT_ONLY مفعّل."
        )

    return res