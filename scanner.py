import hashlib
import importlib
import logging
import math
import os
import sys
import uuid

import pandas as pd
import requests

import scanner_data as md
import news


LOG = logging.getLogger(__name__)
VERSION = "paper-v2"


def stamp(value):
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def event(st, text, identity=None):
    identity = identity or str(uuid.uuid4())

    if identity in st.setdefault("event_ids", []):
        return

    st["event_ids"].append(identity)
    st["event_ids"] = st["event_ids"][-5000:]

    chunks = []
    buffer = ""
    units = 0

    for character in text:
        size = len(
            character.encode("utf-16-le")
        ) // 2

        if units + size > 3500:
            chunks.append(buffer)
            buffer = ""
            units = 0

        buffer += character
        units += size

    if buffer:
        chunks.append(buffer)

    for index, chunk in enumerate(chunks):
        event_id = f"{identity}:{index + 1}"

        st.setdefault("outbox", []).append({
            "id": event_id,
            "text": chunk + "\nحدث: " + event_id,
        })


def flush(st, save, token, chat):
    save(st)

    while st.get("outbox"):
        item = st["outbox"][0]

        try:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat,
                    "text": item["text"],
                },
                timeout=(8, 25),
            )

            body = response.json()

        except Exception:
            LOG.error(
                "Telegram send failed; "
                "durable outbox retained"
            )
            return False

        if not response.ok or body.get("ok") is not True:
            LOG.error(
                "Telegram rejected delivery: "
                "HTTP %s code=%s; outbox retained",
                response.status_code,
                body.get("error_code"),
            )
            return False

        st["outbox"].pop(0)
        save(st)

    return True


def cost_r(sp):
    risk = abs(sp["entry"] - sp["stop0"])

    return (
        sp["entry"]
        * sp.get("cost_bps", 0.0)
        / 10000
        / risk
    )


def close_trade(st, sp, exit_price, label, when):
    side = sp["side"]
    risk = abs(sp["entry"] - sp["stop0"])

    completed = sum(
        (
            (sp["tps"][index] - sp["entry"])
            * side / risk / 3
        )
        for index in range(3)
        if sp["hits"][index]
    )

    remaining = (3 - sum(sp["hits"])) / 3

    gross = completed + (
        remaining
        * (exit_price - sp["entry"])
        * side / risk
    )

    sp.update(
        state="CLOSED",
        r=round(gross - cost_r(sp), 4),
        gross_r=round(gross, 4),
        exit=label,
        t_close=stamp(when).isoformat(),
        exit_price=exit_price,
    )

    st["closed"].append(
        dict(sp, hits=list(sp["hits"]))
    )

    st["cool"][sp["key"]] = stamp(
        when
    ).isoformat()

    ambiguity = (
        "\nترتيب الحركة داخل شمعة ملتبس؛ "
        "النتيجة حسب سياسة محافظة."
        if sp.get("ambiguous")
        else ""
    )

    event(
        st,
        f"📋 {sp['id']} | إغلاق ورقي: {label}\n"
        f"النتيجة الإجمالية {gross:+.2f}R | "
        f"بعد الكلفة المفترضة {sp['r']:+.2f}R"
        + ambiguity,
        sp["id"] + ":closed",
    )


def make_setup(st, res, fr, now):
    plan = res["plan"]
    side = res["side"]
    entry = float(plan["entry"])
    stop = float(plan["stop"])
    targets = list(map(float, plan["tps"]))

    if len(targets) != 3:
        raise ValueError(
            "exactly three targets required"
        )

    if not all(
        math.isfinite(value)
        for value in [entry, stop] + targets
    ):
        raise ValueError("nonfinite plan")

    if (
        (entry - stop) * side <= 0
        or not all(
            (target - entry) * side > 0
            for target in targets
        )
    ):
        raise ValueError("invalid plan geometry")

    if not all(
        (targets[index + 1] - targets[index]) * side > 0
        for index in (0, 1)
    ):
        raise ValueError(
            "targets must be ordered"
        )

    st["n"] += 1

    return {
        "id": f"{res['key']}-{st['n']:03d}",
        "key": res["key"],
        "side": side,
        "style": res["style"],
        "tf": res.get("tf", "15m"),
        "entry": entry,
        "reference_entry": entry,
        "stop": stop,
        "stop0": stop,
        "tps": targets,
        "hits": [False, False, False],
        "score": res["score"],
        "rr": abs(targets[1] - entry) / abs(entry - stop),
        "state": (
            "ARMED"
            if res["tier"] == "ARMED"
            else "PENDING"
        ),
        "t_decl": now.isoformat(),
        "t0": now.isoformat(),
        "not_before": now.ceil("15min").isoformat(),
        "expires": (
            now + pd.Timedelta(hours=4)
        ).isoformat(),
        "source": fr["_source"],
        "engine_version": VERSION,
        "cost_bps": float(
            os.environ.get(
                "ROUNDTRIP_COST_BPS", "14"
            )
        ),
        "zone": res.get("zone"),
        "ambiguous": False,
    }


def manage(st, sp, fr, now, agent):
    key = sp["key"]
    side = sp["side"]
    sp.setdefault("ambiguous", False)

    if not sp.get("source"):
        if key in ("BTC", "ETH", "SOL"):
            sp["source"] = "binance:" + key + "USDT"
            sp["engine_version"] = "legacy-migrated"
            sp.setdefault("cost_bps", 0.0)

            sp.setdefault(
                "not_before",
                (
                    stamp(sp["t_decl"])
                    + pd.Timedelta(minutes=15)
                ).isoformat(),
            )

            sp.setdefault(
                "expires",
                (
                    stamp(sp["t_decl"])
                    + pd.Timedelta(hours=4)
                ).isoformat(),
            )

        else:
            event(
                st,
                f"⚠️ {sp['id']}: مصدر الصفقة القديمة "
                "غير مسجل؛ توقفت متابعة هذه الصفقة "
                "لحين تحديد مصدرها. السجل محفوظ، "
                "ولا تُفتح صفقة أخرى على نفس الأصل.",
                sp["id"] + ":source-unknown",
            )
            return

    if sp["source"] != fr["_source"]:
        event(
            st,
            f"⚠️ {sp['id']}: المصدر تغيّر من "
            f"{sp['source']} إلى {fr['_source']}؛ "
            "المتابعة معلقة بدون احتساب نتيجة.",
            sp["id"] + ":source-change:" + fr["_source"],
        )
        return

    data = fr["15m"]
    first = stamp(
        sp.get("not_before", sp["t0"])
    )

    rows = data[data.index >= first]

    if sp.get("seen"):
        rows = rows[
            rows.index > stamp(sp["seen"])
        ]

    expected = (
        stamp(sp["seen"]) + pd.Timedelta(minutes=15)
        if sp.get("seen")
        else first
    )

    if key in ("BTC", "ETH", "SOL") and not rows.empty:
        indexes = list(rows.index)

        missing = (
            indexes[0] > expected
            or any(
                later - earlier != pd.Timedelta(minutes=15)
                for earlier, later in zip(
                    indexes, indexes[1:]
                )
            )
        )

        if missing:
            event(
                st,
                f"⚠️ {sp['id']}: فجوة ببيانات التتبع؛ "
                "النتيجة معلقة لحين استرجاع "
                "الشموع المفقودة.",
                sp["id"] + ":gap",
            )
            return

    for timestamp, candle in rows.iterrows():
        end = stamp(candle["end"])
        opening = float(candle["o"])
        high = float(candle["h"])
        low = float(candle["l"])

        if sp["state"] in ("PENDING", "ARMED"):
            if end > stamp(sp["expires"]):
                sp["state"] = "EXPIRED"

                event(
                    st,
                    f"⌛ {sp['id']}: انتهت صلاحية "
                    "الإعداد بدون دخول.",
                    sp["id"] + ":expired",
                )
                return

            if sp["state"] == "ARMED":
                if (opening - sp["stop"]) * side <= 0:
                    sp["state"] = "INVALIDATED"

                    event(
                        st,
                        f"❌ {sp['id']}: افتتاح تجاوز "
                        "الإلغاء قبل الدخول.",
                        sp["id"] + ":invalid",
                    )
                    return

                if (opening - sp["entry"]) * side <= 0:
                    entry = opening

                elif low <= sp["entry"] <= high:
                    entry = sp["entry"]

                else:
                    sp["seen"] = timestamp.isoformat()
                    continue

                sp["ambiguous"] = (
                    low <= sp["stop"] <= high
                )

            else:
                entry = opening

                reference_risk = abs(
                    sp["reference_entry"] - sp["stop0"]
                )

                invalid = (
                    abs(entry - sp["reference_entry"])
                    > 0.35 * reference_risk

                    or (entry - sp["stop0"]) * side <= 0

                    or (sp["tps"][0] - entry) * side <= 0

                    or (
                        (sp["tps"][1] - entry)
                        * side
                        / abs(entry - sp["stop0"])
                    ) < 2
                )

                if invalid:
                    sp["state"] = "INVALIDATED"

                    event(
                        st,
                        f"❌ {sp['id']}: أُلغي قبل الدخول "
                        "بسبب تغيّر سعر الافتتاح "
                        "أو ضعف R:R.",
                        sp["id"] + ":invalid",
                    )
                    return

            sp.update(
                entry=entry,
                state="ENTERED",
                t0=timestamp.isoformat(),
            )

            if cost_r(sp) > 0.35:
                sp["state"] = "INVALIDATED"

                event(
                    st,
                    f"❌ {sp['id']}: كلفة التنفيذ المفترضة "
                    "كبيرة مقارنة بالوقف؛ "
                    "لم تُفتح الصفقة.",
                    sp["id"] + ":invalid",
                )
                return

            sp["rr"] = (
                abs(sp["tps"][1] - entry)
                / abs(entry - sp["stop0"])
            )

            event(
                st,
                f"📍 {sp['id']} | دخول ورقي مؤكّد بالتتبع\n"
                f"دخول {agent.fmt(entry)} | "
                f"وقف {agent.fmt(sp['stop'])}\n"
                f"TP1 {agent.fmt(sp['tps'][0])} | "
                f"TP2 {agent.fmt(sp['tps'][1])} | "
                f"TP3 {agent.fmt(sp['tps'][2])}\n"
                f"R:R إلى TP2: {sp['rr']:.2f}",
                sp["id"] + ":entered",
            )

        sp["seen"] = timestamp.isoformat()

        stop_hit = (
            low <= sp["stop"]
            if side == 1
            else high >= sp["stop"]
        )

        next_targets = [
            index for index in range(3)
            if not sp["hits"][index]
        ]

        any_target = any(
            (
                high >= sp["tps"][index]
                if side == 1
                else low <= sp["tps"][index]
            )
            for index in next_targets
        )

        if stop_hit:
            sp["ambiguous"] |= any_target

            exit_price = (
                min(opening, sp["stop"])
                if side == 1
                else max(opening, sp["stop"])
            )

            if (
                sp.get("ambiguous")
                and timestamp == stamp(sp["t0"])
            ):
                exit_price = sp["stop"]

            close_trade(
                st, sp, exit_price, "STOP", end
            )
            return

        for index in next_targets:
            hit = (
                high >= sp["tps"][index]
                if side == 1
                else low <= sp["tps"][index]
            )

            if not hit:
                break

            sp["hits"][index] = True

            event(
                st,
                f"🎯 {sp['id']} | TP{index + 1} تحقق @ "
                f"{agent.fmt(sp['tps'][index])}",
                sp["id"] + f":tp{index + 1}",
            )

            if index == 2:
                close_trade(
                    st, sp, sp["tps"][2], "TP3", end
                )
                return

            sp["stop"] = (
                sp["entry"]
                if index == 0
                else sp["tps"][0]
            )

            event(
                st,
                f"🔒 {sp['id']} | الوقف الورقي صار "
                f"{agent.fmt(sp['stop'])}",
                sp["id"] + f":sl{index + 1}",
            )

            moved_stop_hit = (
                low <= sp["stop"]
                if side == 1
                else high >= sp["stop"]
            )

            if moved_stop_hit:
                sp["ambiguous"] = True

                close_trade(
                    st,
                    sp,
                    sp["stop"],
                    "STOP (OHLC ambiguity)",
                    end,
                )
                return

        hours = {
            "15m": 24,
            "1h": 72,
            "4h": 240,
        }.get(sp.get("tf", "15m"), 24)

        if sp["hits"][0]:
            hours *= 2

        if (
            end - stamp(sp["t0"])
            >= pd.Timedelta(hours=hours)
        ):
            close_trade(
                st,
                sp,
                float(candle["c"]),
                "TIME EXIT",
                end,
            )
            return

    if (
        sp["state"] in ("PENDING", "ARMED")
        and now > stamp(sp["expires"])
    ):
        sp["state"] = "EXPIRED"

        event(
            st,
            f"⌛ {sp['id']}: انتهت صلاحية الإعداد "
            "بدون دخول.",
            sp["id"] + ":expired",
        )


def setup_text(res, sp, fr, agent):
    direction = (
        "LONG" if sp["side"] == 1 else "SHORT"
    )

    targets = sp["tps"]

    title = (
        "انتظار ريتست بسعر محدد"
        if sp["state"] == "ARMED"
        else "إعداد A — ينتظر افتتاح شمعة لاحقة"
    )

    ratios = [
        abs(target - sp["entry"])
        / abs(sp["entry"] - sp["stop0"])
        for target in targets
    ]

    return (
        f"🟠 {title} | "
        f"{agent.ASSETS[sp['key']][2]} {direction}\n"
        f"ID: {sp['id']}\n"
        f"النوع: {sp['style']} | الفريم: {sp['tf']} | "
        f"جودة الإعداد: {sp['score']}/10\n"
        f"المصدر/الأداة: {fr['_source']}\n"
        "آخر إغلاق 15m: "
        f"{fr['15m']['end'].iloc[-1].isoformat()}\n"
        f"سعر مرجعي {agent.fmt(sp['entry'])} | "
        f"وقف {agent.fmt(sp['stop'])}\n"
        f"TP1 {agent.fmt(targets[0])} "
        f"({ratios[0]:.2f}R)\n"
        f"TP2 {agent.fmt(targets[1])} "
        f"({ratios[1]:.2f}R)\n"
        f"TP3 {agent.fmt(targets[2])} "
        f"({ratios[2]:.2f}R)\n"
        "R:R المعروض إلى TP2، قبل الكلفة | "
        "كلفة ذهاب وعودة مفترضة "
        f"{sp['cost_bps']:g} نقطة أساس\n"
        f"{res['obs']}\n"
        "الأهداف قد تتضمن امتدادات حسابية (+1R) "
        "وليست كلها مستويات مرصودة.\n"
        "التأكيد بالتتبع برسالة منفصلة؛ افتراضي فقط، "
        "بدون تنفيذ أو ضمان ربح."
    )


def weekly(st, now):
    due = (
        now.normalize()
        - pd.Timedelta(
            days=(now.weekday() - 4) % 7
        )
        + pd.Timedelta(hours=21)
    )

    if due > now:
        due -= pd.Timedelta(days=7)

    identity = "weekly-v2:" + due.isoformat()

    if identity in st.get("event_ids", []):
        return

    closed = [
        item for item in st["closed"]
        if (
            item.get("engine_version") == VERSION
            and (
                due - pd.Timedelta(days=7)
                < stamp(item["t_close"])
                <= due
            )
        )
    ]

    if not closed:
        return

    net = sum(item["r"] for item in closed)

    ambiguous = sum(
        bool(item.get("ambiguous"))
        for item in closed
    )

    event(
        st,
        "📋 تقرير أسبوع ورقي v2\n"
        f"الصفقات المغلقة {len(closed)} | "
        f"صافي مفترض {net:+.2f}R\n"
        f"حالات OHLC الملتبسة {ambiguous} | "
        "كل هدف يغلق ثلث الحجم.\n"
        "الكلفة تقديرية والعينة لا تثبت "
        "تفوق الاستراتيجية. النتائج القديمة منفصلة.",
        identity,
    )


def run(agent, now=None):
    now = (
        pd.Timestamp.now(tz="UTC")
        if now is None
        else stamp(now)
    )

    st = agent.load()

    if not isinstance(
        st.setdefault("outbox", []), list
    ):
        raise RuntimeError(
            "invalid outbox; state preserved"
        )

    token = os.environ.get("TG_TOKEN")
    chat = os.environ.get("TG_CHAT")

    if not token or not chat:
        raise RuntimeError(
            "TG_TOKEN and TG_CHAT required"
        )

    if not flush(st, agent.save, token, chat):
        return 1

    data = {}
    summary = []

    for key in agent.ASSETS:
        try:
            data[key] = md.frames(key, now)

        except Exception as exc:
            reason = md.safe_error(exc)

            LOG.error(
                "DATA asset=%s ERROR %s",
                key, reason,
            )

            summary.append(f"{key}: {reason}")

            last = st.setdefault(
                "health", {}
            ).get(key)

            if (
                not last
                or now - stamp(last)
                > pd.Timedelta(hours=6)
            ):
                event(
                    st,
                    f"⚠️ بيانات {key} غير متوفرة: "
                    f"{reason}\n"
                    "لا إشارة دخول جديدة لهذا الأصل.",
                )

                st["health"][key] = now.isoformat()

    btc_direction = None

    if (
        "BTC" in data
        and md.freshness(
            data["BTC"], "BTC", now
        )[0]
    ):
        trend4 = agent.trend(data["BTC"]["4h"])
        trend1 = agent.trend(data["BTC"]["1h"])

        btc_direction = (
            trend4 if trend4 == trend1 else 0
        )

    news_context = news.get_news(
        st,
        now,
        os.environ.get("GEMINI_KEY", ""),
    )

    for key, frames in data.items():
        fresh, reason = md.freshness(
            frames, key, now
        )

        last_close = stamp(
            frames["15m"]["end"].iloc[-1]
        )

        position = st["active"].get(key)

        if position:
            manage(
                st,
                position,
                frames,
                now,
                agent,
            )

            if position["state"] in (
                "CLOSED", "INVALIDATED", "EXPIRED"
            ):
                del st["active"][key]

            summary.append(
                f"{key}: {position['state']}"
            )
            continue

        if not fresh:
            LOG.warning(
                "SKIP asset=%s %s", key, reason
            )
            summary.append(f"{key}: {reason}")
            continue

        cooldown = st["cool"].get(key)

        if (
            cooldown
            and now - stamp(cooldown)
            < pd.Timedelta(hours=2)
        ):
            summary.append(f"{key}: cooldown")
            continue

        previous_analysis = st.setdefault(
            "last_analysis", {}
        ).get(key)

        if previous_analysis == last_close.isoformat():
            summary.append(
                f"{key}: no new closed candle"
            )
            continue

        try:
            result = agent.analyze(
                key, frames, btc_direction
            )

            result = news.apply(
                result, news_context
            )

            st["last_analysis"][key] = (
                last_close.isoformat()
            )

            summary.append(
                f"{key}: {result['tier'] or 'NONE'} "
                f"score={result['score']}"
            )

            LOG.info(
                "ANALYSIS asset=%s tier=%s "
                "score=%s close=%s",
                key,
                result["tier"],
                result["score"],
                last_close,
            )

            if result["tier"] in ("A", "ARMED"):
                position = make_setup(
                    st, result, frames, now
                )

                if (
                    position["cost_bps"] < 0
                    or not math.isfinite(
                        position["cost_bps"]
                    )
                ):
                    raise ValueError(
                        "invalid cost assumption"
                    )

                if cost_r(position) > 0.35:
                    LOG.info(
                        "SKIP %s execution cost too "
                        "high relative to risk",
                        key,
                    )
                    continue

                st["active"][key] = position

                st["watch"].pop(key, None)
                st["watch"].pop(key + ":E", None)

                event(
                    st,
                    setup_text(
                        result,
                        position,
                        frames,
                        agent,
                    ),
                    position["id"] + ":setup",
                )

            elif result["tier"] in ("B", "EARLY"):
                watch_key = key + (
                    ":E"
                    if result["tier"] == "EARLY"
                    else ""
                )

                old = st["watch"].get(watch_key)

                fingerprint_text = (
                    f"{result['side']}:"
                    f"{result.get('style')}:"
                    f"{result.get('zone_p', 0):.2f}"
                )

                fingerprint = hashlib.sha256(
                    fingerprint_text.encode()
                ).hexdigest()[:12]

                should_send = (
                    not old
                    or old.get("fingerprint") != fingerprint
                    or now - stamp(old["ts"])
                    >= pd.Timedelta(hours=6)
                )

                if should_send:
                    name = agent.ASSETS[key][2]

                    text = (
                        agent.msg_early(result, name)
                        if result["tier"] == "EARLY"
                        else agent.msg_watch(result, name)
                    )

                    text = text.replace(
                        "السعر الآن:", "آخر إغلاق:"
                    ).replace(
                        "السعر:", "آخر إغلاق:"
                    )

                    text += (
                        "\nالمصدر: "
                        + frames["_source"]
                        + "\nوقت الإغلاق: "
                        + last_close.isoformat()
                    )

                    event(st, text)

                    st["watch"][watch_key] = {
                        "side": result["side"],
                        "ts": now.isoformat(),
                        "fingerprint": fingerprint,
                    }

            elif result["score"] < 5.5:
                st["watch"].pop(key, None)

        except Exception as exc:
            LOG.error(
                "ANALYZE %s %s",
                key,
                md.safe_error(exc),
            )

            event(
                st,
                f"⚠️ خطأ تحليل {key}: "
                f"{md.safe_error(exc)}؛ "
                "لم تُصدر إشارة جديدة.",
            )

    for text in news.alerts(
        news_context, st, now
    ):
        event(st, text)

    weekly(st, now)

    if os.environ.get("EVENT") == "workflow_dispatch":
        event(
            st,
            "ℹ️ فحص يدوي\n"
            + "\n".join(summary)
            + "\n"
            + news_context["status"],
        )

    return (
        0 if flush(st, agent.save, token, chat)
        else 1
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    agent = importlib.import_module("agent")

    try:
        return run(agent)

    except Exception as exc:
        LOG.error(
            "Scanner failed: %s",
            md.safe_error(exc),
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())