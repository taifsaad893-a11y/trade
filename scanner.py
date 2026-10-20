"""Paper scanner. Run upgrade_agent.py first."""
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
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def event(st, text, identity=None):
    identity = identity or str(uuid.uuid4())

    if identity in st.setdefault("event_ids", []):
        return

    st["event_ids"].append(identity)
    st["event_ids"] = st["event_ids"][-5000:]

    chunks, buf, units = [], "", 0

    for ch in text:
        size = len(ch.encode("utf-16-le")) // 2

        if units + size > 3500:
            chunks.append(buf)
            buf, units = "", 0

        buf += ch
        units += size

    if buf:
        chunks.append(buf)

    for i, chunk in enumerate(chunks):
        eid = f"{identity}:{i + 1}"
        st.setdefault("outbox", []).append({
            "id": eid,
            "text": chunk + "\nحدث: " + eid,
        })


def flush(st, save, token, chat):
    save(st)

    while st.get("outbox"):
        item = st["outbox"][0]

        try:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": item["text"]},
                timeout=(8, 25),
            )
            body = response.json()
        except Exception:
            LOG.error(
                "Telegram send failed; durable outbox retained"
            )
            return False

        if not response.ok or body.get("ok") is not True:
            LOG.error(
                "Telegram rejected delivery: HTTP %s code=%s; "
                "outbox retained",
                response.status_code,
                body.get("error_code"),
            )
            return False

        st["outbox"].pop(0)
        save(st)

    return True


def cost_r(sp):
    return (
        sp["entry"] * sp.get("cost_bps", 0.0)
        / 10000
        / abs(sp["entry"] - sp["stop0"])
    )


def close_trade(st, sp, exit_price, label, when):
    side = sp["side"]
    risk = abs(sp["entry"] - sp["stop0"])

    done = sum(
        (sp["tps"][i] - sp["entry"]) * side / risk / 3
        for i in range(3)
        if sp["hits"][i]
    )
    left = (3 - sum(sp["hits"])) / 3
    gross = done + left * (exit_price - sp["entry"]) * side / risk

    sp.update(
        state="CLOSED",
        r=round(gross - cost_r(sp), 4),
        gross_r=round(gross, 4),
        exit=label,
        t_close=stamp(when).isoformat(),
        exit_price=exit_price,
    )

    st["closed"].append(dict(sp, hits=list(sp["hits"])))
    st["cool"][sp["key"]] = stamp(when).isoformat()

    text = (
        f"📋 {sp['id']} | إغلاق ورقي: {label}\n"
        f"النتيجة الإجمالية {gross:+.2f}R | "
        f"بعد الكلفة المفترضة {sp['r']:+.2f}R"
    )

    if sp.get("ambiguous"):
        text += (
            "\nترتيب الحركة داخل شمعة ملتبس؛ "
            "النتيجة حسب سياسة محافظة."
        )

    event(st, text, sp["id"] + ":closed")


def assumed_cost_bps(key):
    value = float(os.environ.get(
        "ROUNDTRIP_COST_BPS_" + key,
        os.environ.get("ROUNDTRIP_COST_BPS", "14"),
    ))

    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid cost assumption for " + key)

    return value


def decision_text(key, res, sp=None):
    if sp is not None:
        cost = cost_r(sp)

        if cost > 0.35:
            return (
                f"{key}: SKIP_COST score={res['score']} | "
                f"كلفة مفترضة {cost:.3f}R > 0.350R "
                f"({sp['cost_bps']:g} bps)"
            )

        return (
            f"{key}: {sp['state']} score={res['score']} | "
            f"{sp['id']}"
        )

    return (
        f"{key}: {res['tier'] or 'NONE'} "
        f"score={res['score']}"
    )


def make_setup(st, res, fr, now):
    plan = res["plan"]
    side = res["side"]
    entry = float(plan["entry"])
    stop = float(plan["stop"])
    targets = list(map(float, plan["tps"]))

    if len(targets) != 3:
        raise ValueError("exactly three targets required")

    if not all(
        math.isfinite(x)
        for x in [entry, stop] + targets
    ):
        raise ValueError("nonfinite plan")

    if (
        (entry - stop) * side <= 0
        or not all((x - entry) * side > 0 for x in targets)
    ):
        raise ValueError("invalid plan geometry")

    if not all(
        (targets[i + 1] - targets[i]) * side > 0
        for i in (0, 1)
    ):
        raise ValueError("targets must be ordered")

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
        "hits": [False] * 3,
        "score": res["score"],
        "rr": abs(targets[1] - entry) / abs(entry - stop),
        "state": (
            "ARMED" if res["tier"] == "ARMED" else "PENDING"
        ),
        "t_decl": now.isoformat(),
        "t0": now.isoformat(),
        "not_before": now.ceil("15min").isoformat(),
        "expires": (
            now + pd.Timedelta(hours=4)
        ).isoformat(),
        "source": fr["_source"],
        "engine_version": VERSION,
        "strategy_version": res.get(
            "strategy_version", "reviewed_core"
        ),
        "cost_bps": assumed_cost_bps(res["key"]),
        "zone": res.get("zone"),
        "ambiguous": False,
    }


def manage(st, sp, fr, now, a):
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
                f"⚠️ {sp['id']}: مصدر الصفقة القديمة غير مسجل؛ "
                "توقفت متابعة هذه الصفقة لحين تحديد مصدرها. "
                "السجل محفوظ، ولا تُفتح صفقة أخرى على نفس الأصل.",
                sp["id"] + ":source-unknown",
            )
            return

    if sp["source"] != fr["_source"]:
        event(
            st,
            f"⚠️ {sp['id']}: المصدر تغيّر من {sp['source']} "
            f"إلى {fr['_source']}؛ "
            "المتابعة معلقة بدون احتساب نتيجة.",
            sp["id"] + ":source-change:" + fr["_source"],
        )
        return

    d = fr["15m"]
    first = stamp(sp.get("not_before", sp["t0"]))
    rows = d[d.index >= first]

    if sp.get("seen"):
        rows = rows[rows.index > stamp(sp["seen"])]

    expected = (
        stamp(sp["seen"]) + pd.Timedelta(minutes=15)
        if sp.get("seen") else first
    )

    if key in ("BTC", "ETH", "SOL") and not rows.empty:
        indexes = list(rows.index)

        if (
            indexes[0] > expected
            or any(
                right - left != pd.Timedelta(minutes=15)
                for left, right in zip(indexes, indexes[1:])
            )
        ):
            event(
                st,
                f"⚠️ {sp['id']}: فجوة ببيانات التتبع؛ "
                "النتيجة معلقة لحين استرجاع الشموع المفقودة.",
                sp["id"] + ":gap",
            )
            return

    for ts, bar in rows.iterrows():
        end = stamp(bar["end"])
        opening = float(bar["o"])
        high = float(bar["h"])
        low = float(bar["l"])

        if sp["state"] in ("PENDING", "ARMED"):
            if end > stamp(sp["expires"]):
                sp["state"] = "EXPIRED"
                event(
                    st,
                    f"⌛ {sp['id']}: انتهت صلاحية الإعداد بدون دخول.",
                    sp["id"] + ":expired",
                )
                return

            entry_mode = "NEXT_OPEN"

            if sp["state"] == "ARMED":
                if (opening - sp["stop"]) * side <= 0:
                    sp["state"] = "INVALIDATED"
                    event(
                        st,
                        f"❌ {sp['id']}: افتتاح تجاوز الإلغاء قبل الدخول.",
                        sp["id"] + ":invalid",
                    )
                    return

                if (opening - sp["entry"]) * side <= 0:
                    entry = opening
                elif low <= sp["entry"] <= high:
                    entry = sp["entry"]
                    entry_mode = "LIMIT_INTRABAR"
                else:
                    sp["seen"] = ts.isoformat()
                    continue

                sp["ambiguous"] = low <= sp["stop"] <= high

            else:
                entry = opening
                reference_risk = abs(
                    sp["reference_entry"] - sp["stop0"]
                )

                if (
                    abs(entry - sp["reference_entry"])
                    > 0.35 * reference_risk
                    or (entry - sp["stop0"]) * side <= 0
                    or (sp["tps"][0] - entry) * side <= 0
                    or (
                        (sp["tps"][1] - entry) * side
                        / abs(entry - sp["stop0"]) < 2
                    )
                ):
                    sp["state"] = "INVALIDATED"
                    event(
                        st,
                        f"❌ {sp['id']}: أُلغي قبل الدخول بسبب تغيّر "
                        "سعر الافتتاح أو ضعف R:R.",
                        sp["id"] + ":invalid",
                    )
                    return

            sp.update(
                entry=entry,
                state="ENTERED",
                t0=ts.isoformat(),
                entry_mode=entry_mode,
            )

            if cost_r(sp) > 0.35:
                sp["state"] = "INVALIDATED"
                event(
                    st,
                    f"❌ {sp['id']}: كلفة التنفيذ المفترضة كبيرة "
                    "مقارنة بالوقف؛ لم تُفتح الصفقة.",
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
                f"دخول {a.fmt(entry)} | وقف {a.fmt(sp['stop'])}\n"
                f"TP1 {a.fmt(sp['tps'][0])} | "
                f"TP2 {a.fmt(sp['tps'][1])} | "
                f"TP3 {a.fmt(sp['tps'][2])}\n"
                f"R:R إلى TP2: {sp['rr']:.2f}",
                sp["id"] + ":entered",
            )

        sp["seen"] = ts.isoformat()

        stop_hit = (
            low <= sp["stop"]
            if side == 1
            else high >= sp["stop"]
        )
        next_targets = [
            i for i in range(3)
            if not sp["hits"][i]
        ]
        any_target = any(
            high >= sp["tps"][i]
            if side == 1
            else low <= sp["tps"][i]
            for i in next_targets
        )

        if stop_hit:
            sp["ambiguous"] |= any_target
            exit_price = (
                min(opening, sp["stop"])
                if side == 1
                else max(opening, sp["stop"])
            )

            if (
                sp.get("entry_mode") == "LIMIT_INTRABAR"
                and ts == stamp(sp["t0"])
            ):
                exit_price = sp["stop"]

            close_trade(st, sp, exit_price, "STOP", end)
            return

        for i in next_targets:
            hit = (
                high >= sp["tps"][i]
                if side == 1
                else low <= sp["tps"][i]
            )

            if not hit:
                break

            sp["hits"][i] = True

            event(
                st,
                f"🎯 {sp['id']} | TP{i + 1} تحقق "
                f"@ {a.fmt(sp['tps'][i])}",
                sp["id"] + f":tp{i + 1}",
            )

            if i == 2:
                close_trade(st, sp, sp["tps"][2], "TP3", end)
                return

            sp["stop"] = (
                sp["entry"] if i == 0 else sp["tps"][0]
            )

            event(
                st,
                f"🔒 {sp['id']} | الوقف الورقي صار "
                f"{a.fmt(sp['stop'])}",
                sp["id"] + f":sl{i + 1}",
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

        hours *= 2 if sp["hits"][0] else 1

        if end - stamp(sp["t0"]) >= pd.Timedelta(hours=hours):
            close_trade(
                st, sp, float(bar["c"]), "TIME EXIT", end
            )
            return

    if (
        sp["state"] in ("PENDING", "ARMED")
        and now > stamp(sp["expires"])
    ):
        sp["state"] = "EXPIRED"
        event(
            st,
            f"⌛ {sp['id']}: انتهت صلاحية الإعداد بدون دخول.",
            sp["id"] + ":expired",
        )


def setup_text(res, sp, fr, a):
    direction = "LONG" if sp["side"] == 1 else "SHORT"
    targets = sp["tps"]
    title = (
        "انتظار ريتست بسعر محدد"
        if sp["state"] == "ARMED"
        else "إعداد A — ينتظر افتتاح شمعة لاحقة"
    )
    distances = [
        abs(x - sp["entry"]) / abs(sp["entry"] - sp["stop0"])
        for x in targets
    ]

    return (
        f"🟠 {title} | {a.ASSETS[sp['key']][2]} {direction}\n"
        f"ID: {sp['id']}\n"
        f"النوع: {sp['style']} | الفريم: {sp['tf']} | "
        f"جودة الإعداد: {sp['score']}/10\n"
        f"المصدر/الأداة: {fr['_source']}\n"
        f"آخر إغلاق 15m: "
        f"{fr['15m']['end'].iloc[-1].isoformat()}\n"
        f"سعر مرجعي {a.fmt(sp['entry'])} | "
        f"وقف {a.fmt(sp['stop'])}\n"
        f"TP1 {a.fmt(targets[0])} ({distances[0]:.2f}R)\n"
        f"TP2 {a.fmt(targets[1])} ({distances[1]:.2f}R)\n"
        f"TP3 {a.fmt(targets[2])} ({distances[2]:.2f}R)\n"
        "R:R المعروض إلى TP2، قبل الكلفة | "
        f"كلفة ذهاب وعودة مفترضة {sp['cost_bps']:g} نقطة أساس\n"
        f"{res['obs']}\n"
        "الأهداف قد تتضمن امتدادات حسابية (+1R) "
        "وليست كلها مستويات مرصودة.\n"
        "التأكيد بالتتبع برسالة منفصلة؛ افتراضي فقط، "
        "بدون تنفيذ أو ضمان ربح."
    )


def weekly(st, now):
    due = (
        now.normalize()
        - pd.Timedelta(days=(now.weekday() - 4) % 7)
        + pd.Timedelta(hours=21)
    )

    if due > now:
        due -= pd.Timedelta(days=7)

    identity = "weekly-v2:" + due.isoformat()

    if identity in st.get("event_ids", []):
        return

    closed = [
        trade for trade in st["closed"]
        if trade.get("engine_version") == VERSION
        and due - pd.Timedelta(days=7)
        < stamp(trade["t_close"]) <= due
    ]

    if not closed:
        return

    net = sum(trade["r"] for trade in closed)
    ambiguous = sum(
        bool(trade.get("ambiguous")) for trade in closed
    )

    event(
        st,
        f"📋 تقرير أسبوع ورقي v2\n"
        f"الصفقات المغلقة {len(closed)} | "
        f"صافي مفترض {net:+.2f}R\n"
        f"حالات OHLC الملتبسة {ambiguous} | "
        "كل هدف يغلق ثلث الحجم.\n"
        "الكلفة تقديرية والعينة لا تثبت تفوق الاستراتيجية. "
        "النتائج القديمة منفصلة.",
        identity,
    )


def market_waiting(st, key, res, fr, now):
    if res.get("tier") is not None:
        return False

    previous = st.setdefault(
        "waiting_status_ts", {}
    ).get(key)

    if (
        previous
        and now - stamp(previous) < pd.Timedelta(hours=6)
    ):
        return False

    observed = res.get("obs", "").strip()
    missing = list(dict.fromkeys(
        str(x) for x in res.get("miss", []) if x
    ))

    if missing:
        reason = " | ".join(missing)
    elif "ناقص:" in observed:
        reason = observed.split(
            "ناقص:", 1
        )[1].split("\n", 1)[0].strip()
    elif observed.startswith((
        "تحليل المناطق: بيانات ناقصة",
        "تحليل المناطق: تعذر",
    )):
        reason = observed.split("\n", 1)[0]
    else:
        reason = "ما تحقق إعداد دخول مؤهل حسب شروط المناطق الحالية"

    closed = stamp(fr["15m"]["end"].iloc[-1])
    price = float(fr["15m"]["c"].iloc[-1])
    name = "الذهب" if key == "GOLD" else "ناسداك"

    text = (
        f"🟡 {name} | مراقبة وانتظار إعداد\n"
        "الحالة: الفحص مستمر، ماكو دخول ورقي جديد\n"
        f"آخر إغلاق: {price:,.2f}\n"
        f"سبب الانتظار: {reason}\n\n"
        f"{observed}\n\n"
        f"المصدر: {fr['_source']}\n"
        f"وقت الشمعة UTC: {closed.isoformat()}\n"
        "هذا تحديث حالة؛ تنبيه الإعداد أو الدخول "
        "يوصلك برسالة منفصلة."
    )

    event(st, text, key + ":waiting:" + closed.isoformat())
    st["waiting_status_ts"][key] = now.isoformat()
    return True


def run(a, now=None):
    now = (
        pd.Timestamp.now(tz="UTC")
        if now is None else stamp(now)
    )
    st = a.load()

    if not isinstance(st.setdefault("outbox", []), list):
        raise RuntimeError("invalid outbox; state preserved")

    token = os.environ.get("TG_TOKEN")
    chat = os.environ.get("TG_CHAT")

    if not token or not chat:
        raise RuntimeError("TG_TOKEN and TG_CHAT required")

    if not flush(st, a.save, token, chat):
        return 1

    for key in ("GOLD", "NDX"):
        versions = st.setdefault("strategy_versions", {})

        if versions.get(key) != "simple_zones_v1":
            st.setdefault("last_analysis", {}).pop(key, None)
            versions[key] = "simple_zones_v1"

    data, summary = {}, []

    for key in a.ASSETS:
        try:
            data[key] = md.frames(key, now)
        except Exception as exc:
            reason = md.safe_error(exc)
            LOG.error("DATA asset=%s ERROR %s", key, reason)
            summary.append(f"{key}: {reason}")
            last = st.setdefault("health", {}).get(key)

            if (
                not last
                or now - stamp(last) > pd.Timedelta(hours=6)
            ):
                event(
                    st,
                    f"⚠️ بيانات {key} غير متوفرة: {reason}\n"
                    "لا إشارة دخول جديدة لهذا الأصل.",
                )
                st["health"][key] = now.isoformat()

    btc_dir = None

    if (
        "BTC" in data
        and md.freshness(data["BTC"], "BTC", now)[0]
    ):
        t4 = a.trend(data["BTC"]["4h"])
        t1 = a.trend(data["BTC"]["1h"])
        btc_dir = t4 if t4 == t1 else 0

    nw = news.get_news(
        st, now, os.environ.get("GEMINI_KEY", "")
    )

    for key, fr in data.items():
        fresh, reason = md.freshness(fr, key, now)
        last_close = stamp(fr["15m"]["end"].iloc[-1])
        sp = st["active"].get(key)

        if sp:
            manage(st, sp, fr, now, a)

            if sp["state"] in (
                "CLOSED", "INVALIDATED", "EXPIRED"
            ):
                del st["active"][key]

            summary.append(f"{key}: {sp['state']}")
            continue

        if not fresh:
            LOG.warning("SKIP asset=%s %s", key, reason)
            summary.append(f"{key}: {reason}")
            continue

        if (
            st["cool"].get(key)
            and now - stamp(st["cool"][key])
            < pd.Timedelta(hours=2)
        ):
            summary.append(f"{key}: cooldown")
            continue

        if (
            st.setdefault("last_analysis", {}).get(key)
            == last_close.isoformat()
        ):
            summary.append(
                st.get("last_decision", {}).get(
                    key, f"{key}: no new closed candle"
                )
            )
            continue

        try:
            res = a.analyze(key, fr, btc_dir)
            res = news.apply(res, nw)

            if key == "GOLD":
                LOG.info(
                    "GOLD_DETAILS observations=%s missing=%s",
                    res.get("obs", "").replace("\n", " | "),
                    res.get("miss", []),
                )

            st["last_analysis"][key] = last_close.isoformat()

            LOG.info(
                "ANALYSIS asset=%s tier=%s score=%s close=%s",
                key, res["tier"], res["score"], last_close,
            )

            if res["tier"] in ("A", "ARMED"):
                sp = make_setup(st, res, fr, now)
                status = decision_text(key, res, sp)
                st.setdefault("last_decision", {})[key] = status
                summary.append(status)

                if cost_r(sp) > 0.35:
                    LOG.info("%s", status)
                    continue

                st["active"][key] = sp
                st["watch"].pop(key, None)
                st["watch"].pop(key + ":E", None)

                event(
                    st,
                    setup_text(res, sp, fr, a),
                    sp["id"] + ":setup",
                )

            else:
                status = decision_text(key, res)
                st.setdefault("last_decision", {})[key] = status
                summary.append(status)

            if res["tier"] in ("B", "EARLY"):
                wkey = key + (
                    ":E" if res["tier"] == "EARLY" else ""
                )
                old = st["watch"].get(wkey)
                raw = (
                    f"{res['side']}:{res.get('style')}:"
                    f"{res.get('zone_p', 0):.2f}"
                )
                fingerprint = hashlib.sha256(
                    raw.encode()
                ).hexdigest()[:12]

                same_area = (
                    key in ("GOLD", "NDX")
                    and old is not None
                    and old.get("side") == res["side"]
                    and old.get("zone_p") is not None
                    and abs(
                        old["zone_p"] - res.get("zone_p", 0.0)
                    ) <= 0.5 * res.get("a1", 0.0)
                )

                if (
                    not old
                    or (
                        old.get("fingerprint") != fingerprint
                        and not same_area
                    )
                    or now - stamp(old["ts"])
                    >= pd.Timedelta(hours=6)
                ):
                    text = (
                        a.msg_early(res, a.ASSETS[key][2])
                        if res["tier"] == "EARLY"
                        else a.msg_watch(res, a.ASSETS[key][2])
                    )
                    text = text.replace(
                        "السعر الآن:", "آخر إغلاق:"
                    ).replace("السعر:", "آخر إغلاق:")

                    text += (
                        "\nالمصدر: " + fr["_source"]
                        + "\nوقت الإغلاق: "
                        + last_close.isoformat()
                    )

                    event(st, text)

                    st["watch"][wkey] = {
                        "side": res["side"],
                        "ts": now.isoformat(),
                        "fingerprint": fingerprint,
                        "zone_p": res.get("zone_p"),
                    }

                    if key in ("GOLD", "NDX"):
                        st.setdefault(
                            "waiting_status_ts", {}
                        )[key] = now.isoformat()

            elif res["score"] < 5.5:
                st["watch"].pop(key, None)

            if (
                key in ("GOLD", "NDX")
                and res.get("tier") is None
            ):
                market_waiting(st, key, res, fr, now)

        except Exception as exc:
            status = f"{key}: ERROR {md.safe_error(exc)}"
            st.setdefault("last_decision", {})[key] = status
            summary.append(status)

            LOG.error(
                "ANALYZE %s %s", key, md.safe_error(exc)
            )
            event(
                st,
                f"⚠️ خطأ تحليل {key}: {md.safe_error(exc)}؛ "
                "لم تُصدر إشارة جديدة.",
            )

    for text in news.alerts(nw, st, now):
        event(st, text)

    weekly(st, now)

    if os.environ.get("EVENT") == "workflow_dispatch":
        event(
            st,
            "ℹ️ فحص يدوي\n"
            + "\n".join(summary)
            + "\n" + nw["status"],
        )

    return 0 if flush(st, a.save, token, chat) else 1


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    a = importlib.import_module("agent")

    try:
        return run(a)
    except Exception as exc:
        LOG.error("Scanner failed: %s", md.safe_error(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())