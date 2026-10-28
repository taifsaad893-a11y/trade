"""Patch the reviewed agent without executing its old main loop."""
import ast
from pathlib import Path


def patch_source(source):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    changes = []

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    required = {
        "analyze_core", "analyze", "gold_vp", "zone_strat",
        "sr_multi", "atr", "trend", "load", "save",
        "trend_breakout_strat",
    }

    if not required <= functions.keys():
        raise RuntimeError(
            "agent.py differs from reviewed version: missing "
            + ", ".join(sorted(required - functions.keys()))
        )

    def edit(name, replacements):
        node = functions[name]
        old = "".join(lines[node.lineno - 1:node.end_lineno])
        new = old

        for before, after in replacements:
            if after in new:
                continue

            if new.count(before) != 1:
                raise RuntimeError(
                    f"{name}: expected exactly one reviewed block; "
                    "no file changed"
                )

            new = new.replace(before, after, 1)

        if new != old:
            changes.append((node.lineno - 1, node.end_lineno, new))

    edit("analyze_core", [
        (
            "rr = min(3.0, room / risk) if room > 0 else 3.0",
            "rr = min(3.0, room / risk) if room > 0 else 0.0",
        ),
        (
            'if style and pl and sc >= A_MIN and pl["rr"] >= 2:',
            'if style and pl and sc >= A_MIN and pl["rr"] >= 2 '
            'and not (brk and ext > 0.5):',
        ),
    ])

    # Trend is context, not an entry gate.  When 1H is range, the breakout
    # engine can still operate; direction is inferred from 4H when available,
    # otherwise from price relative to the 1H EMA20.  Confirmed zone bounces
    # and breaks remain independent entries through zone_strat.
    edit("trend_breakout_strat", [
        (
            '''    # -----------------------------------------------------\n    # 1H decides the trading direction.\n    # -----------------------------------------------------\n    t1 = trend(d1)\n    t4 = trend(d4)\n\n    if t1 == 0:\n        res["side"] = t4 if t4 != 0 else 1\n        res["obs"] = (\n            f"Trend Breakout: 1h {TR[t1]} | 4h {TR[t4]}\\n"\n            "الحالة: 1H رينج — لا دخول، ننتظر اتجاه أو كسر أوضح."\n        )\n        return res\n\n    s = t1\n    res["side"] = s\n''',
            '''    # -----------------------------------------------------\n    # Trend is quality/context, not a mandatory entry gate.\n    # -----------------------------------------------------\n    t1 = trend(d1)\n    t4 = trend(d4)\n\n    if t1 != 0:\n        s = t1\n    elif t4 != 0:\n        s = t4\n    else:\n        e20_1h = float(ema(d1["c"], 20).iloc[-1])\n        s = 1 if price >= e20_1h else -1\n\n    res["side"] = s\n''',
        ),
        (
            "    strong_breakout = fresh_break and strong_volume and strong_body",
            "    strong_breakout = fresh_break and strong_body and (strong_volume or vr is None)",
        ),
        (
            "    score = 3.0  # 1H trend is present and decides direction.",
            "    score = 3.0 if t1 == s else (1.5 if t1 == 0 else 0.5)",
        ),
    ])

    # Runtime strategy: breakout/retest and confirmed support/resistance
    # bounce/break are independent setup families. Trend/EMA/volume affect
    # quality, but a range reading by itself must not veto a valid setup.
    node = functions["analyze"]
    analyze_code = '''def analyze(key, fr, btc_dir):
    trend_res = trend_breakout_strat(key, fr, btc_dir)

    try:
        zone_res = zone_strat(key, fr)
    except Exception as exc:
        print(f"{key}: zone strategy error: {exc}")
        zone_res = None

    confirmed = []
    if trend_res.get("tier") in ("A", "ARMED"):
        confirmed.append(trend_res)
    if zone_res is not None and zone_res.get("tier") == "A":
        confirmed.append(zone_res)

    if confirmed:
        rank = {"A": 2, "ARMED": 1}
        res = max(
            confirmed,
            key=lambda r: (rank.get(r.get("tier"), 0), r.get("score", 0.0)),
        )
    elif trend_res.get("tier") == "B":
        res = trend_res
    else:
        res = trend_res
        if zone_res is not None:
            zobs = zone_res.get("obs")
            if zobs:
                res["obs"] = (res.get("obs") or "") + "\\n\\n" + zobs

    if SPOT_ONLY and res["side"] == -1:
        res["tier"] = None

    return res
'''
    current = "".join(lines[node.lineno - 1:node.end_lineno])
    if current != analyze_code:
        changes.append((node.lineno - 1, node.end_lineno, analyze_code))

    edit("gold_vp", [
        ("prof = vol_profile(d15.tail(300))", "prof = vol_profile(d15.iloc[:-24].tail(300))"),
        ("for i in range(n - 1, -1, -1):", "for i in range(n - 3, -1, -1):"),
        ('"rr": round(min(3.0, rr), 1)', '"rr": round(rr, 2)'),
    ])

    edit("zone_strat", [
        ('"rr": round(min(3.0, rr), 1)', '"rr": round(rr, 2)'),
        (
            '"منخفضة" if q >= 7 else ("متوسطة" if q >= 5 else "عالية")',
            '"غير مقاسة إحصائياً (جودة مرتفعة)" if q >= 7 else '
            '("غير مقاسة إحصائياً (جودة متوسطة)" if q >= 5 else '
            '"غير مقاسة إحصائياً (جودة محدودة)")',
        ),
    ])

    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "main"
        ):
            changes.append((
                node.lineno - 1,
                node.end_lineno,
                'if __name__ == "__main__":\n    main()\n',
            ))

    for start, end, value in sorted(changes, reverse=True):
        lines[start:end] = [value]

    output = "".join(lines).replace(
        'GEMINI_KEY = os.environ["GEMINI_KEY"]',
        'GEMINI_KEY = os.environ.get("GEMINI_KEY", "")',
    )

    ast.parse(output)
    return output


def main():
    path = Path("agent.py")
    original = path.read_text(encoding="utf-8")
    updated = patch_source(original)

    if updated != original:
        path.write_text(updated, encoding="utf-8")

    print(
        "Reviewed agent patches validated. "
        "Entry point: python scanner.py"
    )


if __name__ == "__main__":
    main()
