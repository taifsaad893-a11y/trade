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
            changes.append(
                (node.lineno - 1, node.end_lineno, new)
            )

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

    node = functions["analyze"]

    analyze_code = '''def analyze(key, fr, btc_dir):
    if key in ("GOLD", "NDX"):
        from simple_zones import analyze_zones
        res = analyze_zones(key, fr, globals())
    elif key in ZONE_ASSETS and ZONE_MODE:
        res = zone_strat(key, fr)
    elif key == "GOLD" and GOLD_MTF:
        res = gold_best(fr, btc_dir)
    else:
        res = analyze_core(key, fr, btc_dir)
        res["tf"] = "15m"
    if SPOT_ONLY and res["side"] == -1:
        res["tier"] = None
    return res
'''

    current = "".join(
        lines[node.lineno - 1:node.end_lineno]
    )

    if current != analyze_code:
        changes.append(
            (node.lineno - 1, node.end_lineno, analyze_code)
        )

    edit("gold_vp", [
        (
            "prof = vol_profile(d15.tail(300))",
            "prof = vol_profile(d15.iloc[:-24].tail(300))",
        ),
        (
            "for i in range(n - 1, -1, -1):",
            "for i in range(n - 3, -1, -1):",
        ),
        (
            '"rr": round(min(3.0, rr), 1)',
            '"rr": round(rr, 2)',
        ),
    ])

    edit("zone_strat", [
        (
            '"rr": round(min(3.0, rr), 1)',
            '"rr": round(rr, 2)',
        ),
        (
            '"منخفضة" if q >= 7 else '
            '("متوسطة" if q >= 5 else "عالية")',
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