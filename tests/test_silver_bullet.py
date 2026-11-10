import unittest
from unittest.mock import patch

import pandas as pd

import silver_bullet_scanner as sb


def demo():
    # Nov 3, 2026 is NY winter time: 10:00 NY = 15:00 UTC.
    idx = pd.date_range("2026-11-03T12:35:00Z", periods=32, freq="5min")
    rows = [
        {"o": 99.5, "h": 105 if i == 17 else 100,
         "l": 98 if i == 18 else 99, "c": 99.5}
        for i in range(32)
    ]
    rows[29] = {"o": 99.5, "h": 100, "l": 99, "c": 99.8}
    rows[30] = {"o": 100.0, "h": 101.8, "l": 99.7, "c": 101.1}
    rows[31] = {"o": 101.2, "h": 102, "l": 101, "c": 101.6}
    return pd.DataFrame(rows, index=idx)


def state():
    return {"active": {}, "used": [], "closed": [], "outbox": [],
            "event_ids": [], "n": 0}


class SilverBulletTests(unittest.TestCase):
    def test_nearby_target_rejected_even_when_farther_liquidity_exists(self):
        # Previously a target less than $1 away was accepted despite a much
        # larger stop. A farther level must not hide the nearer obstacle.
        for key in ("GOLD", "NDX"):
            with self.subTest(key=key), patch.object(sb, "levels", return_value=([101.0, 110.0], [])):
                self.assertIsNone(sb.setup(key, demo(), "test",
                                          pd.Timestamp("2026-11-03T15:16:00Z")))

    def test_rr_threshold_applies_to_both_directions_and_markets(self):
        for key in ("GOLD", "NDX"):
            for side in (1, -1):
                d = demo()
                if side == -1:
                    original = d.copy()
                    for col, reflected in (("o", "o"), ("h", "l"), ("l", "h"), ("c", "c")):
                        d[col] = 200 - original[reflected]
                entry = 100.5 if side == 1 else 99.5
                tick = 0.01 if key == "GOLD" else 0.25
                stop = (float(d["l"].tail(8).min()) - 2 * tick if side == 1
                        else float(d["h"].tail(8).max()) + 2 * tick)
                risk = (entry - stop) * side
                for ratio in (1.49, 1.51):
                    target = entry + side * risk * ratio
                    pools = ([target], []) if side == 1 else ([], [target])
                    with self.subTest(key=key, side=side, ratio=ratio), patch.object(sb, "levels", return_value=pools):
                        plan = sb.setup(key, d, "test", pd.Timestamp("2026-11-03T15:16:00Z"))
                        if ratio < 1.5:
                            self.assertIsNone(plan)
                        else:
                            self.assertIsNotNone(plan)
                            self.assertEqual(plan["target"], round(target, 4))

    def test_ny_dst_and_final_bar_exclusion(self):
        self.assertEqual(sb.session_of("2026-11-03T15:00:00Z")["name"], "New York AM")
        self.assertEqual(sb.session_of("2026-06-05T14:00:00Z")["name"], "New York AM")
        self.assertIsNone(sb.session_of("2026-11-03T15:55:00Z"))
        self.assertIsNone(sb.session_of("2026-11-07T15:00:00Z"))

    def test_three_candle_gap_and_prior_liquidity(self):
        plan = sb.setup("GOLD", demo(), "Twelve:XAU/USD",
                        pd.Timestamp("2026-11-03T15:16:00Z"))
        self.assertIsNotNone(plan)
        self.assertEqual(plan["side"], 1)
        self.assertEqual(plan["entry"], 100.5)
        self.assertEqual(plan["target"], 105.0)
        self.assertEqual(plan["state"], "PENDING")
        self.assertEqual(plan["gap"], [100.0, 101.0])

    def test_paper_fill_target_and_stop_tracking(self):
        d = demo()
        plan = sb.setup("GOLD", d, "Twelve:XAU/USD",
                        pd.Timestamp("2026-11-03T15:16:00Z"))
        plan["id"] = "SB-GOLD-test"
        st = state()
        d.loc[pd.Timestamp("2026-11-03T15:15:00Z")] = {
            "o": 101.5, "h": 101.7, "l": 100.4, "c": 100.6,
        }
        sb.advance(st, plan, d, "Twelve:XAU/USD",
                   pd.Timestamp("2026-11-03T15:21:00Z"))
        self.assertEqual(plan["state"], "FILLED")
        d.loc[pd.Timestamp("2026-11-03T15:20:00Z")] = {
            "o": 100.6, "h": 105.5, "l": 100.5, "c": 105.0,
        }
        sb.advance(st, plan, d, "Twelve:XAU/USD",
                   pd.Timestamp("2026-11-03T15:26:00Z"))
        self.assertEqual(plan["exit"], "TARGET")
        self.assertEqual(len(st["closed"]), 1)
        self.assertEqual(len(st["outbox"]), 2)

    def test_source_switch_never_reprices_active_trade(self):
        plan = sb.setup("GOLD", demo(), "Twelve:XAU/USD",
                        pd.Timestamp("2026-11-03T15:16:00Z"))
        plan["id"] = "SB-GOLD-test2"
        st = state()
        sb.advance(st, plan, demo(), "Yahoo:GC=F (futures proxy)",
                   pd.Timestamp("2026-11-03T15:30:00Z"))
        self.assertEqual(plan["state"], "PENDING")
        sb.advance(st, plan, demo(), "Yahoo:GC=F (futures proxy)",
                   pd.Timestamp("2026-11-03T16:01:00Z"))
        self.assertEqual(plan["state"], "EXPIRED")

    def test_no_retroactive_gap_signal(self):
        self.assertIsNone(sb.setup("GOLD", demo(), "Twelve:XAU/USD",
                                   pd.Timestamp("2026-11-03T15:28:00Z")))


if __name__ == "__main__":
    unittest.main()
