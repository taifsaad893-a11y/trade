import os
import unittest

import pandas as pd

from taifmarket_strategy import analyze_sequence


class FakeAgent:
    SPOT_ONLY = False

    def atr(self, frame):
        return pd.Series([1.0] * len(frame), index=frame.index)

    def ema(self, series, period):
        # Deliberately make EMA context unfavorable to prove it is score-only.
        if period == 20:
            value = float(series.iloc[-1]) + 1.0
        else:
            value = float(series.iloc[-1]) + 2.0
        return pd.Series([value] * len(series), index=series.index)

    def trend(self, frame):
        return int(frame.attrs.get("trend_value", 0))

    def sr_multi(self, fr, tolerance):
        return []


def frame(rows, freq="15min", trend=0):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, index=idx)
    df["v"] = df.get("v", 0.0)
    df.attrs["trend_value"] = trend
    return df


def base_frames(trend_1h=0, trend_4h=0):
    d1_rows = [{"o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0, "v": 100.0}] * 60
    d4_rows = [{"o": 100.0, "h": 102.0, "l": 98.0, "c": 100.0, "v": 100.0}] * 30
    return (
        frame(d1_rows, "1h", trend_1h),
        frame(d4_rows, "4h", trend_4h),
    )


class TaifmarketStrategyTests(unittest.TestCase):
    def setUp(self):
        self.old_cost = os.environ.get("ROUNDTRIP_COST_BPS")
        os.environ["ROUNDTRIP_COST_BPS"] = "0"
        self.agent = FakeAgent()

    def tearDown(self):
        if self.old_cost is None:
            os.environ.pop("ROUNDTRIP_COST_BPS", None)
        else:
            os.environ["ROUNDTRIP_COST_BPS"] = self.old_cost

    def test_one_hour_range_does_not_block_clear_breakout(self):
        rows = [{"o": 99.5, "h": 100.0, "l": 99.0, "c": 99.6, "v": 100.0}] * 58
        rows.append({"o": 99.6, "h": 100.0, "l": 99.1, "c": 99.8, "v": 100.0})
        rows.append({"o": 99.8, "h": 100.9, "l": 99.7, "c": 100.8, "v": 10.0})
        d15 = frame(rows)
        d1, d4 = base_frames(0, 0)

        result = analyze_sequence(
            self.agent,
            {},
            "BTC",
            {"15m": d15, "1h": d1, "4h": d4},
            None,
        )

        self.assertEqual(result["tier"], "A")
        self.assertEqual(result["side"], 1)
        self.assertEqual(result["style"], "ClearBreakout")
        self.assertNotIn("لا دخول حالياً", result["obs"])

    def test_higher_timeframes_ema_and_volume_are_score_only(self):
        rows = [{"o": 99.5, "h": 100.0, "l": 99.0, "c": 99.6, "v": 100.0}] * 58
        rows.append({"o": 99.6, "h": 100.0, "l": 99.1, "c": 99.8, "v": 100.0})
        rows.append({"o": 99.8, "h": 100.9, "l": 99.7, "c": 100.8, "v": 1.0})
        d15 = frame(rows)
        d1, d4 = base_frames(-1, -1)

        result = analyze_sequence(
            self.agent,
            {},
            "BTC",
            {"15m": d15, "1h": d1, "4h": d4},
            None,
        )

        self.assertEqual(result["tier"], "A")
        self.assertEqual(result["side"], 1)
        self.assertIn("عوامل جودة وScore فقط", result["obs"])

    def test_support_bounce_can_enter_while_one_hour_is_range(self):
        rows = [{"o": 99.5, "h": 100.0, "l": 99.0, "c": 99.5, "v": 100.0}] * 59
        rows.append({"o": 99.1, "h": 99.8, "l": 98.95, "c": 99.7, "v": 50.0})
        d15 = frame(rows)
        d1, d4 = base_frames(0, 0)

        result = analyze_sequence(
            self.agent,
            {},
            "BTC",
            {"15m": d15, "1h": d1, "4h": d4},
            None,
        )

        self.assertEqual(result["tier"], "A")
        self.assertEqual(result["side"], 1)
        self.assertEqual(result["style"], "SupportResistanceBounce")


if __name__ == "__main__":
    unittest.main()
