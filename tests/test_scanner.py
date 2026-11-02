import os
import unittest

import pandas as pd

from taifmarket_strategy import analyze_sequence


class FakeAgent:
    SPOT_ONLY = False

    def atr(self, frame):
        return pd.Series([1.0] * len(frame), index=frame.index)

    def trend(self, frame):
        return int(frame.attrs.get("trend_value", 0))


def frame(rows, freq="15min", trend=0):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, index=idx)
    df["v"] = df.get("v", 100.0)
    df.attrs["trend_value"] = trend
    return df


def base_frames(trend_1h=0, trend_4h=0):
    d1 = frame([{"o":100,"h":101,"l":99,"c":100,"v":100}] * 60, "1h", trend_1h)
    d4 = frame([{"o":100,"h":102,"l":98,"c":100,"v":100}] * 30, "4h", trend_4h)
    return d1, d4


def long_setup():
    rows = []
    for i in range(45):
        rows.append({"o":100.0,"h":100.8,"l":99.0 if i in (32, 38) else 99.4,"c":100.0,"v":100})
    rows.append({"o":99.7,"h":100.0,"l":98.6,"c":99.4,"v":140})
    rows.append({"o":99.4,"h":101.5,"l":99.3,"c":101.2,"v":180})
    rows.append({"o":101.1,"h":101.2,"l":99.5,"c":99.8,"v":110})
    rows.append({"o":99.8,"h":100.7,"l":99.6,"c":100.5,"v":130})
    return frame(rows)


def short_setup():
    rows = []
    for i in range(45):
        rows.append({"o":100.0,"h":101.0 if i in (32, 38) else 100.6,"l":99.2,"c":100.0,"v":100})
    rows.append({"o":100.3,"h":101.4,"l":100.0,"c":100.6,"v":140})
    rows.append({"o":100.6,"h":100.7,"l":98.5,"c":98.8,"v":180})
    rows.append({"o":98.9,"h":100.5,"l":98.8,"c":100.2,"v":110})
    rows.append({"o":100.2,"h":100.4,"l":99.3,"c":99.5,"v":130})
    return frame(rows)


class SweepMssIdmDemandSupplyTests(unittest.TestCase):
    def setUp(self):
        self.old_cost = os.environ.get("ROUNDTRIP_COST_BPS")
        os.environ["ROUNDTRIP_COST_BPS"] = "0"
        self.agent = FakeAgent()

    def tearDown(self):
        if self.old_cost is None:
            os.environ.pop("ROUNDTRIP_COST_BPS", None)
        else:
            os.environ["ROUNDTRIP_COST_BPS"] = self.old_cost

    def analyze(self, d15, t1=0, t4=0):
        d1, d4 = base_frames(t1, t4)
        return analyze_sequence(self.agent, {}, "BTC", {"15m": d15, "1h": d1, "4h": d4}, None)

    def assert_tradeable_contract(self, result):
        self.assertIn(result["tier"], ("A", "ARMED"))
        self.assertIn("plan", result)
        plan = result["plan"]
        self.assertEqual(len(plan["tps"]), 3)
        self.assertGreaterEqual(plan["rr"], 2.0)

    def test_long_sequence_detected(self):
        result = self.analyze(long_setup())
        self.assertEqual(result["side"], 1)
        self.assert_tradeable_contract(result)
        self.assertIn("Sweep", result["obs"])
        self.assertIn("MSS", result["obs"])
        self.assertIn("IDM", result["obs"])
        self.assertIn("Demand", result["obs"])

    def test_short_sequence_detected(self):
        result = self.analyze(short_setup())
        self.assertEqual(result["side"], -1)
        self.assert_tradeable_contract(result)
        self.assertIn("Sweep", result["obs"])
        self.assertIn("MSS", result["obs"])
        self.assertIn("Supply", result["obs"])

    def test_higher_timeframe_context_does_not_veto_valid_long(self):
        result = self.analyze(long_setup(), -1, -1)
        self.assertEqual(result["side"], 1)
        self.assert_tradeable_contract(result)
        self.assertIn("1H", result["obs"])

    def test_strategy_version(self):
        result = self.analyze(long_setup())
        self.assertEqual(result["strategy_version"], "sweep_mss_idm_demand_supply_v2")


if __name__ == "__main__":
    unittest.main()
