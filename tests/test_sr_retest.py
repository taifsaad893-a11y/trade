import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import pandas as pd
import requests

import sr_retest_scanner as sr


class RetestTests(unittest.TestCase):
    def fixture(self, side=1):
        idx = pd.date_range("2026-11-09", periods=144, freq="5min", tz="UTC")
        d = pd.DataFrame({"o": 99., "h": 99.5, "l": 98.5, "c": 99.}, index=idx)
        d.iloc[-6:-3] = [99.8, 101.5, 99.7, 101.]
        d.iloc[-3:] = [100.1, 100.8, 99.9, 100.5]
        if side == -1:
            d = pd.DataFrame({"o": 200-d.o, "h": 200-d.l,
                              "l": 200-d.h, "c": 200-d.c}, index=idx)
        return d, idx[-1]+sr.feed.FIVE

    def test_confirmed_retest_both_sides_one_target_and_min_rr(self):
        for side in (1, -1):
            d, now = self.fixture(side)
            with patch.object(sr, "levels", return_value=[94, 100, 106]):
                rec = sr.setup("GOLD", d, "test", now)
            self.assertIsNotNone(rec)
            self.assertEqual(rec["side"], side)
            self.assertGreaterEqual(rec["rr"], 1.5)
            self.assertEqual(rec["state"], "FILLED")
            self.assertGreater((rec["target"]-rec["entry"])*side, 0)

    def test_nearby_opposing_level_blocks_farther_target(self):
        d, now = self.fixture()
        with patch.object(sr, "levels", return_value=[100, 101, 106]):
            self.assertIsNone(sr.setup("GOLD", d, "test", now))

    def test_no_retest_or_stale_confirmation_no_entry(self):
        d, now = self.fixture()
        with patch.object(sr, "levels", return_value=[100, 106]):
            self.assertIsNone(sr.setup("GOLD", d, "test", now+pd.Timedelta(minutes=9)))
            d.iloc[-3:] = [101, 102, 100.8, 101.5]
            self.assertIsNone(sr.setup("GOLD", d, "test", now))

    def test_partial_htf_candle_excluded(self):
        d, now = self.fixture()
        self.assertEqual(len(sr.aggregate(d.iloc[:-1], 240, now)), 2)
        h = sr.aggregate(d, 240, now-pd.Timedelta(minutes=5))
        self.assertTrue((h.index+pd.Timedelta(hours=4) <= now-pd.Timedelta(minutes=5)).all())

    def test_only_post_entry_bars_and_stop_first(self):
        d, now = self.fixture()
        with patch.object(sr, "levels", return_value=[94, 100, 106]):
            rec = sr.setup("GOLD", d, "test", now)
        rec["id"] = "SR-test"
        sb = {"closed": [], "outbox": [], "event_ids": []}
        sr.advance(sb, rec, d, "test", now)
        self.assertEqual(rec["state"], "FILLED")
        future = pd.DataFrame({"o": [rec["entry"]], "h": [107], "l": [90],
                               "c": [100]}, index=[now])
        sr.advance(sb, rec, future, "different-source", now)
        self.assertEqual(rec["state"], "FILLED")
        sr.advance(sb, rec, future, "test", now)
        self.assertEqual(rec["r"], -1)
        self.assertEqual(len(sb["closed"]), 1)

    def test_telegram_retry_and_queue_persistence(self):
        for success in (True, False):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"state.json"
                sb = {"outbox": [{"id": "x", "text": "entry"}]}
                st = {"sr_retest": sb}
                response = Mock(ok=True)
                response.json.return_value = {"ok": True}
                attempts = ([requests.ConnectionError(), response] if success
                            else [requests.ConnectionError()]*3)
                with patch.object(sr.feed, "STATE", path), patch.dict(sr.os.environ, {"TG_TOKEN": "test", "TG_CHAT": "test"}), patch.object(sr.requests, "post", side_effect=attempts), patch.object(sr.time, "sleep"):
                    self.assertEqual(sr.flush(st, sb), success)
                self.assertEqual(len(sb["outbox"]), 0 if success else 1)
                self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
