"""Save public candles for offline replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TFS = ("15m", "1h", "4h", "1d")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(folder, data, gc, start, end, failures):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": 1,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "assets": {},
        "load_errors": failures,
        "gc": None,
        "note": (
            "Frozen inputs only. Replay uses the current code "
            "and configured assumptions."
        ),
        "source_sha256": {},
    }

    for name in (
        "agent.py",
        "scanner.py",
        "scanner_data.py",
        "gold_volume.py",
        "simple_zones.py",
        "backtest.py",
        "upgrade_agent.py",
        "candle_archive.py",
    ):
        path = Path(name)

        if path.is_file():
            manifest["source_sha256"][name] = digest(path)

    def write(frame, name):
        path = folder / name

        frame.to_csv(
            path,
            index_label="time",
            compression="gzip",
            float_format="%.17g",
        )

        return {
            "file": name,
            "sha256": digest(path),
        }

    for key, frames in data.items():
        if key not in ("BTC", "ETH", "SOL", "GOLD", "NDX"):
            raise ValueError("unknown asset")

        manifest["assets"][key] = {
            "metadata": {
                name: value
                for name, value in frames.items()
                if name.startswith("_")
                and isinstance(value, (str, int, float, bool))
            },
            "frames": {
                tf: write(frames[tf], f"{key}_{tf}.csv.gz")
                for tf in TFS
                if tf in frames
            },
        }

    if gc is not None:
        manifest["gc"] = write(gc, "GC_15m.csv.gz")

    (folder / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def load(folder):
    folder = Path(folder).resolve()

    manifest = json.loads(
        (folder / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if manifest.get("version") != 1:
        raise ValueError(
            "unsupported candle archive version"
        )

    start = pd.Timestamp(manifest["start"])
    end = pd.Timestamp(manifest["end"])

    if (
        start.tzinfo is None
        or end.tzinfo is None
        or start >= end
    ):
        raise ValueError("invalid archive period")

    def read(info):
        path = (folder / info["file"]).resolve()

        if (
            path.parent != folder
            or digest(path) != info["sha256"]
        ):
            raise ValueError(
                "archive path or checksum mismatch"
            )

        frame = pd.read_csv(
            path,
            index_col="time",
            float_precision="round_trip",
        )
        frame.index = pd.to_datetime(
            frame.index, utc=True
        )
        frame["end"] = pd.to_datetime(
            frame["end"], utc=True
        )

        columns = ["o", "h", "l", "c", "v"]

        invalid = (
            frame.empty
            or frame.index.has_duplicates
            or not frame.index.is_monotonic_increasing
            or frame.index.isna().any()
            or frame["end"].isna().any()
            or not np.isfinite(
                frame[columns].to_numpy()
            ).all()
            or (frame["end"] <= frame.index).any()
            or (
                frame[["o", "h", "l", "c"]] <= 0
            ).any().any()
            or (frame["v"] < 0).any()
            or (
                frame["h"]
                < frame[["o", "l", "c"]].max(axis=1)
            ).any()
            or (
                frame["l"]
                > frame[["o", "h", "c"]].min(axis=1)
            ).any()
        )

        if invalid:
            raise ValueError("invalid archived candles")

        if "tb" in frame and (
            not np.isfinite(frame["tb"]).all()
            or (frame["tb"] < 0).any()
            or (frame["tb"] > frame["v"]).any()
        ):
            raise ValueError(
                "invalid archived taker volume"
            )

        return frame

    data = {}

    for key, entry in manifest["assets"].items():
        if key not in ("BTC", "ETH", "SOL", "GOLD", "NDX"):
            raise ValueError("unknown archived asset")

        frames = entry["frames"]

        if (
            not all(
                tf in frames
                for tf in ("15m", "1h", "4h")
            )
            or any(tf not in TFS for tf in frames)
        ):
            raise ValueError(
                "invalid archive timeframes"
            )

        data[key] = {
            tf: read(info)
            for tf, info in frames.items()
        }

        data[key].update({
            name: value
            for name, value in entry["metadata"].items()
            if name in (
                "_source", "_provider", "_symbol"
            )
        })

        if not all(
            name in data[key]
            for name in (
                "_source", "_provider", "_symbol"
            )
        ):
            raise ValueError(
                "missing archived source metadata"
            )

    gc = read(manifest["gc"]) if manifest.get("gc") else None

    return (
        data,
        gc,
        start,
        end,
        manifest.get("load_errors", {}),
    )