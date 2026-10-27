import os
import time
import traceback

import pandas as pd

import scanner
import scanner_data


# Twelve Data is rate-limited. The scanner still runs every minute, but a
# closed candle does not change until the next candle closes. Reuse the last
# successful GOLD frame until that point instead of requesting 15m/1h/4h/1d
# history every minute. If a refresh is rate-limited, keep the last good frame
# for that iteration and retry on the next scan.
_TWELVE_ORIGINAL = scanner_data.twelve
_TWELVE_CACHE = {}


def _cached_twelve(symbol, tf, now):
    cache_key = (symbol, tf)
    cached = _TWELVE_CACHE.get(cache_key)

    if cached is not None:
        frame = cached["frame"]
        last_close = frame["end"].iloc[-1]
        next_close = last_close + scanner_data.DUR[tf]

        # Give the provider a few seconds after candle close before refreshing.
        if now < next_close + pd.Timedelta(seconds=5):
            return frame.copy()

    try:
        frame = _TWELVE_ORIGINAL(symbol, tf, now)
        _TWELVE_CACHE[cache_key] = {
            "frame": frame.copy(),
            "saved_at": now,
        }
        return frame
    except Exception as exc:
        if cached is not None and "429" in str(exc):
            print(
                f"Twelve Data rate limit for {symbol} {tf}; "
                "using last good closed-candle cache and retrying next scan",
                flush=True,
            )
            return cached["frame"].copy()
        raise


scanner_data.twelve = _cached_twelve


def env_int(name, default, minimum=1):
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def main():
    scans = env_int("LOOP_SCANS", 15)
    interval = env_int("SCAN_INTERVAL_SECONDS", 60, minimum=10)
    failed_scans = 0

    for index in range(scans):
        started = time.monotonic()
        code = 1

        try:
            result = scanner.main()
            code = 0 if result is None else int(result)
        except Exception as exc:
            print(
                f"continuous scan {index + 1}/{scans} crashed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            code = 1

        if code != 0:
            failed_scans += 1
            print(
                f"continuous scan {index + 1}/{scans} returned {code}; "
                "state is preserved and the next minute will retry",
                flush=True,
            )
        else:
            print(
                f"continuous scan {index + 1}/{scans} completed",
                flush=True,
            )

        if index < scans - 1:
            elapsed = time.monotonic() - started
            time.sleep(max(1.0, interval - elapsed))

    if failed_scans == scans:
        raise SystemExit(
            "all continuous scanner iterations failed"
        )


if __name__ == "__main__":
    main()
