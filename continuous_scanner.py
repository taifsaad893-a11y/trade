import os
import time
import traceback

import scanner


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
