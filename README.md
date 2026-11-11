# Trade paper alerts

Active workflow: `.github/workflows/hourly.yml` runs `sr_retest_scanner.py`.
The external cron dispatch remains every five minutes. Entries use only completed
15-minute candles; target/stop monitoring uses completed five-minute candles.
Markets: GOLD and NASDAQ only, weekdays; no trade execution.

## Strategy

- Build support/resistance from confirmed two-sided pivots and recent 20-bar
  range boundaries on completed 1h and 4h candles from the same source.
- A 15m close must break a level by 0.08 ATR; a later candle within eight bars
  must revisit within 0.15 ATR and close back on the breakout side with a
  directional body. Reject an intervening close beyond the level by 0.10 ATR.
- Entry reference is the retest confirmation close, at most eight minutes old
  and no farther than 0.75 ATR from the level. Alerts identify this as a paper
  reference and instruct users not to chase. No live executable quote is implied.
- Stop is beyond the retest extreme plus 0.15 ATR, with minimum risk of
  0.8 ATR and 0.05% of entry. ATR is the last 14 completed 15m ranges before
  the confirmation candle.
- One fixed target: nearest opposing higher-timeframe level. Reject if below
  1.5R or one ATR away; never skip a nearby obstacle for a farther target.
- Hourly EMA50 alignment is informational and never blocks entry.
- No Silver Bullet hour restriction, no breakeven or partial targets.

Feed fallbacks retain explicit source labels. Active trades stay pinned to
their entry source. Completed 5m bars are aggregated only when every expected
bar is present; partial higher-timeframe candles are excluded.

Existing filled Silver Bullet trades continue under their original rules;
unfilled old orders expire on migration. State/history is preserved. A breakout
and level cannot generate duplicate entries. Alerts display Baghdad time.

Telegram has no blocking getMe/getChat preflight. Analysis and state saving
finish first. Notifications are retried up to three times and retained in a
persistent outbox on failure. Telegram does not offer an idempotent send API:
a lost response after successful delivery can still cause a repeated message.

Validation: `python -m unittest discover -s tests -v`.
These are strategy/operational tests, not evidence of profitability. The legacy
`backtest.py` has not been converted to this strategy.
