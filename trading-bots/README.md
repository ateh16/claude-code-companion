# Trading bots

Three tools that share one setup (`common/`):

| Tool | Run | Settings |
|---|---|---|
| AAPL 20/50-day moving-average crossover bot | `python -m ma_crossover.bot` | `ma_crossover/bot.conf` |
| Four-program ensemble bot | `python -m ensemble.run` | `ensemble/ensemble.conf` |
| Ensemble backtest (2004 → today) | `python -m ensemble.backtest` | `ensemble/ensemble.conf` `[backtest]` |
| Mid-cap growth momentum picker | `python -m stock_picker.picker` | `stock_picker/picker.conf` |

All commands run from this `trading-bots/` folder. Both bots accept
`--dry-run` (show the signal, place no order) and `--force` (ignore the
9:25 AM–2:00 PM ET run window).

## Install

```bash
cd trading-bots
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then paste your Alpaca PAPER keys
python -m pytest -q tests # 19 offline tests, no internet needed
```

## First things to run

```bash
python -m ensemble.backtest          # downloads ~25 years of prices, then prints the report
python -m ma_crossover.bot --dry-run --force
python -m ensemble.run --dry-run --force
python -m stock_picker.picker
```

Backtest output lands in `backtest_results/` (report, CSVs, equity-curve chart).
Picker output lands in `reports/`. Price cache lives in `data_cache/`. None of
those are committed.

## Running in the cloud (GitHub Actions)

`.github/workflows/ma_crossover.yml` and `ensemble.yml` (at the repo root) run
each bot on weekdays at 9:30 ET and commit the day's log to `trading-bots/logs/`.

1. In the repo on GitHub: **Settings → Secrets and variables → Actions** → add
   `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (paper keys).
2. Scheduled workflows only run from the repo's **default branch** — merge this
   branch into `main` before expecting daily runs.
3. Test once by hand: **Actions → MA crossover bot → Run workflow**.

The workflows hard-code `ALPACA_PAPER: "true"`. Live money requires changing
that on purpose.

Nothing here is investment advice. Paper-trade first.
