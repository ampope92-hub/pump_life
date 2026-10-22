# Pump Seal Life and Service Intervals (Censored Survival in Rust)

A pump rebuilder needs two numbers for every pump in a batch it is quoting service contracts on: the expected mechanical-seal life, and a service interval set so that no more than 10% of similarly configured pumps fail before it. The agent gets a maintenance-system export of the existing fleet (`train.csv`), the quote batch (`test_features.csv`) and a data dictionary, and must fill in a Rust Cargo project (`smartcore` and `csv` pre-wired, offline, no new crates) so that `cargo run --release` reads both CSVs, fits a model, and writes `/app/predictions.csv` with `pump_id,life_days,service_days`.

## What makes it hard

- **Right-censoring with no flag.** `runtime_days` is time on the seal so far. For the 38% of pumps still in service it equals `observed_window_days` and is a lower bound, not a life. Fitting only the changed-out pumps fits the ones that died young (R2 0.2873 against a 0.36 bar); taking `runtime_days` at face value scores -0.2347. The fleet was commissioned in waves, with the longest-lived duty types and gentler fluids coming online last, so the bias bends the coefficients rather than shifting the intercept.
- **A leaky shortcut that wins every check the agent can run.** `cmms_expected_life_days` is reconciled against the work order once a seal is changed, so on failed training rows it is the realised life; on the quote batch it is a pre-condition-monitoring table lookup. Using it to fill censored rows plus OLS scores R2 0.308; shipping it directly scores 0.167. An internal holdout on training data scores the contaminated fit at R2 0.957 and the correct censored model near zero, because the only observable lives are the contaminated ones. `bearing_temp_rise_c` is a smaller version of the same leak (R2 -0.917 as a feature).
- **The interval is a quantile, and the spread is not constant.** Seal-life sigma rises roughly three-fold with vibration. A correct censored fit with one global sigma clears the R2 bar (0.4567) and lands 8.2% fleet-wide, but only 2.4% of the smoothest third fail before their quoted age against a 5.5% floor.
- Everything is built by hand: no Python data stack, no survival library, and Rust's standard library has no error function.

Bars were calibrated across 14 generator seeds: censoring-ignoring routes top out at R2 0.3211, the correct route floors at 0.3966; every constant-sigma route falls below 3.7% on the smooth band while the correct route holds 7.6-12.4% across bands.

## How it's graded

The verifier runs in a separate, offline container. It collects only the agent's crate sources and `/app/data`, rebuilds the crate from clean on a pre-compiled dependency skeleton, and runs it, so a hand-written predictions file never reaches the grader. Checks in `tests/test_outputs.py`:

- Shipped data unmodified (sha256); exact schema; one row per quoted pump; finite, plausible values with `service_days <= life_days`; neither column flat.
- `life_days` R2 >= 0.36 against withheld true lives.
- Share failing before `service_days` within [8%, 12%] fleet-wide and within [5.5%, 14.5%] in each vibration tercile (6000 scored pumps, 2000 per tercile).
- **Holdout re-run:** `/app/data` is swapped for a batch from a different generator seed (never present in the agent's container) and the same binary is run again: R2 > 0.30, coverage within [6%, 14%], and fewer than half of quotes byte-identical across batches. This defeats predictions hard-coded into the source.

Anti-cheat: agent code is built and run as an unprivileged user via `setpriv` (no supplementary groups, no-new-privs); the answer key lives root-only in the verifier image; `/tests` is closed to the build user and the run is scored 0 if it is still readable; the build's process group and all build-user processes are killed before pytest runs; pytest runs with `python -P` so a planted `pytest.py` cannot shadow it.

## Repository layout

- `instruction.md` - the task prompt given to the agent.
- `task.toml` - metadata, difficulty/solution/verification write-ups, timeouts and network policy.
- `environment/` - agent image: `Dockerfile`, the starter Cargo crate in `predictor/`, and the export plus data dictionary in `data/`.
- `solution/` - reference solution (`solve.sh`, `main.rs`: heteroscedastic log-normal AFT fitted by censored maximum likelihood), data generators (`generate_data.py`, `generate_holdout.py`) and `calibrate.py`, which reproduces the route table.
- `tests/` - verifier image (`Dockerfile`), `test.sh`, `test_outputs.py`, the compiled crate skeleton, golden labels and the holdout batch.

## Running it

Requires Docker and Harbor (`uv tool install harbor` or `pip install harbor`). From the repo root:

```
harbor run -p . -a oracle
```

runs the reference solution through the verifier (should score reward 1). To try an agent instead:

```
harbor run -p . -a <agent> -m <model>
```

for example `-a claude-code`.

---

Task design, data generators, reference solution and verifier by Audrey Pope.
