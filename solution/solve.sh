#!/usr/bin/env bash
set -euo pipefail

# Reference solution for pump_life.
#
# Two things have to be right, and the same defect in the data damages both.
#
# First, what runtime_days means. No flag says so: a pump whose seal has been changed
# out has run for less time than it has been installed, while one still in service has
# run for its whole window, so runtime_days == observed_window_days marks a life that
# has not finished yet. 38% of the export is in that state. Fit the changed-out pumps
# alone and you are fitting the ones that died young — R2 0.29 against a 0.36 bar — and
# the install waves were not random, so it bends the coefficients rather than just the
# intercept: the long-lived duty types and gentler fluids came online late, carry short
# windows, and are the most likely to still be running.
#
# Second, the service interval. It is the age by which a tenth of the pumps in a given
# configuration have failed, i.e. the 0.10 quantile of that pump's own life distribution
# — so it depends on the spread, not just the level. And the spread is not constant:
# seal life on a rough pump is both shorter and less predictable, so sigma rises with
# vibration. Assume one global sigma and the interval is far too long on smooth pumps
# and too short on rough ones; the overall failure rate can still look close to 10%
# while both ends are wrong, which is why coverage is checked within vibration bands.
# Censoring makes the spread harder to recover than the level: the residuals still
# visible are the ones that failed early.
#
# So: log-normal AFT with log-sigma linear in vibration, fitted by exact censored
# maximum likelihood — failed pumps contribute their density, still-running pumps the
# probability of surviving past where they had got to — by gradient ascent with a
# backtracking step. Scores R2 0.479 with 9.2% realised coverage (7.6/10.2/9.8% across
# the vibration terciles) against bars of 0.36 and 5.5-14.5% per band.

# The model itself is solution/main.rs, next to this script — Rust is not a shell
# language and 130 lines of it inside a heredoc is neither reviewable nor compilable
# on its own. Drop it in and let cargo build it.
install -m 644 "$(dirname "$0")/main.rs" /app/predictor/src/main.rs

cd /app/predictor
cargo run --release --offline
