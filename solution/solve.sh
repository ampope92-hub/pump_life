#!/usr/bin/env bash
set -euo pipefail

# Reference solution for pump_life.
#
# The catch is what runtime_days means. For a pump with still_running = 1 the
# seal had NOT failed yet, so that number is a lower bound on its life, not the
# life. Nearly 38% of the export is in that state. Fit on the changed-out pumps
# only and you are fitting the pumps that died young, which lands around R2 0.28
# and predicts an average life well under what the quote batch actually managed.
#
# Worse, the install waves were not random: the long-lived duty types and the
# gentler fluids came online late, so they have short windows and are the most
# likely to still be running. Throwing the censored rows away therefore bends the
# coefficients, not just the intercept. It under-credits exactly the setups that
# last longest.
#
# So: log-normal accelerated failure time, fitted by EM over every row. Censored
# rows enter each pass at the expectation of a normal truncated at where they had
# got to, E[y | y > c] = mu + sd * phi(a)/(1 - Phi(a)), which pulls them up to
# where that pump would plausibly have ended rather than pinning them at the
# report date. Ten passes is plenty. Scores ~0.49 against a noise ceiling of 0.50.

# The model itself is solution/main.rs, next to this script — Rust is not a shell
# language and 130 lines of it inside a heredoc is neither reviewable nor compilable
# on its own. Drop it in and let cargo build it.
install -m 644 "$(dirname "$0")/main.rs" /app/predictor/src/main.rs

cd /app/predictor
cargo run --release --offline
