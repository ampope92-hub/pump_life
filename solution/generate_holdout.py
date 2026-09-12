"""Generates the hidden holdout batch the verifier re-runs the agent's program against.

Same generative process as the shipped instance (solution/generate_data.py), different
seed. The shipped instance is SEED 7; this must never be 7.

Why it exists: the graded quote batch is visible to the agent, so a program could fit once,
paste the resulting predictions into the Rust source as constants, and pass without ever
reading the CSVs at run time. The verifier swaps /app/data for this batch and runs the same
binary again. pump_ids are PM00000.. in both batches, so a program returning saved
predictions emits the right ids with values fitted to the wrong pumps, and both the R2 and
the coverage bars collapse. An honest program refits and scores normally.

Writes tests/holdout/{train.csv,test_features.csv,test_labels.csv}. The features go into the
verifier image at /opt/holdout and are copied into /app/data for the second run; the labels
stay root-only and are never in any container the agent can reach.

Usage: python solution/generate_holdout.py <seed>
"""
import hashlib
import pathlib
import sys

from generate_data import make_pumps

SHIPPED_SEED = 7
FEATURES = ["duty_cycle", "fluid", "inlet_bar", "rpm", "temp_c",
            "vibration_mm_s", "bearing_temp_rise_c", "impeller_mm",
            "cmms_expected_life_days"]

seed = int(sys.argv[1])
if seed == SHIPPED_SEED:
    sys.exit(f"seed {seed} is the shipped instance — the holdout must differ")

out = pathlib.Path(__file__).resolve().parent.parent / "tests" / "holdout"
out.mkdir(parents=True, exist_ok=True)
tr, te = make_pumps(seed)

tr[FEATURES + ["observed_window_days", "runtime_days"]].to_csv(out / "train.csv", index=False)
te[["pump_id"] + FEATURES].to_csv(out / "test_features.csv", index=False)
te[["pump_id", "true_life_days"]].rename(columns={"true_life_days": "life_days"}).to_csv(
    out / "test_labels.csv", index=False)

print(f"holdout seed {seed} -> {out}")
for f in ("train.csv", "test_features.csv", "test_labels.csv"):
    print(f"  {f:20s} sha256 {hashlib.sha256((out / f).read_bytes()).hexdigest()}")
