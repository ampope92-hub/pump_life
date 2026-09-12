"""Regenerates the shipped pump_life CSVs, byte for byte.

Run it beside the task (`python solution/generate_data.py`) and compare the printed sha256s
against the ones pinned in tests/test_outputs.py — that is the audit trail for the data.
Needs numpy and pandas; it never runs inside the task container, and it lives in solution/,
which is not part of the agent image, so shipping it cannot leak anything.

Pump seal life generator — the censoring trap plus a provenance trap.

v2 was measured saturated on 2026-08-29. Its "PASS CORE at 75%" rested entirely on one
opus run that the SDK killed after 71 seconds and the platform scored as a genuine failure;
counting only sessions that actually ran, every trial passed (3/3). Same story on 08-25 and
08-28. The censored-AFT axis is something both frontier models simply do correctly.

v3 keeps every bit of v2's physics — the censoring, the heteroscedastic scale, the bars, the
withheld labels are all unchanged — and adds the lever that worked for battery_fade v3 and,
before it, for the one accepted task (fleet_failures): make the agent's own honest validation
point at the WRONG answer.

  cmms_expected_life_days
    train: the maintenance system's expected-life figure. When a seal is changed the system
      RECONCILES this field against the work order, so on rows whose seal has already been
      changed it is the realised life to within a few days. On rows still running it is the
      planner's forward estimate off a duty/fluid rule-of-thumb table.
    quote batch: nobody has changed a seal yet, so every row carries the rule-of-thumb
      estimate — which knows nothing about vibration, temperature or speed, and nothing about
      the pump-to-pump scatter.

    This is the dangerous one, because it appears to DISSOLVE the task's central problem. The
    intended crux is that runtime_days is right-censored and needs a censored likelihood. An
    agent that finds this column sees a ready-made "true life" on the failed rows and a
    plausible number on the running ones -- so the censoring looks like a data-quality wrinkle
    it has already routed around, and there is no reason left to write the survival fit.
    In the quote batch that column is a rule-of-thumb table, and the fit collapses.

  bearing_temp_rise_c
    train: condition monitoring's last reading before the seal came out, so on failed rows it
      is a near-failure symptom -- bearing temperature climbs as a seal degrades.
    quote batch: a routine survey reading taken mid-life on a pump that is running normally.
    So in the maintenance history it looks like a strong early-warning feature, and on the
    quote batch it is close to noise.

The honest route -- ignore both, fit the censored AFT on runtime_days/observed_window_days --
is completely unchanged, and so is its score. This adds no ceiling; it adds ways to fall off
one. Both columns are drawn AFTER every other quantity, so a given seed's physics, censoring
and withheld labels are bit-for-bit what v2 shipped.
"""
import numpy as np
import pandas as pd

DUTY = ["continuous", "intermittent", "standby"]
DUTY_K = {"continuous": -0.45, "intermittent": 0.0, "standby": 0.38}
FLUID = ["water", "slurry", "solvent", "brine"]
FLUID_K = {"water": 0.22, "slurry": -0.40, "solvent": -0.10, "brine": -0.25}

# Spread grows with vibration. The clip keeps both ends physical: even a smooth pump
# scatters, and a rough one is not infinitely unpredictable.
# The BASE level is held at v1's 0.45: dropping it to 0.30 raised the R2 ceiling and
# collapsed the gap between the correct fit and a censoring-ignoring one to ~0.05, which
# is not a separable bar. Only the vibration SLOPE is new.
SIGMA_0 = 0.45
SIGMA_V = 0.13
SIGMA_LO = 0.28
SIGMA_HI = 0.90

SHIFT = 1.0

# The CMMS planner's expected-life table. Duty and fluid only -- it predates condition
# monitoring, so it cannot see vibration, temperature or speed, and it has no notion of
# pump-to-pump scatter. This is what the quote batch carries for every row, and what the
# still-running training rows carry.
PLAN_BASE = 690.0
PLAN_DUTY = {"continuous": -190.0, "intermittent": 0.0, "standby": 165.0}
PLAN_FLUID = {"water": 95.0, "slurry": -175.0, "solvent": -40.0, "brine": -105.0}
PLAN_SD = 0.16          # planner noise, log scale
RECONCILE_SD = 0.03     # work-order date lags the actual change-out by a few days

# Bearing temperature rise. Baseline is set by duty and speed; on a seal that is about to go
# it climbs with how far through its life the pump has got.
BTR_BASE = 11.0
BTR_RPM = 0.0022
BTR_WEAR = 9.5          # extra degrees at end of life (train, failed rows)
BTR_SD = 1.6

# The service interval is quoted so that at most this fraction have failed by then.
TARGET_Q = 0.10


def sigma_of(vibration):
    return np.clip(SIGMA_0 + SIGMA_V * (vibration - 3.1), SIGMA_LO, SIGMA_HI)


def make_pumps(seed=0, n_train=5000, n_test=6000, horizon=1400.0):
    rng = np.random.default_rng(seed)

    def block(n, with_window):
        duty = rng.choice(DUTY, n, p=[0.5, 0.35, 0.15])
        fluid = rng.choice(FLUID, n, p=[0.4, 0.25, 0.2, 0.15])
        inlet_bar = rng.normal(4.2, 1.1, n).clip(1.0, 9.0)
        rpm = rng.normal(2900, 420, n).clip(1200, 4200)
        temp_c = rng.normal(52, 14, n).clip(10, 110)
        vibration_mm_s = np.abs(rng.normal(3.1, 1.4, n)).clip(0.2, 12)
        impeller_mm = rng.choice([120, 145, 170, 195], n)
        df = pd.DataFrame({
            "duty_cycle": duty, "fluid": fluid,
            "inlet_bar": inlet_bar.round(2), "rpm": rpm.round(0),
            "temp_c": temp_c.round(1), "vibration_mm_s": vibration_mm_s.round(2),
            "impeller_mm": impeller_mm,
        })
        mu = (
            6.55
            + df.duty_cycle.map(DUTY_K).to_numpy()
            + df.fluid.map(FLUID_K).to_numpy()
            - 0.085 * (df.inlet_bar.to_numpy() - 4.2)
            - 0.00042 * (df.rpm.to_numpy() - 2900)
            - 0.0125 * (df.temp_c.to_numpy() - 52)
            - 0.115 * (df.vibration_mm_s.to_numpy() - 3.1)
            + 0.0016 * (df.impeller_mm.to_numpy() - 145)
        )
        sd = sigma_of(df.vibration_mm_s.to_numpy())
        life = np.exp(mu + rng.normal(0, 1.0, n) * sd)
        df["true_life_days"] = life.round(1)
        # What a correct service interval is: the TARGET_Q quantile of THIS pump's own
        # life distribution. Kept for calibration only — never shipped.
        df["true_service_days"] = np.exp(
            mu + sd * -1.2815515655446004  # Phi^-1(0.10)
        ).round(1)
        if with_window:
            shift = SHIFT * (
                df.duty_cycle.map({"continuous": 330.0, "intermittent": 0.0,
                                   "standby": -330.0}).to_numpy()
                + df.fluid.map({"water": -220.0, "slurry": 240.0,
                                "solvent": 60.0, "brine": 140.0}).to_numpy())
            df["window_days"] = (rng.uniform(60, horizon, n) + shift).clip(45, horizon).round(0)
        return df

    tr = block(n_train, True)
    te = block(n_test, False)

    failed = tr.true_life_days <= tr.window_days
    tr["still_running"] = (~failed).astype(int)
    tr["runtime_days"] = np.where(failed, tr.true_life_days, tr.window_days).round(1)
    tr = tr.rename(columns={"window_days": "observed_window_days"})
    tr = tr.drop(columns=["true_life_days", "true_service_days"])

    te.insert(0, "pump_id", [f"PM{i:05d}" for i in range(len(te))])

    # ---- the two provenance columns ---------------------------------------------------
    # Drawn last, so every quantity above -- features, lives, censoring, the withheld
    # labels -- is bit-for-bit identical to v2 for the same seed.
    def plan(df):
        """The planner's rule-of-thumb expected life. Duty and fluid only."""
        return (PLAN_BASE
                + df.duty_cycle.map(PLAN_DUTY).to_numpy()
                + df.fluid.map(PLAN_FLUID).to_numpy())

    # train: reconciled to the work order where the seal has been changed, planner estimate
    # where it has not.
    n = len(tr)
    plan_tr = plan(tr) * np.exp(rng.normal(0, PLAN_SD, n))
    recon = tr.runtime_days.to_numpy() * np.exp(rng.normal(0, RECONCILE_SD, n))
    changed = tr.still_running.to_numpy() == 0
    tr["cmms_expected_life_days"] = np.where(changed, recon, plan_tr).clip(30, None).round(1)

    # quote batch: nothing has been changed out, so it is the planner figure throughout.
    m = len(te)
    te["cmms_expected_life_days"] = (
        plan(te) * np.exp(rng.normal(0, PLAN_SD, m))).clip(30, None).round(1)

    # bearing temperature rise: a near-failure symptom on the rows that failed, a routine
    # mid-life survey reading on the quote batch.
    base_tr = BTR_BASE + BTR_RPM * (tr.rpm.to_numpy() - 2900)
    tr["bearing_temp_rise_c"] = (
        base_tr + np.where(changed, BTR_WEAR, 0.0)
        + rng.normal(0, BTR_SD, n)).clip(2.0, None).round(1)
    base_te = BTR_BASE + BTR_RPM * (te.rpm.to_numpy() - 2900)
    te["bearing_temp_rise_c"] = (base_te + rng.normal(0, BTR_SD, m)).clip(2.0, None).round(1)

    return tr, te

if __name__ == "__main__":
    import hashlib
    import pathlib as _pathlib

    # The shipped instance. The two provenance columns are drawn AFTER every other quantity,
    # so this seed's features, lives and censoring are bit for bit what the previous version
    # shipped and tests/golden/test_labels.csv is unchanged.
    SEED = 7
    FEATURES = ["duty_cycle", "fluid", "inlet_bar", "rpm", "temp_c",
                "vibration_mm_s", "bearing_temp_rise_c", "impeller_mm",
                "cmms_expected_life_days"]

    out = _pathlib.Path("regenerated")
    (out / "golden").mkdir(parents=True, exist_ok=True)
    tr, te = make_pumps(SEED)

    tr[FEATURES + ["observed_window_days", "runtime_days"]].to_csv(
        out / "train.csv", index=False)
    te[["pump_id"] + FEATURES].to_csv(out / "test_features.csv", index=False)
    te[["pump_id", "true_life_days"]].rename(
        columns={"true_life_days": "life_days"}).to_csv(
        out / "golden" / "test_labels.csv", index=False)

    for f in ("train.csv", "test_features.csv", "golden/test_labels.csv"):
        print(f"{f:24s} sha256 {hashlib.sha256((out / f).read_bytes()).hexdigest()}")
    print()
    cens = (tr.runtime_days == tr.observed_window_days)
    print(f"train {len(tr)} pumps, {cens.mean():.1%} still running (censored)")
    print(f"quote batch {len(te)} pumps, mean true life {te.true_life_days.mean():.1f} days")
