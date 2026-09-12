"""pump_life — route sweep and bar validation. Run it beside the task:

    python solution/calibrate.py            # 8 seeds
    python solution/calibrate.py 7          # just the shipped instance

It never runs inside the task container; it reads only generate_data.py, so a reviewer can
re-derive every number in task.toml rather than take it on trust. Needs numpy, pandas, scipy.

Does the provenance trap deceive?

Three things must hold at once, and the third is the point:

  1. the HONEST route still clears both bars (the ceiling must not move);
  2. every LEAK route fails at least one bar decisively;
  3. the honest validation an agent can actually build on train.csv RANKS THE LEAK FIRST.

For (3): the only rows in train.csv whose true life is observable are the ones whose seal
has already been changed. So the natural, careful validation is to hold out some of THOSE
rows and score against their runtime_days. That is exactly the set on which the maintenance
system reconciled cmms_expected_life_days against the work order -- so the holdout reports
the leak as near-perfect, and it is a rule-of-thumb table on the quote batch.
"""
import sys

import generate_data as G
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

TARGET_Q = 0.10
Z_Q = norm.ppf(TARGET_Q)
CATS = {"duty_cycle": ["continuous", "intermittent", "standby"],
        "fluid": ["water", "slurry", "solvent", "brine"]}


def design(df):
    """Model matrix: intercept, one-hot categoricals (drop first), centred numerics."""
    cols = [np.ones(len(df))]
    for c, levels in CATS.items():
        for lv in levels[1:]:
            cols.append((df[c].to_numpy() == lv).astype(float))
    for c, centre in [("inlet_bar", 4.2), ("rpm", 2900.0), ("temp_c", 52.0),
                      ("vibration_mm_s", 3.1), ("impeller_mm", 145.0)]:
        cols.append(df[c].to_numpy() - centre)
    return np.column_stack(cols)


def scale_design(df, hetero):
    """Scale model matrix: intercept alone, or intercept + vibration."""
    if not hetero:
        return np.ones((len(df), 1))
    return np.column_stack([np.ones(len(df)), df.vibration_mm_s.to_numpy() - 3.1])


def fit_censored(X, Z, y, censored):
    """Exact censored log-normal MLE. mu = X@beta, sigma = exp(Z@gamma)."""
    nb, ng = X.shape[1], Z.shape[1]

    def nll(p):
        beta, gamma = p[:nb], p[nb:]
        mu = X @ beta
        log_sd = np.clip(Z @ gamma, -3.0, 1.5)
        sd = np.exp(log_sd)
        z = (y - mu) / sd
        out = np.empty_like(z)
        # failed: density.  censored: survival above where the pump had got to.
        out[~censored] = -0.5 * z[~censored] ** 2 - log_sd[~censored]
        out[censored] = norm.logsf(z[censored])
        return -out.sum()

    # start from a complete-case OLS fit, which is biased but in the right region
    obs = ~censored
    beta0 = np.linalg.lstsq(X[obs], y[obs], rcond=None)[0]
    resid = y[obs] - X[obs] @ beta0
    p0 = np.concatenate([beta0, [np.log(resid.std())] + [0.0] * (ng - 1)])
    res = minimize(nll, p0, method="L-BFGS-B",
                   options={"maxiter": 4000, "maxfun": 8000})
    return res.x[:nb], res.x[nb:]


def predict(df, beta, gamma, hetero):
    mu = design(df) @ beta
    sd = np.exp(np.clip(scale_design(df, hetero) @ gamma, -3.0, 1.5))
    life = np.exp(mu + 0.5 * sd ** 2)          # E[life] for a log-normal
    service = np.exp(mu + sd * Z_Q)            # the TARGET_Q quantile
    return life, service


def r2(pred, truth):
    mean = truth.mean()
    return 1.0 - ((truth - pred) ** 2).sum() / ((truth - mean) ** 2).sum()


def coverage(service, truth):
    return float((truth < service).mean())


def tercile_masks(te):
    v = te.vibration_mm_s.to_numpy()
    lo, hi = np.quantile(v, [1 / 3, 2 / 3])
    return [("vib_low", v <= lo), ("vib_mid", (v > lo) & (v < hi)), ("vib_high", v >= hi)]



R2_BAR = 0.36
COV_LO, COV_HI = 0.08, 0.12
BAND_LO, BAND_HI = 0.055, 0.145


def _leak_design(df, extra):
    X = design(df)
    cols = [X]
    if "cmms" in extra:
        cols.append((np.log(df.cmms_expected_life_days.to_numpy()) - 6.5).reshape(-1, 1))
    if "btr" in extra:
        cols.append((df.bearing_temp_rise_c.to_numpy() - 12.0).reshape(-1, 1))
    return np.column_stack(cols)


def route_correct(tr, te):
    y = np.log(tr.runtime_days.to_numpy())
    cens = tr.still_running.to_numpy() == 1
    beta, gamma = fit_censored(design(tr), scale_design(tr, True), y, cens)
    return predict(te, beta, gamma, True)


def route_cmms_direct(tr, te):
    """Just ship the planner's number."""
    life = te.cmms_expected_life_days.to_numpy().astype(float)
    return life, life * 0.63          # the ratio the correct route averages


def route_cmms_fills_censoring(tr, te):
    """THE trap. Use cmms as the 'true life' for rows still running, so the censoring
    problem disappears and an ordinary least-squares fit is all that is left."""
    runtime = tr.runtime_days.to_numpy().astype(float)
    cmms = tr.cmms_expected_life_days.to_numpy().astype(float)
    cens = tr.still_running.to_numpy() == 1
    y = np.log(np.where(cens, cmms, runtime))
    X = design(tr)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    Zt = scale_design(tr, True)
    g = np.linalg.lstsq(Zt, np.log(np.abs(resid) + 1e-9) + 0.5723, rcond=None)[0]
    return predict(te, beta, g, True)


def _route_feature(tr, te, extra):
    y = np.log(tr.runtime_days.to_numpy())
    cens = tr.still_running.to_numpy() == 1
    Xtr, Xte = _leak_design(tr, extra), _leak_design(te, extra)
    beta, gamma = fit_censored(Xtr, scale_design(tr, True), y, cens)
    mu = Xte @ beta
    sd = np.exp(np.clip(scale_design(te, True) @ gamma, -3.0, 1.5))
    return np.exp(mu + 0.5 * sd ** 2), np.exp(mu + sd * Z_Q)


def route_cmms_feature(tr, te):  return _route_feature(tr, te, ("cmms",))
def route_btr_feature(tr, te):   return _route_feature(tr, te, ("btr",))
def route_both_feature(tr, te):  return _route_feature(tr, te, ("cmms", "btr"))


def route_complete_case(tr, te):
    """The classic mistake: drop the rows still running, fit on what failed."""
    obs = tr.still_running.to_numpy() == 0
    sub = tr[obs]
    y = np.log(sub.runtime_days.to_numpy())
    X = design(sub)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    Zt = scale_design(sub, True)
    g = np.linalg.lstsq(Zt, np.log(np.abs(resid) + 1e-9) + 0.5723, rcond=None)[0]
    return predict(te, beta, g, True)


ROUTES = {
    "1. honest censored AFT (reference)": route_correct,
    "2. leak: ship cmms as life": route_cmms_direct,
    "3. leak: cmms fills the censored rows": route_cmms_fills_censoring,
    "4. leak: cmms as a feature": route_cmms_feature,
    "5. leak: bearing_temp as a feature": route_btr_feature,
    "6. leak: both as features": route_both_feature,
    "7. complete-case (drop censored)": route_complete_case,
}


def holdout_rank(tr):
    """What an honest validation built on train.csv alone reports.

    Fit on half the rows; score on the OTHER half's already-changed seals, the only rows
    whose true life is observable. If the trap works this ranks the leak above the truth.
    """
    idx = np.arange(len(tr))
    a, b = tr.iloc[idx % 2 == 0].copy(), tr.iloc[idx % 2 == 1].copy()
    scored = b[b.still_running == 0]
    truth = scored.runtime_days.to_numpy().astype(float)
    out = {}
    for nm, extra in [("honest", ()), ("+cmms", ("cmms",)), ("+btr", ("btr",))]:
        y = np.log(a.runtime_days.to_numpy())
        cens = a.still_running.to_numpy() == 1
        Xa, Xs = _leak_design(a, extra), _leak_design(scored, extra)
        beta, gamma = fit_censored(Xa, scale_design(a, True), y, cens)
        sd = np.exp(np.clip(scale_design(scored, True) @ gamma, -3.0, 1.5))
        out[nm] = r2(np.exp(Xs @ beta + 0.5 * sd ** 2), truth)
    out["cmms direct"] = r2(scored.cmms_expected_life_days.to_numpy().astype(float), truth)
    return out


def run(seed):
    tr, te = G.make_pumps(seed)
    truth = te.true_life_days.to_numpy()
    masks = tercile_masks(te)
    rows = []
    for nm, fn in ROUTES.items():
        life, service = fn(tr, te)
        cov = coverage(service, truth)
        terc = [coverage(service[m], truth[m]) for _, m in masks]
        ok = (r2(life, truth) >= R2_BAR and COV_LO <= cov <= COV_HI
              and all(BAND_LO <= t <= BAND_HI for t in terc))
        rows.append({"seed": seed, "route": nm, "R2": r2(life, truth), "cov": cov * 100,
                     "t_lo": terc[0] * 100, "t_mid": terc[1] * 100, "t_hi": terc[2] * 100,
                     "PASS": ok})
    return rows, holdout_rank(tr)


if __name__ == "__main__":
    import pandas as pd
    seeds = [int(a) for a in sys.argv[1:]] or list(range(6))
    allrows, holds = [], []
    for s in seeds:
        r, h = run(s); allrows += r; holds.append(h)
    df = pd.DataFrame(allrows); pd.set_option("display.width", 220)
    print(f"=== QUOTE BATCH (graded) — bars: R2>={R2_BAR}, overall {COV_LO:.0%}-{COV_HI:.0%}, "
          f"terciles {BAND_LO:.1%}-{BAND_HI:.1%} ===")
    g = df.groupby("route").agg(R2=("R2","mean"), cov=("cov","mean"), t_lo=("t_lo","mean"),
                                t_mid=("t_mid","mean"), t_hi=("t_hi","mean"),
                                n_pass=("PASS","sum"), n=("PASS","size"))
    print(g.to_string(float_format=lambda v: f"{v:7.3f}"))
    print("\n=== TRAIN-ONLY HONEST HOLDOUT (fit on half, score on the other half's failed seals) ===")
    print(pd.DataFrame(holds).mean().to_string(float_format=lambda v: f"{v:.4f}"))
