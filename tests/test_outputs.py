"""Verifier for the pump seal-life model.

test.sh deletes any predictions.csv the agent left behind and rebuilds and runs the Cargo
project at /app/predictor, so the file graded here is always the output of a clean release
build.

Two graded quantities, because a single accuracy number does not say whether a fleet can be
scheduled off this model:

  life_days     expected seal life. Scored by R2 against the run-to-failure lives.
  service_days  the age by which at most a tenth of pumps in that configuration have
                failed. Scored by REALISED coverage — the share of pumps that actually
                failed before the age quoted for them — overall and within vibration
                bands.

The bands are the load-bearing part. Overall coverage on its own is a proxy a wrong answer
also satisfies: one number quoted for every pump, or any fixed fraction of a good life
prediction, can be tuned to land on 10% overall while being far too long for smooth pumps
and far too short for rough ones. Both were tried against this suite with the withheld
truth in hand — more than an agent can do — and both are rejected by the band check.

The withheld lives live at /opt/golden inside this image, never in the agent environment.
Only pytest is installed, so the CSV reading and the arithmetic come from the standard
library rather than pandas/scikit-learn.
"""

import csv
import math
import pathlib

import pytest

PRED_PATH = pathlib.Path("/app/predictions.csv")
# Read from the root-only copy baked into the verifier image, not the /tests mount: the
# agent's crate is rebuilt and run in this container, and only this path is closed to it.
LABELS_PATH = pathlib.Path("/opt/golden/test_labels.csv")
TEST_FEATURES = pathlib.Path("/app/data/test_features.csv")

R2_BAR = 0.36
TARGET_RATE = 0.10
# Tolerance around the target, stated in instruction.md so it is a rule the agent is given
# rather than one it has to guess. Calibration behind the numbers is in task.toml, not here.
COVERAGE_LO, COVERAGE_HI = 0.08, 0.12
BAND_LO, BAND_HI = 0.055, 0.145

# frozen at task-build time; the shipped export must not be altered
TEST_FEATURES_SHA = "aed6f3be5e4f98edf6bad88dcf5db3b9e22b9311184abe627e902bed6de0bc01"
TRAIN_SHA = "3f384ebf005c404cf498730ddee45716e8b844ab856722b3f2cbb7fd965fd1ab"


def _read_csv(path):
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        return header, [row for row in reader if row]


@pytest.fixture(scope="session")
def preds():
    assert PRED_PATH.exists(), (
        "/app/predictions.csv was not written — `cargo run --release` in "
        "/app/predictor must produce it"
    )
    return _read_csv(PRED_PATH)


@pytest.fixture(scope="session")
def labels():
    _, rows = _read_csv(LABELS_PATH)
    return {row[0]: float(row[1]) for row in rows}


@pytest.fixture(scope="session")
def vibration():
    """Vibration per quoted pump, read from the shipped export the agent was given."""
    header, rows = _read_csv(TEST_FEATURES)
    i_id = header.index("pump_id")
    i_vib = header.index("vibration_mm_s")
    return {row[i_id]: float(row[i_vib]) for row in rows}


@pytest.fixture(scope="session")
def quoted(preds):
    """predictions.csv as {pump_id: (life_days, service_days)}."""
    _, rows = preds
    return {row[0]: (float(row[1]), float(row[2])) for row in rows}


def _sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r2(pred, truth):
    mean = sum(truth) / len(truth)
    ss_res = sum((t - p) ** 2 for p, t in zip(pred, truth))
    ss_tot = sum((t - mean) ** 2 for t in truth)
    assert ss_tot > 0, "the run-to-failure lives have no variance"
    return 1.0 - ss_res / ss_tot


def _bands(vibration, ids):
    """Split the quoted pumps into vibration terciles, by the shipped feature values."""
    order = sorted(vibration[i] for i in ids)
    lo_cut = order[len(order) // 3]
    hi_cut = order[2 * len(order) // 3]
    out = {"low": [], "mid": [], "high": []}
    for i in ids:
        v = vibration[i]
        out["low" if v <= lo_cut else ("high" if v >= hi_cut else "mid")].append(i)
    return out


def test_shipped_data_unmodified():
    """The shipped train.csv and test_features.csv are byte-for-byte as issued."""
    assert _sha(TEST_FEATURES) == TEST_FEATURES_SHA, "test_features.csv was altered"
    assert _sha(pathlib.Path("/app/data/train.csv")) == TRAIN_SHA, "train.csv was altered"


def test_schema(preds):
    """predictions.csv carries exactly pump_id, life_days and service_days."""
    header, _ = preds
    assert header == ["pump_id", "life_days", "service_days"], (
        "predictions.csv must have exactly columns pump_id,life_days,service_days, "
        f"got {header}"
    )


def test_covers_every_pump_exactly_once(quoted, labels, preds):
    """Every quoted pump has exactly one row — no duplicates, omissions or unknown ids."""
    _, rows = preds
    assert len(rows) == len(labels), (
        f"expected one row per quoted pump ({len(labels)}), got {len(rows)}"
    )
    assert set(quoted) == set(labels), "predictions.csv pump_ids do not match the quote batch"


def test_values_valid(quoted):
    """Both columns are finite, positive, physically plausible days."""
    for pid, (life, service) in quoted.items():
        assert math.isfinite(life) and math.isfinite(service), f"{pid}: non-finite value"
        assert 0.0 < life < 100000.0, f"{pid}: implausible life_days {life}"
        assert 0.0 < service < 100000.0, f"{pid}: implausible service_days {service}"
        assert service <= life, (
            f"{pid}: service_days {service} exceeds life_days {life} — the age by which a "
            "tenth have failed cannot be later than the expected life"
        )


def test_predictions_not_degenerate(quoted):
    """Neither column is a constant — a single number quoted for every pump is not a model."""
    for col, idx in (("life_days", 0), ("service_days", 1)):
        vals = [v[idx] for v in quoted.values()]
        spread = (max(vals) - min(vals)) / (sum(vals) / len(vals))
        assert spread > 0.1, (
            f"{col} varies by only {spread:.3f} of its mean across the quote batch"
        )


def test_r2_bar(quoted, labels):
    """Expected-life predictions reach R2 0.36 against the run-to-failure lives."""
    ids = sorted(labels)
    r2 = _r2([quoted[i][0] for i in ids], [labels[i] for i in ids])
    assert r2 >= R2_BAR, f"life_days R2 {r2:.4f} is below the {R2_BAR} bar"


def test_service_interval_coverage(quoted, labels):
    """At most a tenth of pumps fail before the age quoted for them, fleet-wide."""
    ids = sorted(labels)
    failed_early = sum(1 for i in ids if labels[i] < quoted[i][1])
    rate = failed_early / len(ids)
    assert COVERAGE_LO <= rate <= COVERAGE_HI, (
        f"{rate:.3%} of pumps failed before their quoted service age, outside "
        f"[{COVERAGE_LO:.1%}, {COVERAGE_HI:.1%}] around the {TARGET_RATE:.0%} target"
    )


def test_service_interval_coverage_within_vibration_bands(quoted, labels, vibration):
    """The interval holds its failure rate for smooth and rough pumps alike, not just on average."""
    ids = sorted(labels)
    for band, members in _bands(vibration, ids).items():
        failed_early = sum(1 for i in members if labels[i] < quoted[i][1])
        rate = failed_early / len(members)
        assert BAND_LO <= rate <= BAND_HI, (
            f"{band}-vibration pumps: {rate:.3%} failed before their quoted service age, "
            f"outside [{BAND_LO:.1%}, {BAND_HI:.1%}] — the interval does not track how "
            "much seal life varies across the fleet"
        )
