"""Verifier for the pump seal-life model.

test.sh deletes any predictions.csv the agent left behind and rebuilds and runs the
Cargo project at /app/predictor, so the file graded here is always the output of a
clean release build. We check that the file exists, its schema, that it covers every
quoted pump exactly once, that the values are numeric and physically plausible, that
the predictions are a real spread rather than a constant, that the shipped export was
not edited, and that the R2 against the run-to-failure lives in
/opt/golden/test_labels.csv (never present in the agent environment) reaches the 0.33
stated in the task.

Only pytest is installed, so the CSV reading and the R2 come from the standard library
rather than pandas/scikit-learn.
"""

import csv
import hashlib
import math
import pathlib

import pytest

PRED_PATH = pathlib.Path("/app/predictions.csv")
# Read from the root-only copy baked into the verifier image, not the /tests mount: the
# agent's crate is rebuilt and run in this container, and only this path is closed to it.
LABELS_PATH = pathlib.Path("/opt/golden/test_labels.csv")
TEST_FEATURES = pathlib.Path("/app/data/test_features.csv")
R2_BAR = 0.33

# frozen at task-build time; the shipped export must not be altered
TEST_FEATURES_SHA = "67bac94350dc4d5cdec6867315bdec3b722d959d712b7490aa3f4a624cd1220e"
TRAIN_SHA = "11ac515f5bc2e66346bd26e814ce0af07801333212546783443480296bf7cc43"


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


def _values(preds):
    """life_days column as floats, in file order."""
    _, rows = preds
    return [float(row[1]) for row in rows]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r2(pred, truth):
    mean = sum(truth) / len(truth)
    ss_res = sum((t - p) ** 2 for p, t in zip(pred, truth))
    ss_tot = sum((t - mean) ** 2 for t in truth)
    assert ss_tot > 0, "the run-to-failure lives have no variance"
    return 1.0 - ss_res / ss_tot


def test_shipped_data_unmodified():
    """The shipped train.csv and test_features.csv are byte-for-byte as issued."""
    assert _sha(TEST_FEATURES) == TEST_FEATURES_SHA, "test_features.csv was altered"
    assert _sha(pathlib.Path("/app/data/train.csv")) == TRAIN_SHA, "train.csv was altered"


def test_schema(preds):
    """predictions.csv carries exactly the two columns pump_id and life_days."""
    header, _ = preds
    assert header == ["pump_id", "life_days"], (
        f"predictions.csv must have exactly columns pump_id,life_days, got {header}"
    )


def test_covers_every_pump_exactly_once(preds, labels):
    """Every quoted pump has exactly one row — no duplicates, omissions or unknown ids."""
    _, rows = preds
    ids = [row[0] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate pump_id entries"
    missing = set(labels) - set(ids)
    extra = set(ids) - set(labels)
    assert not missing, f"{len(missing)} quoted pumps missing (e.g. {sorted(missing)[:3]})"
    assert not extra, f"{len(extra)} unknown pump_ids (e.g. {sorted(extra)[:3]})"


def test_lives_valid(preds):
    """Each life_days is a finite, strictly positive and physically plausible duration."""
    _, rows = preds
    for row in rows:
        try:
            value = float(row[1])
        except ValueError:
            pytest.fail(f"life_days is not numeric for pump_id {row[0]!r}: {row[1]!r}")
        assert math.isfinite(value), f"life_days is not finite for pump_id {row[0]!r}"
        assert 0.0 < value <= 100000.0, (
            f"life_days is not a plausible seal life for pump_id {row[0]!r}: {value}"
        )


def test_predictions_not_degenerate(preds):
    """Predictions differ across pumps rather than being one repeated constant."""
    v = _values(preds)
    assert len(set(v)) >= 50, "predictions are (near-)constant — not a real model"
    mean = sum(v) / len(v)
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))
    assert sd > 5.0, "predictions carry almost no variation"


def test_r2_bar(preds, labels):
    """R2 against the withheld run-to-failure lives reaches the 0.33 the task requires."""
    _, rows = preds
    pred = [float(row[1]) for row in rows]
    truth = [labels[row[0]] for row in rows]
    r2 = _r2(pred, truth)
    assert r2 >= R2_BAR, (
        f"R2 against the run-to-failure lives is {r2:.4f}, below the required "
        f"{R2_BAR:.2f} — the model does not carry to the quote batch"
    )
