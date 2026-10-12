"""Tests du prétraitement et du protocole."""
import numpy as np
import pandas as pd

from scripts.data import INPUTS, NAMES, TARGET, canonical_id, load_data, make_groups, parse_numeric
from scripts.study import FeatureFilter, assert_separate, masked_training, observed_labels, preprocessing


def test_missing_target_is_not_class_zero():
    np.testing.assert_array_equal(observed_labels([500, 600, np.nan], 550), [0, 1, -1])


def test_documented_numeric_formats():
    assert parse_numeric("<5")[0] == 2.5
    assert parse_numeric("<5", "upper")[0] == 5
    assert parse_numeric("<5", "zero")[0] == 0
    assert parse_numeric("150-200")[0] == 175
    assert parse_numeric("67tot33res")[0] == 67
    assert parse_numeric("66totndres")[0] == 66
    assert parse_numeric("203(Hv30)")[0] == 203
    assert parse_numeric("0")[0] == 0


def test_inputs_exclude_all_mechanical_outputs():
    assert len(INPUTS) == 30
    assert set(INPUTS).isdisjoint(NAMES[30:])


def test_missing_is_not_zero_in_actual_archive():
    frame = load_data()
    assert frame.shape == (1652, 44)
    assert frame.loc[0, "Ni"] == 0
    assert pd.isna(frame.loc[0, "Cr"])
    assert frame[TARGET].notna().sum() == 738


def test_group_transitivity_and_suffixes():
    assert canonical_id("Evans-Ni/CMn-1990/1991-0Aaw") == canonical_id("Evans-Ni/CMn-1990/1991-0Aawch")
    assert canonical_id("Evans-StressRelief-1991-0.065Bch-70") == "Evans-StressRelief-1991-0.065B"
    frame = pd.DataFrame({c: [1., 2., 2., 9.] for c in INPUTS})
    frame["weld_id"] = ["Study-1991-Aaw", "Study-1991-Ach", "Other-1992-B", "Other-1992-C"]
    groups = make_groups(frame)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
    assert_separate(groups, [0, 1, 2], [3])


def test_feature_selection_uses_training_only():
    train = pd.DataFrame({"C": [1., 2., np.nan, 4.], "Ni": [np.nan, np.nan, np.nan, 1.],
                          "Cr": [0., 0., 0., 0.]})
    valid = pd.DataFrame({"C": [1000.], "Ni": [50.], "Cr": [12.]})
    selector = FeatureFilter().fit(train)
    assert selector.columns_ == ["C"]
    assert selector.transform(valid).columns.tolist() == ["C"]
    prep = preprocessing(scale=True).fit(train)
    imputer = prep.named_steps["transform"].named_transformers_["num"].named_steps["impute"]
    assert imputer.statistics_[0] == 2.
    prep.transform(valid)
    assert imputer.statistics_[0] == 2.


def test_unknown_categorical_is_supported():
    train = pd.DataFrame({"C": [1., 2., 3.], "weld_type": ["MMA", "SA", np.nan]})
    valid = pd.DataFrame({"C": [np.nan], "weld_type": ["inconnu"]})
    assert np.isfinite(preprocessing().fit(train).transform(valid)).all()


def test_masked_targets_never_set_threshold_or_labels():
    frame = pd.DataFrame({TARGET: np.arange(16, dtype=float) * 10 + 400})
    ids = frame.index
    visible, threshold, labels = masked_training(frame, ids, .25, 42)
    hidden = ids.difference(visible)
    perturbed = frame.copy()
    perturbed.loc[hidden, TARGET] = 100_000
    visible2, threshold2, labels2 = masked_training(perturbed, ids, .25, 42)
    np.testing.assert_array_equal(visible, visible2)
    assert threshold == threshold2
    pd.testing.assert_series_equal(labels, labels2)
    assert (labels.loc[hidden] == -1).all()
