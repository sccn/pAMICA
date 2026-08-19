import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from pamica.mne_compat.core import AMICAICA


# ============================================================
# Configuration
# ============================================================

sfreq = 100.0
duration = 30.0
n_samples = int(sfreq * duration)

ch_names = [
    "MEG001",
    "MEG002",
    "MEG003",
    "MEG004",
]

ch_types = ["mag"] * len(ch_names)


# ============================================================
# Create artificial Raw data
# ============================================================

rng = np.random.default_rng(42)

data = rng.standard_normal(
    (len(ch_names), n_samples)
)

info = mne.create_info(
    ch_names=ch_names,
    sfreq=sfreq,
    ch_types=ch_types,
)

raw = mne.io.RawArray(
    data,
    info,
)


# ============================================================
# Add bad annotations
# ============================================================

raw.set_annotations(
    mne.Annotations(
        onset=[
            5.0,
            15.0,
            25.0,
        ],
        duration=[
            2.0,
            3.0,
            1.0,
        ],
        description=[
            "bad_test_1",
            "bad_test_2",
            "bad_test_3",
        ],
    )
)


# ============================================================
# Display Raw information
# ============================================================

print("=" * 70)
print("Raw data information")
print("=" * 70)

print(raw)

print("\nAnnotations:")
print(raw.annotations)

print("\nOriginal data shape:")
print(raw.get_data().shape)


# ============================================================
# Verify MNE annotation rejection
# ============================================================

data_all = raw.get_data(
    reject_by_annotation=None,
)

data_without_bad = raw.get_data(
    reject_by_annotation="omit",
)

print("\n" + "=" * 70)
print("MNE annotation rejection")
print("=" * 70)

print(
    f"Without annotation rejection: "
    f"{data_all.shape}"
)

print(
    f"With bad annotations omitted: "
    f"{data_without_bad.shape}"
)


# ============================================================
# Calculate expected sample counts
# ============================================================

bad_duration = sum(
    [2.0, 3.0, 1.0]
)

expected_bad_samples = int(
    bad_duration * sfreq
)

expected_remaining_samples = (
    n_samples - expected_bad_samples
)

print("\nExpected:")
print(f"Total samples:     {n_samples}")
print(f"Bad samples:       {expected_bad_samples}")
print(f"Remaining samples: {expected_remaining_samples}")


# ============================================================
# Check MNE behavior
# ============================================================

assert data_all.shape[1] == n_samples

assert (
    data_without_bad.shape[1]
    == expected_remaining_samples
)

print("\nPASS: MNE annotation rejection behaves as expected.")


# ============================================================
# Test pAMICA with reject_by_annotation=True
# ============================================================

print("\n" + "=" * 70)
print("Testing pAMICA: reject_by_annotation=True")
print("=" * 70)

ica_rej_ann = AMICAICA(
    n_models=2,
    random_state=42,
    verbose=True,
).fit(
    raw,
    max_iter=100,
    reject_by_annotation=True,
)

ica_rej_ann_mne = (
    ica_rej_ann.to_mne_ica(model_idx=0)
)


print(
    "\nMNE ICA data shape "
    "(reject_by_annotation=True):"
)
print(ica_rej_ann_mne)


# ============================================================
# Test pAMICA with reject_by_annotation=False
# ============================================================

print("\n" + "=" * 70)
print("Testing pAMICA: reject_by_annotation=False")
print("=" * 70)

ica_no_rej_ann = AMICAICA(
    n_models=2,
    random_state=42,
    verbose=True,
).fit(
    raw,
    max_iter=100,
    reject_by_annotation=False,
)

ica_no_rej_ann_mne = (
    ica_no_rej_ann.to_mne_ica(model_idx=0)
)

print(
    "\nMNE ICA data shape "
    "(reject_by_annotation=False):"
)
print(ica_no_rej_ann_mne)

print("\n" + "=" * 70)
print("pAMICA sample mask")
print("=" * 70)

print("Mask shape:", ica_rej_ann.good_sample_mask_.shape)
print("Good samples:", ica_rej_ann.good_sample_mask_.sum())
print("Rejected samples:", (~ica_rej_ann.good_sample_mask_).sum())
print("AMICA fitting samples:", ica_rej_ann._n_samples)

assert ica_rej_ann.good_sample_mask_.shape == (raw.n_times,)
assert ica_rej_ann.good_sample_mask_.sum() == 2400
assert (~ica_rej_ann.good_sample_mask_).sum() == 600
assert ica_rej_ann._n_samples == 2400

print("\nPASS: pAMICA sample mask is correct.")
ica_rej_ann._data_for(inst=raw)
model_probability = ica_rej_ann.get_model_probability(
    inst=raw,
)

print("Original model probability shape:")
print(model_probability.shape)


# ============================================================
# Get the sample mask used during AMICA fitting
# ============================================================

good_sample_mask = ica_rej_ann.good_sample_mask_

print("\nGood sample mask shape:")
print(good_sample_mask.shape)

print("Good samples:")
print(good_sample_mask.sum())

print("Rejected samples:")
print((~good_sample_mask).sum())


# ============================================================
# Reconstruct model probability on the original Raw timeline
#
# model_probability contains probabilities only for the samples
# that were used during AMICA fitting.
#
# good_sample_mask maps these probabilities back to the original
# Raw sample axis.
#
# Good samples  -> probability value
# Rejected      -> NaN
# ============================================================

n_models = model_probability.shape[0]
n_samples = raw.n_times

full_model_probability = np.full(
    (n_models, n_samples),
    np.nan,
    dtype=float,
)

full_model_probability[
    :,
    good_sample_mask,
] = model_probability


# ============================================================
# Verify rejected samples
# ============================================================

bad_mask = ~good_sample_mask

print("\nFull model probability shape:")
print(full_model_probability.shape)

print("\nProbability values in bad segments:")
print(
    full_model_probability[
        :,
        bad_mask,
    ]
)

n_bad_nan = np.isnan(
    full_model_probability[
        :,
        bad_mask,
    ]
).sum()

print("\nNumber of NaNs in bad segments:")
print(n_bad_nan)

expected_nan = n_models * bad_mask.sum()

if n_bad_nan == expected_nan:
    print(
        "PASS: All rejected samples contain NaN "
        "in the reconstructed model probability."
    )
else:
    print(
        "FAIL: Some rejected samples contain "
        "non-NaN probability values."
    )


# ============================================================
# Verify that good-sample probabilities were not changed
# ============================================================

reconstructed_good_probability = full_model_probability[
    :,
    good_sample_mask,
]

if np.allclose(
    reconstructed_good_probability,
    model_probability,
):
    print(
        "PASS: Good-sample probabilities were correctly "
        "mapped back to the original timeline."
    )
else:
    print(
        "FAIL: Good-sample probabilities changed "
        "during reconstruction."
    )


# ============================================================
# Create the original Raw time axis
# ============================================================

time = np.arange(
    n_samples
) / raw.info["sfreq"]


# ============================================================
# Convert to DataFrame
# ============================================================

model_probability_df = pd.DataFrame(
    full_model_probability.T,
    columns=[
        f"model_{i}"
        for i in range(n_models)
    ],
)

model_probability_df.insert(
    0,
    "time_s",
    time,
)


# ============================================================
# Plot model probabilities
# ============================================================

plt.figure(figsize=(14, 5))

for model_idx in range(n_models):
    plt.plot(
        time,
        full_model_probability[model_idx],
        label=f"model_{model_idx}",
    )

plt.xlabel("Time (s)")
plt.ylabel("Model probability")
plt.title("AMICA Model Probability")

if n_models > 1:
    plt.legend()

plt.tight_layout()
plt.show()
