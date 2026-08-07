"""Project validation checks; not required to run the application."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessing import (
    build_feature_frame,
    canonicalize_smiles,
    maximum_tanimoto_similarity,
)


PROJECT_DIR = Path(__file__).resolve().parent
bundle = joblib.load(PROJECT_DIR / "model" / "toxicity_model.joblib")
training = pd.read_csv(PROJECT_DIR / "data" / "training data 872026.csv")
test = pd.read_csv(PROJECT_DIR / "data" / "Test data 872026.csv")

endpoint_reverse = {value: key for key, value in bundle["endpoint_mapping"].items()}
cell_reverse = {value: key for key, value in bundle["cell_mapping"].items()}

maximum_difference = 0.0
maximum_detail = None
differences_over_tolerance = []
for frame in (training, test):
    for _, row in frame.iterrows():
        generated, canonical, _ = build_feature_frame(
            smiles=row["Canonical SMILES"],
            endpoint=endpoint_reverse[int(row["Endpoint"])],
            cell=cell_reverse[int(row["Cell"])],
            feature_names=bundle["feature_names"],
            descriptor_names=bundle["descriptor_names"],
        )
        expected = row[bundle["feature_names"]].to_numpy(dtype=float)
        absolute_differences = np.abs(
            generated.iloc[0].to_numpy(dtype=float) - expected
        )
        difference = float(np.max(absolute_differences))
        if difference > maximum_difference:
            position = int(np.argmax(absolute_differences))
            maximum_detail = {
                "smiles": row["Canonical SMILES"],
                "feature": bundle["feature_names"][position],
                "stored": float(expected[position]),
                "generated": float(generated.iloc[0, position]),
            }
        if difference >= 1e-4:
            differences_over_tolerance.append(difference)
        maximum_difference = max(maximum_difference, difference)
        assert canonical == canonicalize_smiles(row["Canonical SMILES"])[0]

print("Maximum detail:", maximum_detail)
print("Rows with a descriptor difference >= 1e-4:", len(differences_over_tolerance))
assert maximum_difference < 1e-4, maximum_difference

expected_reference_smiles = list(
    dict.fromkeys(
        canonicalize_smiles(smiles)[0]
        for smiles in training["Canonical SMILES"].astype(str)
    )
)
assert bundle["ad_reference_smiles"] == expected_reference_smiles
assert bundle["ad_method"] == {
    "fingerprint": "Morgan",
    "radius": 2,
    "n_bits": 2048,
    "similarity": "Tanimoto",
    "reported_value": "maximum similarity to unique training compounds",
}

for smiles in bundle["ad_reference_smiles"]:
    _, molecule = canonicalize_smiles(smiles)
    similarity = maximum_tanimoto_similarity(
        molecule,
        bundle["ad_reference_smiles"],
    )
    assert abs(similarity - 1.0) < 1e-12

_, novel_molecule = canonicalize_smiles("C")
novel_similarity = maximum_tanimoto_similarity(
    novel_molecule,
    bundle["ad_reference_smiles"],
)
assert 0.0 <= novel_similarity <= 1.0

sample = test.iloc[[0]][bundle["feature_names"]]
direct_prediction = float(bundle["model"].predict(sample)[0])
generated, _, _ = build_feature_frame(
    smiles=test.iloc[0]["Canonical SMILES"],
    endpoint=endpoint_reverse[int(test.iloc[0]["Endpoint"])],
    cell=cell_reverse[int(test.iloc[0]["Cell"])],
    feature_names=bundle["feature_names"],
    descriptor_names=bundle["descriptor_names"],
)
generated_prediction = float(bundle["model"].predict(generated)[0])
assert abs(direct_prediction - generated_prediction) < 1e-8

try:
    canonicalize_smiles("not_a_valid_smiles")
except ValueError:
    pass
else:
    raise AssertionError("Invalid SMILES was not rejected")

with (PROJECT_DIR / "training_notebook.ipynb").open(encoding="utf-8") as file:
    notebook = json.load(file)
assert notebook["nbformat"] == 4

print(f"Validated {len(training) + len(test)} dataset rows")
print(f"Maximum descriptor difference: {maximum_difference:.3e}")
print(f"Example prediction: {generated_prediction:.6f}")
print(f"AD reference chemicals: {len(bundle['ad_reference_smiles'])}")
print("Training-reference self-similarity: 1.000")
print(f"Novel test similarity: {novel_similarity:.3f}")
print(f"Notebook cells: {len(notebook['cells'])}")
