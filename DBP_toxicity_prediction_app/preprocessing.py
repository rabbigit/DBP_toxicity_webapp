"""SMILES validation and feature generation for toxicity prediction."""

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ENDPOINT_MAPPING = {
    "Cytotoxicity": 0,
    "Developmental": 1,
    "Genotoxicity": 2,
}

CELL_MAPPING = {
    "CHO": 0,
    "Hep G2": 1,
    "Zebra fish": 2,
}

MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def canonicalize_smiles(smiles: str) -> tuple[str, Chem.Mol]:
    """Validate a SMILES string and return its canonical form and molecule."""
    cleaned = str(smiles).strip()
    if not cleaned:
        raise ValueError("Please enter a SMILES string.")

    molecule = Chem.MolFromSmiles(cleaned)
    if molecule is None:
        raise ValueError("RDKit could not parse this SMILES string.")

    canonical = Chem.MolToSmiles(molecule, canonical=True)
    return canonical, molecule


@lru_cache(maxsize=8)
def _selected_calculator(descriptor_names: tuple[str, ...]) -> Calculator:
    """Build and cache a calculator containing only the required descriptors."""
    all_calculator = Calculator(descriptors, ignore_3D=True)
    descriptor_lookup = {
        str(descriptor): descriptor for descriptor in all_calculator.descriptors
    }
    missing = [name for name in descriptor_names if name not in descriptor_lookup]
    if missing:
        raise RuntimeError(
            "The installed Mordred version does not provide these descriptors: "
            + ", ".join(missing)
        )
    selected = [descriptor_lookup[name] for name in descriptor_names]
    return Calculator(selected, ignore_3D=True)


def _as_finite_float(value: Any, descriptor_name: str) -> float:
    """Convert a descriptor result to a finite numeric value."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Mordred could not calculate the required descriptor '{descriptor_name}'."
        ) from error

    if not np.isfinite(numeric_value):
        raise ValueError(
            f"Descriptor '{descriptor_name}' produced a non-finite value."
        )
    return numeric_value


def calculate_selected_descriptors(
    molecule: Chem.Mol,
    descriptor_names: list[str],
) -> dict[str, float]:
    """Calculate the selected 2D Mordred descriptors in a fixed order."""
    calculator = _selected_calculator(tuple(descriptor_names))
    result = calculator(molecule)
    values: dict[str, float] = {}
    for descriptor, value in zip(calculator.descriptors, result):
        name = str(descriptor)
        values[name] = _as_finite_float(value, name)
    return values


def build_feature_frame(
    smiles: str,
    endpoint: str,
    cell: str,
    feature_names: list[str],
    descriptor_names: list[str],
) -> tuple[pd.DataFrame, str, Chem.Mol]:
    """Create one model-ready row with exact training names and order."""
    if endpoint not in ENDPOINT_MAPPING:
        raise ValueError(f"Unknown endpoint: {endpoint}")
    if cell not in CELL_MAPPING:
        raise ValueError(f"Unknown cell: {cell}")

    canonical, molecule = canonicalize_smiles(smiles)
    row = calculate_selected_descriptors(molecule, descriptor_names)
    row["Endpoint"] = ENDPOINT_MAPPING[endpoint]
    row["Cell"] = CELL_MAPPING[cell]

    missing = [name for name in feature_names if name not in row]
    if missing:
        raise RuntimeError("Missing model features: " + ", ".join(missing))

    frame = pd.DataFrame([[row[name] for name in feature_names]], columns=feature_names)
    return frame, canonical, molecule


def count_outside_training_range(
    frame: pd.DataFrame,
    descriptor_names: list[str],
    descriptor_ranges: dict[str, dict[str, float]],
) -> list[str]:
    """List descriptors outside the observed training minimum/maximum."""
    outside: list[str] = []
    for name in descriptor_names:
        value = float(frame.iloc[0][name])
        limits = descriptor_ranges[name]
        if value < limits["min"] or value > limits["max"]:
            outside.append(name)
    return outside


@lru_cache(maxsize=1)
def _morgan_generator():
    """Return the fixed Morgan fingerprint generator used for similarity."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
    )


@lru_cache(maxsize=4)
def _reference_fingerprints(reference_smiles: tuple[str, ...]):
    """Generate and cache fingerprints for the unique training chemicals."""
    generator = _morgan_generator()
    fingerprints = []
    for smiles in reference_smiles:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise RuntimeError(f"Invalid training reference SMILES: {smiles}")
        fingerprints.append(generator.GetFingerprint(molecule))
    return tuple(fingerprints)


def maximum_tanimoto_similarity(
    molecule: Chem.Mol,
    reference_smiles: list[str],
) -> float:
    """Return maximum Morgan–Tanimoto similarity to the training chemicals."""
    if not reference_smiles:
        raise RuntimeError("No training compounds are available for similarity checking.")
    query_fingerprint = _morgan_generator().GetFingerprint(molecule)
    similarities = DataStructs.BulkTanimotoSimilarity(
        query_fingerprint,
        list(_reference_fingerprints(tuple(reference_smiles))),
    )
    return float(max(similarities))
