# Model Card: Chemical Toxicity Predictor

## Intended use

This LightGBM regression model predicts the dataset target labeled `Value` from a chemical structure, toxicity endpoint, and cell/test system. It is intended for research demonstration and screening support, not as a substitute for experimental testing or regulatory assessment.

## Inputs

- One valid SMILES string
- Endpoint: Cytotoxicity, Developmental, or Genotoxicity
- Cell/test system: CHO, Hep G2, or Zebra fish
- Internally calculated features: 48 selected 2D Mordred descriptors

## Output

- Predicted `Value`
- Canonical SMILES
- Maximum Morgan–Tanimoto similarity to the unique training chemicals
- Count of selected descriptors outside their individual observed training ranges

The scientific definition, transformation, and unit of `Value` must be added before publication.

## Model and data

- Algorithm: `LGBMRegressor`
- Training records: 420
- Provided test records: 90
- Model features: 50 (48 descriptors + Endpoint + Cell)
- Test R²: 0.8401
- Test RMSE: 0.4721

## Preprocessing

RDKit validates and canonicalizes each SMILES. Mordred 1.2.0 calculates the required 2D descriptors with `ignore_3D=True`. The application then selects and orders the exact training columns before prediction.

For the similarity calculation, RDKit generates a 2,048-bit Morgan fingerprint with radius 2 for the submitted molecule and each unique training compound. The application reports only the maximum Tanimoto similarity value. It does not apply a threshold or assign an inside/outside applicability-domain label.

## Known limitations

- The supplied training and test sets share 57 unique chemicals.
- Thirteen `SMILES + Endpoint + Cell` combinations occur in both sets.
- Fourteen exact model-input combinations occur in both sets.
- Consequently, the reported test performance may be optimistic for genuinely unseen chemicals.
- The maximum Tanimoto similarity is a structural-neighbor measure and is reported without an applicability-domain threshold.
- The current model does not report a prediction interval or calibrated uncertainty.
- Descriptor generation and model loading depend on the pinned software versions in `requirements.txt`.

## Recommended work before journal publication

1. Define the target transformation and unit.
2. Evaluate using a chemical-group split or an independent external dataset.
3. Add a validated applicability-domain method.
4. Add prediction uncertainty if supported by the study.
5. Add the manuscript citation, authors, data availability statement, license, and model version.
