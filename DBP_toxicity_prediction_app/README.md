# Chemical Toxicity Predictor — Professional Interface v2

This project trains a LightGBM regression model using selected 2D Mordred descriptors plus encoded toxicity endpoint and cell/test-system variables. It provides a professionally styled Streamlit interface for single and batch predictions and reports maximum Morgan–Tanimoto similarity to the unique training chemicals.

## Prediction workflow

1. Validate the submitted SMILES with RDKit.
2. Convert it to canonical SMILES.
3. Calculate the 48 selected 2D Mordred descriptors.
4. Encode the selected Endpoint and Cell values.
5. Arrange all 50 features in the exact training order.
6. Generate a prediction with the saved LightGBM model.
7. Calculate a Morgan fingerprint with radius 2 and 2,048 bits.
8. Report maximum Tanimoto similarity to all unique training chemicals.
9. Flag descriptors outside their individual training ranges in detailed diagnostics.

## Encodings

| Endpoint | Code |
| --- | ---: |
| Cytotoxicity | 0 |
| Developmental | 1 |
| Genotoxicity | 2 |

| Cell/test system | Code |
| --- | ---: |
| CHO | 0 |
| Hep G2 | 1 |
| Zebra fish | 2 |

## Install with Anaconda Prompt

Open Anaconda Prompt inside this project folder and run these commands one at a time:

```text
conda create -n toxicity-webapp python=3.12 -y
conda activate toxicity-webapp
pip install -r requirements.txt
```

The dependency versions are pinned because the descriptor values in the supplied datasets were generated with Mordred 1.2.0. A newer Mordred-compatible implementation produced different `CIC2` values for two dataset rows during verification.

## Train the model

The ZIP already includes a trained model. To reproduce it from the included datasets, run:

```text
python train_model.py
```

This evaluates the model using the provided test dataset and replaces `model/toxicity_model.joblib`.

## Start the application

```text
streamlit run app.py
```

Streamlit normally opens `http://localhost:8501` in your browser.

## Deploy publicly

Upload the contents of this folder—not the outer folder or ZIP—to a GitHub repository. `packages.txt` must be at the repository root. Confirm that GitHub contains all of these paths:

```text
app.py
preprocessing.py
requirements.txt
packages.txt
model/toxicity_model.joblib
```

Create a Streamlit Community Cloud app using `app.py` as the main file and Python 3.12 as the runtime.

## Batch-input format

Upload a CSV with these exact columns:

```text
SMILES,Endpoint,Cell
O=C1C=CC(=O)C(Cl)=C1,Cytotoxicity,CHO
```

## Important scientific notes

- The application reports the target as `Value` because no scientific unit or transformed-target label was specified.
- Maximum similarity is reported numerically without assigning an inside/outside label or applying a threshold.
- Similarity uses Morgan fingerprints (radius 2, 2,048 bits) and the Tanimoto coefficient.
- The supplied training and test datasets contain overlapping chemicals and some identical model inputs. This may make the reported test performance optimistic for genuinely unseen chemicals.
- Before journal publication, document the target definition and unit, final validation design, applicability domain, uncertainty, citation, authorship, and model version.

## Project structure

```text
chemical_toxicity_streamlit_app/
├── app.py
├── preprocessing.py
├── train_model.py
├── training_notebook.ipynb
├── requirements.txt
├── packages.txt
├── README.md
├── MODEL_CARD.md
├── data/
│   ├── training data 872026.csv
│   └── Test data 872026.csv
└── model/
    └── toxicity_model.joblib
```
