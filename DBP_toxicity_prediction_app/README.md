# DBP Toxicity Predictor

This project trains a LightGBM regression model using Mordred descriptors plus toxicity endpoint and cell/test-system variables. It provides an interface for single and batch predictions and reports maximum Morgan–Tanimoto similarity to the unique training chemicals.

## Prediction workflow

1. Validate the submitted SMILES with RDKit.
2. Convert it to canonical SMILES.
3. Calculate the Mordred descriptors.
4. Encode the selected Endpoint and Cell values.
5. Arrange all 50 features in the exact training order.
6. Generate a prediction with the saved LightGBM model.
7. Report the maximum Tanimoto similarity to all unique training chemicals.

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
| Zebrafish | 2 |

## Install with Anaconda Prompt

Open Anaconda Prompt inside this project folder and run these commands one at a time:

```text
conda create -n toxicity-webapp python=3.11 -y
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

Create a Streamlit Community Cloud app using `app.py` as the main file and Python 3.11 as the runtime.

## Batch-input format

Upload a CSV with these exact columns:

```text
SMILES,Endpoint,Cell
O=C1C=CC(=O)C(Cl)=C1,Cytotoxicity,CHO
```

## Important scientific notes

- The application reports the target as `Value`.
- Similarity uses Morgan fingerprints (radius 2, 2,048 bits) and the Tanimoto coefficient.

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
