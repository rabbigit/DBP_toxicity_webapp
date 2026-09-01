# DBP Toxicity Predictor

This project trains a LightGBM regression model using Mordred descriptors plus toxicity endpoint and cell/test-system variables. It provides an interface for single and batch predictions and reports maximum Morgan–Tanimoto similarity to the unique training chemicals.


## Encodings

| Endpoint | Code |
| --- | ---: |
| Cytotoxicity | 0 |
| Developmental | 1 |
| Genotoxicity | 2 |

| Cell/test system | Code |
| --- | ---: |
| CHO | 0 |
| HepG2 | 1 |
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

## Optional usage analytics

The app can record privacy-conscious usage statistics in Supabase:

- Total visits and estimated unique visitors
- Visitors by country or general region
- A global access map
- Single and batch analysis runs
- Total number of successfully predicted compounds

The tracker does **not** store submitted SMILES, uploaded files, prediction
values, or raw IP addresses. Unique visitors are estimates based on a salted,
one-way hash because public users do not sign in. Locations are rounded to
whole-degree coordinates before storage.

### 1. Create the analytics table

Create a free Supabase project, open its SQL Editor, and run
`supabase_analytics.sql`.

### 2. Add Streamlit secrets

In the Streamlit app settings, add the following secrets. Generate long random
values for `visitor_salt` and `dashboard_password`.

```toml
[analytics]
supabase_url = "https://YOUR_PROJECT.supabase.co"
supabase_key = "YOUR_SUPABASE_SECRET_KEY"
visitor_salt = "A_LONG_RANDOM_SECRET"
dashboard_password = "A_DIFFERENT_LONG_PASSWORD"
```

Use a current Supabase secret key (`sb_secret_...`). A legacy `service_role`
key is also supported. Never put either key in GitHub: these backend keys bypass
row-level security and must remain only in Streamlit's encrypted secrets.

If these secrets are absent or Supabase is temporarily unavailable, analytics
are silently skipped and predictions continue normally.

### 3. Deploy the private dashboard

Create a second Streamlit Community Cloud app from the same repository and use
`analytics_dashboard.py` as its entrypoint. Add the same secrets to that app.
Prefer Streamlit's private-app setting; the dashboard also requires the
configured password as a second safeguard.

The geographic lookup uses `ipapi.co` once per visitor session. When analytics
are enabled, the public app automatically displays a short collection notice.
Confirm whether your institution requires any additional privacy language.

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
