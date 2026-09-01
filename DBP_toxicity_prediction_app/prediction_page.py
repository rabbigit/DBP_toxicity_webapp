"""Professional Streamlit interface for chemical toxicity prediction."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from rdkit.Chem import Draw

from analytics import analytics_enabled, log_prediction
from preprocessing import (
    CELL_MAPPING,
    ENDPOINT_MAPPING,
    build_feature_frame,
    count_outside_training_range,
    maximum_tanimoto_similarity,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "toxicity_model.joblib"

@st.cache_resource
def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model is missing. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_one(bundle: dict, smiles: str, endpoint: str, cell: str) -> dict:
    frame, canonical, molecule = build_feature_frame(
        smiles=smiles,
        endpoint=endpoint,
        cell=cell,
        feature_names=bundle["feature_names"],
        descriptor_names=bundle["descriptor_names"],
    )
    prediction = float(bundle["model"].predict(frame)[0])
    similarity = maximum_tanimoto_similarity(
        molecule,
        bundle["ad_reference_smiles"],
    )
    outside = count_outside_training_range(
        frame,
        bundle["descriptor_names"],
        bundle["descriptor_ranges"],
    )
    return {
        "prediction": prediction,
        "maximum_similarity": similarity,
        "canonical_smiles": canonical,
        "molecule": molecule,
        "outside_descriptors": outside,
    }


def render_metric_card(label: str, value: str) -> None:
    """Render a metric card with colors that are independent of Streamlit's theme."""
    st.markdown(
        f"""
        <div class="custom-metric-card">
            <div class="custom-metric-label">{label}</div>
            <div class="custom-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(15,118,110,0.10), transparent 28%),
            radial-gradient(circle at 95% 8%, rgba(30,94,135,0.09), transparent 25%),
            #F4F8FA;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }
    /*
       The page uses a light canvas even when Streamlit is running with a dark
       theme. Native Markdown otherwise inherits the theme's white text color.
    */
    [data-testid="stMain"] [data-testid="stMarkdownContainer"]
        :is(h1, h2, h3, h4, h5, h6, p, li, strong, em) {
        color: #16324F !important;
        -webkit-text-fill-color: #16324F !important;
        opacity: 1 !important;
    }
    [data-testid="stMain"] label,
    [data-testid="stMain"] label p,
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        color: #16324F !important;
        -webkit-text-fill-color: #16324F !important;
        opacity: 1 !important;
    }
    .hero {
        background: linear-gradient(120deg, #0A3D62 0%, #0F766E 100%);
        border-radius: 22px;
        padding: 2.2rem 2.5rem;
        color: white !important;
        box-shadow: 0 18px 45px rgba(10,61,98,0.18);
        margin-bottom: 1.4rem;
    }
    .hero h1 {
        color: white !important;
        -webkit-text-fill-color: white !important;
        font-size: 2.45rem;
        line-height: 1.1;
        margin: 0 0 0.7rem 0;
        letter-spacing: -0.035em;
    }
    .hero p {
        color: rgba(255,255,255,0.88) !important;
        -webkit-text-fill-color: rgba(255,255,255,0.88) !important;
        font-size: 1.05rem;
        max-width: 780px;
        margin: 0;
        line-height: 1.65;
    }
    .eyebrow {
        display: inline-block;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.24);
        border-radius: 999px;
        padding: 0.32rem 0.75rem;
        margin-bottom: 0.9rem;
        color: #D9FFFF !important;
        -webkit-text-fill-color: #D9FFFF !important;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .section-title {
        color: #16324F;
        font-size: 1.25rem;
        font-weight: 750;
        margin: 0.25rem 0 0.25rem 0;
    }
    .section-copy {
        color: #5A6E7F;
        margin-bottom: 1.1rem;
    }
    .custom-metric-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid #D7E5E8;
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 8px 24px rgba(22,50,79,0.07);
        min-height: 108px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .custom-metric-card .custom-metric-label,
    .custom-metric-card .custom-metric-label * {
        color: #16324F !important;
        -webkit-text-fill-color: #16324F !important;
        opacity: 1 !important;
        font-size: 0.95rem;
        font-weight: 700;
        line-height: 1.3;
        margin-bottom: 0.5rem;
    }
    .custom-metric-value {
        color: #0F766E !important;
        -webkit-text-fill-color: #0F766E !important;
        font-size: 2.25rem;
        font-weight: 500;
        line-height: 1.1;
    }
    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid #D7E5E8;
        border-radius: 18px;
        padding: 1.2rem 1.25rem 0.5rem 1.25rem;
        box-shadow: 0 10px 28px rgba(22,50,79,0.06);
    }
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        min-height: 2.75rem;
        font-weight: 700;
        border: 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: rgba(255,255,255,0.72);
        border-radius: 14px;
        padding: 0.35rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding-left: 1rem;
        padding-right: 1rem;
        font-weight: 650;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #000000 !important;
        opacity: 1 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span,
    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span {
        color: #0F766E !important;
        opacity: 1 !important;
    }
    .method-note {
        background: #E8F4F3;
        border-left: 4px solid #0F766E;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        color: #244B55 !important;
        -webkit-text-fill-color: #244B55 !important;
        margin-top: 1rem;
    }
    .footer {
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #D7E5E8;
        color: #6C7F8D;
        text-align: center;
        font-size: 0.84rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    bundle = load_bundle()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

metrics = bundle["metrics"]

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine learning · Molecular descriptors</div>
        <h1>Multi-Endpoint DBP Toxicity Predictor</h1>
        <p>
            Predict a toxicity value from molecular structure, endpoint, and cell/test
            system using selected Mordred descriptors and a LightGBM regression model.
            Each result includes maximum structural similarity to the training compounds.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Model snapshot")
    render_metric_card("Test R²", f"{metrics['r2']:.2f}")
    render_metric_card("Test RMSE", f"{metrics['rmse']:.2f}")
    st.markdown("---")
    st.markdown("**Training records**")
    st.write(f"{metrics['training_rows']:,}")
    st.markdown("**Selected descriptors**")
    st.write(f"{len(bundle['descriptor_names'])}")
    st.markdown("**Similarity method**")
    st.write("Morgan radius 2 · 2,048 bits · Tanimoto")
    st.markdown("---")
    st.caption(
        "Research-use predictor. Interpret outputs together with validation, "
        "study limitations, and experimental evidence."
    )
    if analytics_enabled():
        st.caption(
            "Anonymous usage statistics and approximate geographic location "
            "are collected. Chemical inputs and prediction results are not stored."
        )

prediction_tab, batch_tab, information_tab = st.tabs(
    ["Predict one chemical", "Batch prediction", "Model & methods"]
)

with prediction_tab:
    st.markdown('<div class="section-title">Prediction inputs</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Provide a valid SMILES and select the experimental context.</div>',
        unsafe_allow_html=True,
    )

    with st.form("single_prediction_form"):
        smiles = st.text_input(
            "Chemical SMILES",
            value="O=C1C=CC(=O)C(Cl)=C1",
            help="The structure will be validated and converted to canonical SMILES.",
        )
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            endpoint = st.selectbox("Toxicity endpoint", list(ENDPOINT_MAPPING))
        with input_col2:
            cell = st.selectbox("Cell/test system", list(CELL_MAPPING))
        submitted = st.form_submit_button(
            "Generate prediction",
            type="primary",
            width="stretch",
        )

    if submitted:
        try:
            result = predict_one(bundle, smiles, endpoint, cell)
            log_prediction("single_prediction", item_count=1)
            st.markdown("### Prediction result")
            structure_col, result_col = st.columns([1.05, 1.35], gap="large")

            with structure_col:
                st.markdown("**Molecular structure**")
                image = Draw.MolToImage(result["molecule"], size=(520, 330))
                st.image(image, width="stretch")

            with result_col:
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    render_metric_card(
                        "Predicted Value", f"{result['prediction']:.4f}"
                    )
                with metric_col2:
                    render_metric_card(
                        "Maximum Tanimoto similarity",
                        f"{result['maximum_similarity']:.3f}",
                    )
                st.markdown("**Canonical SMILES**")
                st.code(result["canonical_smiles"], language=None)
                st.markdown(
                    """
                    <div class="method-note">
                        Similarity is the maximum Tanimoto coefficient between the
                        submitted molecule and all unique training compounds using
                        Morgan fingerprints (radius 2, 2,048 bits). If the similarity
                        value is 1, it indicates the DBP is present in the training data. 
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                outside = result["outside_descriptors"]
                with st.expander("Descriptor diagnostics"):
                    st.write(
                        f"Descriptors outside their individual training ranges: "
                        f"**{len(outside)} of {len(bundle['descriptor_names'])}**"
                    )
                    if outside:
                        st.write(", ".join(outside))
                    else:
                        st.write("None")
        except (ValueError, RuntimeError) as error:
            st.error(str(error))

with batch_tab:
    st.markdown('<div class="section-title">Batch prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Upload multiple chemicals and download predictions with maximum similarity values.</div>',
        unsafe_allow_html=True,
    )
    template = pd.DataFrame(
        {
            "SMILES": ["O=C1C=CC(=O)C(Cl)=C1"],
            "Endpoint": ["Cytotoxicity"],
            "Cell": ["CHO"],
        }
    )
    button_col, upload_col = st.columns([1, 2])
    with button_col:
        st.download_button(
            "Download CSV template",
            data=template.to_csv(index=False),
            file_name="prediction_template.csv",
            mime="text/csv",
            width="stretch",
        )
    with upload_col:
        uploaded = st.file_uploader(
            "Upload prediction CSV",
            type=["csv"],
            label_visibility="collapsed",
        )

    if uploaded is not None and st.button(
        "Run batch prediction", type="primary", width="stretch"
    ):
        try:
            batch = pd.read_csv(uploaded)
            required = ["SMILES", "Endpoint", "Cell"]
            missing = [name for name in required if name not in batch.columns]
            if missing:
                raise ValueError("Missing CSV columns: " + ", ".join(missing))

            rows = []
            progress = st.progress(0, text="Calculating descriptors and similarities...")
            total = max(len(batch), 1)
            for position, (_, row) in enumerate(batch.iterrows(), start=1):
                try:
                    output = predict_one(
                        bundle,
                        str(row["SMILES"]),
                        str(row["Endpoint"]),
                        str(row["Cell"]),
                    )
                    rows.append(
                        {
                            "Canonical SMILES": output["canonical_smiles"],
                            "Predicted Value": output["prediction"],
                            "Maximum Tanimoto Similarity": output["maximum_similarity"],
                            "Out-of-range descriptor count": len(
                                output["outside_descriptors"]
                            ),
                            "Status": "Success",
                        }
                    )
                except (ValueError, RuntimeError) as error:
                    rows.append(
                        {
                            "Canonical SMILES": "",
                            "Predicted Value": None,
                            "Maximum Tanimoto Similarity": None,
                            "Out-of-range descriptor count": None,
                            "Status": str(error),
                        }
                    )
                progress.progress(
                    min(position / total, 1.0),
                    text=f"Processed {position} of {len(batch)} chemicals",
                )
            progress.empty()

            output_table = pd.concat(
                [batch.reset_index(drop=True), pd.DataFrame(rows)], axis=1
            )
            successful_predictions = sum(
                row["Status"] == "Success" for row in rows
            )
            if successful_predictions:
                log_prediction(
                    "batch_prediction", item_count=successful_predictions
                )
            st.success(f"Completed {len(output_table)} rows.")
            st.dataframe(output_table, width="stretch", hide_index=True)
            st.download_button(
                "Download prediction results",
                data=output_table.to_csv(index=False),
                file_name="toxicity_predictions.csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
        except Exception as error:
            st.error(f"Could not process the uploaded file: {error}")

with information_tab:
    st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        render_metric_card("Test R²", f"{metrics['r2']:.2f}")
    with metric_col2:
        render_metric_card("Test RMSE", f"{metrics['rmse']:.2f}")
    with metric_col3:
        render_metric_card("Training rows", f"{metrics['training_rows']:,}")
    with metric_col4:
        render_metric_card("Test rows", f"{metrics['test_rows']:,}")

    methods_col, encoding_col = st.columns(2, gap="large")
    with methods_col:
        st.markdown("### Prediction workflow")
        st.markdown(
            """
            1. Validate and canonicalize SMILES with RDKit.
            2. Calculate the 48 selected 2D Mordred descriptors.
            3. Encode endpoint and cell/test system.
            4. Align all features to the exact training order.
            5. Predict `Value` using LightGBM.
            6. Calculate maximum Morgan–Tanimoto similarity.
            """
        )
    with encoding_col:
        st.markdown("### Encodings")
        endpoint_table = pd.DataFrame(
            ENDPOINT_MAPPING.items(), columns=["Endpoint", "Code"]
        )
        cell_table = pd.DataFrame(CELL_MAPPING.items(), columns=["Cell", "Code"])
        st.dataframe(endpoint_table, hide_index=True, width="stretch")
        st.dataframe(cell_table, hide_index=True, width="stretch")

    st.info(
        "The target is displayed as `Value`"    )

st.markdown(
    """
    <div class="footer">
        DBP Toxicity Predictor · LightGBM + Mordred + RDKit
    </div>
    """,
    unsafe_allow_html=True,
)
