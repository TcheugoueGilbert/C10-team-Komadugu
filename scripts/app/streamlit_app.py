"""
Streamlit app for the Gemma-2-2B layer-14 toxicity linear probe.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from src import config
from src.inference import ToxicityClassifier

st.set_page_config(
    page_title="Toxicity Probe",
    page_icon="🛡️",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading Gemma-2-2B backbone and trained probe...")
def get_classifier():
    """Load the backbone + probe once and cache across reruns/sessions."""
    return ToxicityClassifier(model_path=config.MODEL_PATH, backbone_name=config.MODEL_NAME)


def render_result(result: dict):
    label = result["label_name"]
    prob = result["probability_toxic"]

    if label == "toxic":
        st.error(f"⚠️ **Toxic** — P(toxic) = {prob:.3f}")
    else:
        st.success(f"✅ **Safe** — P(toxic) = {prob:.3f}")

    st.progress(min(max(prob, 0.0), 1.0))


def main():
    st.title("🛡️ Toxicity Probe")
    st.caption(
        "A linear probe (StandardScaler + Logistic Regression) trained on "
        f"frozen layer-{config.LAYER} embeddings from **{config.MODEL_NAME}**, "
        "fit on the Civil Comments dataset."
    )

    if not os.path.exists(config.MODEL_PATH):
        st.warning(
            f"No trained probe found at `{config.MODEL_PATH}`.\n\n"
            "Run `scripts/01_extract_embeddings.py` then "
            "`scripts/02_train_probe.py` first, or place a `classifier.joblib` "
            "in the `models/` directory."
        )
        st.stop()

    tab_single, tab_batch, tab_about = st.tabs(["Single comment", "Batch (CSV)", "About"])

    with tab_single:
        text = st.text_area(
            "Enter a comment to classify:",
            placeholder="Type or paste a comment here...",
            height=120,
        )

        if st.button("Classify", type="primary", use_container_width=True):
            if not text.strip():
                st.info("Please enter some text.")
            else:
                with st.spinner("Running inference..."):
                    classifier = get_classifier()
                    result = classifier.predict(text)
                render_result(result)

    with tab_batch:
        st.write("Upload a CSV with a `text` column to classify many comments at once.")
        uploaded = st.file_uploader("CSV file", type=["csv"])

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns:
                st.error("CSV must contain a `text` column.")
            else:
                if st.button("Classify batch", type="primary"):
                    with st.spinner(f"Classifying {len(df)} comments..."):
                        classifier = get_classifier()
                        results = classifier.predict(df["text"].astype(str).tolist())
                        results_df = pd.DataFrame(results)

                    st.dataframe(results_df, use_container_width=True)
                    st.download_button(
                        "Download results as CSV",
                        data=results_df.to_csv(index=False).encode("utf-8"),
                        file_name="toxicity_predictions.csv",
                        mime="text/csv",
                    )

    with tab_about:
        st.markdown(
            f"""
            **Model architecture**

            - Backbone: `{config.MODEL_NAME}` (frozen, not fine-tuned)
            - Probed layer: hidden-state index `{config.LAYER}`
            - Pooling: mean pooling over non-padding tokens
            - Classifier head: `StandardScaler` + `LogisticRegression`
            - Max input length: `{config.MAX_LENGTH}` tokens

            **Training data**

            - Dataset: [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments)
            - Binary label: `toxicity >= {config.TOXICITY_THRESHOLD}`
            - Train subset size: `{config.TRAIN_SIZE}`
            - Validation subset size: `{config.VAL_SIZE}`

            See `README.md` for full training and evaluation details.
            """
        )


if __name__ == "__main__":
    main()
