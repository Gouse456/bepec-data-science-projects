# Save as streamlit_app_pipeline.py
import streamlit as st
import requests
import pandas as pd
from io import BytesIO

API_BASE = "http://localhost:8000"  # Change if deployed elsewhere

st.title("Hiring Predictor (Pipeline)")

tab1, tab2 = st.tabs(["Batch CSV Upload", "Single Candidate"])

# --- Tab 1: Batch CSV Upload ---
with tab1:
    uploaded = st.file_uploader("Upload CSV with candidates (C1..C15 + Id)", type=["csv"])
    if uploaded is not None:
        files = {"file": ("input.csv", uploaded, "text/csv")}
        with st.spinner("Processing..."):
            resp = requests.post(f"{API_BASE}/process_csv", files=files)
        if resp.status_code == 200:
            data = resp.json()
            out_file = data.get("output_file")
            download_url = data.get("download_url")
            count = data.get("count", 0)

            st.success(f"Processed {count} rows. Output file: {out_file}")

            if download_url:
                r = requests.get(f"{API_BASE}{download_url}")
                if r.status_code == 200:
                    content = r.content
                    st.download_button("Download results", data=BytesIO(content),
                                       file_name=out_file, mime="text/csv")
                    preview = pd.read_csv(BytesIO(content))
                    st.write("Preview:")
                    st.dataframe(preview.head())
                else:
                    st.error("Could not fetch output file.")
            else:
                st.info("No download URL returned.")
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")

# --- Tab 2: Single Candidate ---
with tab2:
    st.write("Enter candidate details (C1..C15):")
    candidate = {}
    for col in [f"C{i}" for i in range(1, 16)]:
        val = st.text_input(col)
        try:
            # Try converting to float if numeric
            if val != "":
                val = float(val) if val.replace('.', '', 1).isdigit() else val
        except:
            pass
        candidate[col] = val if val != "" else None

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            resp = requests.post(f"{API_BASE}/predict_single", json=candidate)
        if resp.status_code == 200:
            result = resp.json()
            st.success(f"Label: {result['Label']}")
            st.write(f"Probability: {result['Probability']:.4f}")
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")
