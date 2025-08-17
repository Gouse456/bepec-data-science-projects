from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import joblib
import os

# Load model and training feature names
model, feature_names = joblib.load("model_pipeline.joblib")
categorical_cols = ['C1','C4','C5','C6','C7','C9','C10','C12','C13']

app = FastAPI()

@app.post("/predict_single")
async def predict_single(data: dict):
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][pred]

    return JSONResponse({"Label": int(pred), "Probability": float(prob)})

@app.post("/process_csv")
async def process_csv(file: UploadFile = File(...)):
    # Read uploaded CSV
    df = pd.read_csv(file.file)

    # Store ID column if exists
    if "Id" in df.columns:
        ids = df["Id"]
    else:
        ids = pd.Series(range(len(df)), name="Id")

    # Encode categorical columns
    df_enc = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_enc = df_enc.reindex(columns=feature_names, fill_value=0)

    # Predict
    preds = model.predict(df_enc)
    probs = model.predict_proba(df_enc).max(axis=1)

    # Create output DataFrame
    results = pd.DataFrame({"Id": ids, "Label": preds, "Probability": probs})

    # Save to file
    output_file = "output.csv"
    results.to_csv(output_file, index=False)

    return JSONResponse({
        "output_file": output_file,
        "download_url": "/download/output.csv",
        "count": len(results)
    })

@app.get("/download/{filename}")
async def download_file(filename: str):
    filepath = os.path.join(".", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="text/csv", filename=filename)
    return JSONResponse({"error": "File not found"}, status_code=404)
