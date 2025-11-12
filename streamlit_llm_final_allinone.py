import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.prompts.prompt import PromptTemplate
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from fpdf import FPDF
import joblib

# -------------------------
# Load Environment
# -------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------
# Init LLM
# -------------------------
llm = ChatGroq(
    temperature=0.7,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)

# -------------------------
# Helper functions
# -------------------------
def run_llm(prompt_template: str, data_sample: pd.DataFrame):
    prompt = PromptTemplate.from_template(prompt_template).format(data=data_sample.to_string(index=False))
    return llm.invoke(prompt).content

def clean_python_code(raw_code: str):
    return raw_code.strip().strip("```").replace("python", "").strip()

def save_fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    return output.getvalue()

def generate_pdf(df_results, insights, figs_dict, filename="ML_Report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "All-in-One Auto ML Report", ln=True, align="C")
    
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 10, "Metrics Table:", ln=True)
    pdf.set_font("Arial", "", 10)
    for i, row in df_results.iterrows():
        line = ", ".join([f"{col}: {row[col]}" for col in df_results.columns])
        pdf.multi_cell(0, 5, line)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Insights:", ln=True)
    pdf.set_font("Arial", "", 10)
    for insight in insights:
        pdf.multi_cell(0, 5, f"- {insight}")
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Visualizations:", ln=True)
    for key, fig in figs_dict.items():
        buf = save_fig_to_bytes(fig)
        pdf.image(buf, w=180)
    
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

def get_excel_download_link(df, filename="results.xlsx"):
    val = to_excel(df)
    b64 = base64.b64encode(val).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Excel</a>'

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="All-in-One LLM Auto ML", layout="wide")
st.title("🚀 All-in-One Production-ready LLM Auto ML Dashboard")

# Upload training dataset
uploaded_train = st.file_uploader("Upload Training Dataset (CSV)", type="csv", key="train")
if uploaded_train:
    df_train_iter = pd.read_csv(uploaded_train, chunksize=100000)
    df_train = pd.concat([chunk for chunk in df_train_iter], ignore_index=True)
else:
    st.stop()

st.subheader("Train Dataset Preview (first 100 rows)")
st.dataframe(df_train.head(100))

# Target & features
target_col = st.selectbox("Pilih kolom Target", df_train.columns)
feature_cols = st.multiselect("Pilih fitur (kosong = semua kecuali target)", [c for c in df_train.columns if c != target_col])
if feature_cols:
    df_train_features = df_train[feature_cols + [target_col]]
else:
    df_train_features = df_train.copy()

# Downsample for LLM prompt
df_sample = df_train_features.sample(n=min(50000, len(df_train_features)), random_state=42) if len(df_train_features) > 50000 else df_train_features.copy()

# Upload holdout / outvalidation dataset
uploaded_holdout = st.file_uploader("Upload Out-of-Validation Dataset (CSV)", type="csv", key="holdout")

# Model selection
available_models = ["Logistic", "RandomForest", "XGBoost", "GradientBoosting", "AdaBoost", "ExtraTrees", "SVM", "KNN"]
selected_models = st.multiselect("Pilih model yang ingin dijalankan", available_models, default=available_models)

# LLM prompt
default_prompt = f"""
Buat pipeline classification untuk dataset ini dengan kolom Target = {target_col}.
- Jalankan model: {', '.join(selected_models)}
- Sertakan preprocessing, cross-validation, hyperparameter tuning
- Pilih model terbaik berdasarkan F1-score atau AUC
- Simpan pipeline terbaik ke variable 'pipeline_best'
- Buat feature importance plot & performance comparison figure(s)
- Tulis insight narasi model terbaik & ranking
- Simpan hasil ke df_results, insights, pipeline_best, y_proba, figure(s)
- Gunakan sample {len(df_sample)} row untuk prompt LLM
{df_sample.head(5)}
"""
user_prompt = st.text_area("Custom prompt for LLM", value=default_prompt, height=300)

# Run pipeline
if st.button("Generate & Run All-in-One Pipeline"):
    with st.spinner("LLM generate pipeline & running..."):
        raw_code = run_llm(user_prompt, df_sample)
        cleaned_code = clean_python_code(raw_code)
        st.subheader("Generated Python Script")
        st.code(cleaned_code, language="python")
        
        local_vars = {'df': df_train_features.copy(), 'pd': pd, 'plt': plt, 'np': np}
        exec(cleaned_code, {}, local_vars)
        
        # Cache best pipeline
        if 'pipeline_best' in local_vars:
            pipeline_best = local_vars['pipeline_best']
            st.success("✅ Pipeline terbaik siap untuk deployment.")
            
            # Download pickle
            joblib.dump(pipeline_best, "pipeline_best.joblib")
            with open("pipeline_best.joblib", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="pipeline_best.joblib">Download Pipeline Pickle (.joblib)</a>', unsafe_allow_html=True)
        
        # Display metrics & results
        if 'df_results' in local_vars:
            st.subheader("Model Performance Results")
            st.dataframe(local_vars['df_results'].sort_values(by="F1-score", ascending=False))
            st.markdown(get_excel_download_link(local_vars['df_results']), unsafe_allow_html=True)
        
        if 'insights' in local_vars:
            st.subheader("Insights")
            for insight in local_vars['insights']:
                st.write(f"- {insight}")
        
        # Holdout prediction
        if uploaded_holdout and 'pipeline_best' in locals():
            df_holdout = pd.read_csv(uploaded_holdout)
            st.subheader("Holdout Dataset Preview")
            st.dataframe(df_holdout.head(100))
            
            st.subheader("Prediksi Holdout Dataset")
            y_pred_holdout = pipeline_best.predict(df_holdout)
            y_proba_holdout = pipeline_best.predict_proba(df_holdout)[:,1] if hasattr(pipeline_best, "predict_proba") else None
            df_out = df_holdout.copy()
            df_out["Prediction"] = y_pred_holdout
            if y_proba_holdout is not None:
                df_out["Probability"] = y_proba_holdout
            
            st.dataframe(df_out.head(100))
            val = to_excel(df_out)
            b64 = base64.b64encode(val).decode()
            st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="holdout_predictions.xlsx">Download Holdout Predictions</a>', unsafe_allow_html=True)
        
        # Generate PDF report
        figs_dict = {k:v for k,v in local_vars.items() if 'fig' in k}
        pdf_output = generate_pdf(
            df_results=local_vars.get('df_results', pd.DataFrame()),
            insights=local_vars.get('insights', []),
            figs_dict=figs_dict
        )
        b64_pdf = base64.b64encode(pdf_output.read()).decode()
        st.markdown(f'<a href="data:application/pdf;base64,{b64_pdf}" download="ML_Report.pdf">Download PDF Report</a>', unsafe_allow_html=True)
        
        st.success("✅ All-in-One Pipeline berhasil dijalankan!")

