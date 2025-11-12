# =========================
# Imports
# =========================
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

# === LangChain terbaru compatible ===
try:
    # Versi LangChain terbaru (v1+)
    from langchain.prompts.prompt import PromptTemplate
except ModuleNotFoundError:
    # fallback untuk versi lama
    from langchain.prompts import PromptTemplate

import matplotlib.pyplot as plt
from io import BytesIO
import base64
from fpdf import FPDF
import joblib

# =========================
# Load Environment
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# Init LLM
# =========================
llm = ChatGroq(
    temperature=0.7,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)

# =========================
# Helper Functions
# =========================
def run_llm(prompt_template: str, data: pd.DataFrame):
    """
    Run LLM with PromptTemplate, compatible with LangChain v1+ (Pydantic strict)
    """
    if not isinstance(prompt_template, str) or len(prompt_template.strip()) == 0:
        raise ValueError("prompt_template harus string yang valid dan tidak kosong")
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data harus DataFrame yang tidak kosong")
    
    # Pastikan template memiliki {data} jika input_variables=["data"]
    if "{data}" not in prompt_template:
        prompt_template = "{data}\n" + prompt_template
    
    prompt = PromptTemplate(
        input_variables=["data"],  # harus sama dengan placeholder {data} di template
        template=prompt_template
    )

    formatted_prompt = prompt.format(data=data.to_string(index=False))
    return llm.invoke(formatted_prompt).content








