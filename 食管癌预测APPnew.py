import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =======================
# 0. 文件路径（确保和 app.py 同目录）
# =======================
MODEL_PATH = Path("RF.pkl")
ZPARAMS_PATH = Path("zscore_params.pkl")

# =======================
# 1. 加载模型与预处理参数（缓存）
# =======================
@st.cache_resource
def load_model_and_params():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")
    if not ZPARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing parameter file not found: {ZPARAMS_PATH.resolve()}"
        )

    model = joblib.load(MODEL_PATH)
    zparams = joblib.load(ZPARAMS_PATH)

    offset = float(zparams.get("offset", 0.0))
    mean = zparams["mean"]
    std = zparams["std"]
    return model, offset, mean, std

try:
    model, offset, mean, std = load_model_and_params()
except Exception as e:
    st.error(f"Failed to load the model or preprocessing parameters: {e}")
    st.stop()

# =======================
# 2. 页面标题
# =======================
st.title("ESCC Prediction System (RF Model)")
st.markdown(
    "### Enter the **raw metabolite values**. The system will automatically "
    "perform **log2 transformation and Z-score standardization using the "
    "training-cohort parameters** and predict the probability of esophageal "
    "squamous cell carcinoma (ESCC)."
)

with st.expander("Preprocessing Information", expanded=False):
    st.write(
        "- Log2 transformation: log2(x + offset)\n"
        "- Z-score standardization using the training-cohort mean and standard deviation: "
        "(log2-transformed value - mean) / standard deviation\n"
        "- The mean and standard deviation must not be recalculated during prediction "
        "to avoid data leakage."
    )
    st.write(f"Offset = {offset}")

# =======================
# 3. 输入特征（原始代谢物值）
# =======================
st.sidebar.header("Input Raw Metabolite Values")
st.sidebar.subheader("Metabolites")

Asparagine = st.sidebar.number_input("Asparagine", value=1.0, format="%.6f")
Choline = st.sidebar.number_input("Choline", value=1.0, format="%.6f")
Glutamate = st.sidebar.number_input("Glutamate", value=1.0, format="%.6f")
Sarcosine = st.sidebar.number_input("Sarcosine", value=1.0, format="%.6f")

feature_names = ["Asparagine", "Choline", "Glutamate", "Sarcosine"]

raw_df = pd.DataFrame([{
    "Asparagine": Asparagine,
    "Choline": Choline,
    "Glutamate": Glutamate,
    "Sarcosine": Sarcosine
}])

# =======================
# 4. 预测按钮
# =======================
if st.button("Generate Prediction"):

    # 检查 mean/std 是否包含 4 个代谢物
    missing = [c for c in feature_names if c not in mean.index or c not in std.index]
    if missing:
        st.error(
            "The zscore_params.pkl file does not contain the mean and standard "
            "deviation values for the following metabolites:\n"
            f"{missing}\n\n"
            "Please confirm that the feature names in the training dataset are "
            "identical to those used in this application."
        )
        st.stop()

    # 检查 log2 是否可计算
    min_allowed = -offset + 1e-12
    if (raw_df[feature_names] <= min_allowed).any().any():
        st.error(
            f"One or more input values are less than or equal to {-offset}, "
            "which prevents calculation of log2(x + offset).\n"
            f"Please ensure that each metabolite value satisfies x > {-offset} "
            f"(offset = {offset})."
        )
        st.stop()

    # 1) log2
    log2_df = np.log2(raw_df[feature_names].astype(float) + offset)

    # 2) Z-score（训练组参数）
    z_df = (log2_df - mean[feature_names]) / std[feature_names]

    # 3) 输入模型
    input_values = z_df[feature_names].values

    # 预测
    pred = int(model.predict(input_values)[0])
    probas = model.predict_proba(input_values)[0]  # [P(0), P(1)]

    st.markdown(
        f"### 🩺 Prediction Result: {'ESCC' if pred == 1 else 'Non-ESCC'}"
    )
    st.write(
        f"**Predicted probabilities:** "
        f"Non-ESCC (0) = {probas[0]:.4f}, "
        f"ESCC (1) = {probas[1]:.4f}"
    )

    # 展示预处理值（便于核对）
    with st.expander(
        "View Preprocessed Values (Raw / Log2 / Z-score)",
        expanded=True
    ):
        show_df = pd.concat(
            [
                raw_df[feature_names].rename(columns=lambda x: f"{x} (raw)"),
                log2_df.rename(columns=lambda x: f"{x} (log2)"),
                z_df.rename(columns=lambda x: f"{x} (zscore)")
            ],
            axis=1
        )
        st.dataframe(show_df)

    # 建议文本
    prob_escc = probas[1] * 100
    if pred == 1:
        st.info(
            f"The model generated an **ESCC-positive prediction**, with an "
            f"estimated ESCC probability of **{prob_escc:.2f}%**. "
            "Further evaluation using endoscopy, histopathology, and relevant "
            "clinical information is recommended."
        )
    else:
        st.info(
            f"The model generated a **Non-ESCC prediction**, with an estimated "
            f"ESCC probability of **{prob_escc:.2f}%**. "
            "The result should still be interpreted together with clinical risk "
            "factors and appropriate follow-up examinations."
        )

    # 可视化
    plt.figure(figsize=(6, 3))
    plt.barh( ["Non-ESCC (0)", "ESCC (1)"], [probas[0], probas[1]], color=["#2E86C1", "#E74C3C"] )
    plt.xlabel("Predicted probability")
    for i, v in enumerate(probas):
        plt.text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")
    plt.xlim(0, 1)
    plt.tight_layout()
    st.pyplot(plt)

