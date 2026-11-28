
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Income Prediction",
    page_icon="💼",
    layout="wide"
)

# ================= DESAIN =================
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
.main .block-container {
    max-width: 1150px;
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(120deg, #142d4e, #2476a8);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 34px;
}
.hero p {
    color: #e3edf7;
    font-size: 16px;
}
.section {
    background: white;
    padding: 18px 22px;
    border-radius: 13px;
    border: 1px solid #e2e8f0;
    margin-bottom: 15px;
}
.result {
    background: white;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #dce5ef;
    text-align: center;
}
div.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 10px;
    background: #1769aa;
    color: white;
    font-weight: bold;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="hero">
    <h1>💼 Income Prediction System</h1>
    <p>Sistem Prediksi Kategori Pendapatan Individu
    Menggunakan Machine Learning - Random Forest</p>
    <p>Prediksi kategori pendapatan berdasarkan karakteristik
    individu, pekerjaan, dan kondisi keuangan.</p>
</div>
""", unsafe_allow_html=True)

# ================= MODEL =================
@st.cache_resource
def train_model():
    df = pd.read_csv("adult.csv", skipinitialspace=True)
    df.replace("?", np.nan, inplace=True)
    df = df.drop_duplicates().reset_index(drop=True)

    if "Education" in df.columns:
        df.drop(columns=["Education"], inplace=True)

    X = df.drop(columns=["Income"])
    y = df["Income"].map({"<=50K": 0, ">50K": 1})

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)
    return model

with st.spinner("Menyiapkan model prediksi..."):
    model = train_model()

# ================= FORM INPUT =================
st.markdown("## 📝 Data Individu")
st.caption("Lengkapi informasi berikut untuk melakukan prediksi pendapatan.")

with st.form("income_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Informasi Individu")

        age = st.number_input("Usia", 17, 90, 35)

        gender = st.selectbox(
            "Jenis Kelamin",
            ["Male", "Female"]
        )

        race = st.selectbox(
            "Ras",
            ["White", "Black", "Asian-Pac-Islander",
             "Amer-Indian-Eskimo", "Other"]
        )

        marital_status = st.selectbox(
            "Status Pernikahan",
            ["Married-civ-spouse", "Never-married", "Divorced",
             "Separated", "Widowed", "Married-spouse-absent",
             "Married-AF-spouse"]
        )

        relationship = st.selectbox(
            "Hubungan Keluarga",
            ["Husband", "Not-in-family", "Own-child",
             "Unmarried", "Wife", "Other-relative"]
        )

    with col2:
        st.markdown("### 💼 Informasi Pekerjaan")

        workclass = st.selectbox(
            "Jenis Pekerjaan",
            ["Private", "Self-emp-not-inc", "Self-emp-inc",
             "Federal-gov", "Local-gov", "State-gov",
             "Without-pay", "Never-worked"]
        )

        occupation = st.selectbox(
            "Bidang Pekerjaan",
            ["Exec-managerial", "Prof-specialty", "Craft-repair",
             "Adm-clerical", "Sales", "Other-service",
             "Machine-op-inspct", "Transport-moving",
             "Handlers-cleaners", "Farming-fishing",
             "Tech-support", "Protective-serv",
             "Priv-house-serv", "Armed-Forces"]
        )

        education_num = st.slider(
            "Lama Pendidikan (1–16)", 1, 16, 10
        )

        hours_per_week = st.slider(
            "Jam Kerja per Minggu", 1, 99, 40
        )

    st.markdown("---")
    st.markdown("### 💰 Informasi Keuangan")

    col3, col4 = st.columns(2)

    with col3:
        fnlwgt = st.number_input(
            "Final Weight",
            min_value=10000,
            max_value=1500000,
            value=180000
        )

        capital_gain = st.number_input(
            "Capital Gain",
            min_value=0,
            max_value=100000,
            value=0
        )

    with col4:
        capital_loss = st.number_input(
            "Capital Loss",
            min_value=0,
            max_value=10000,
            value=0
        )

        native_country = st.selectbox(
            "Negara Asal",
            ["United-States", "Mexico", "Philippines",
             "Germany", "Canada", "Puerto-Rico",
             "El-Salvador", "India", "Cuba", "England", "Other"]
        )

    submitted = st.form_submit_button(
        "🔍 PREDIKSI PENDAPATAN"
    )

# ================= HASIL PREDIKSI =================
if submitted:

    input_data = pd.DataFrame([{
        "Age": age,
        "Workclass": workclass,
        "Final Weight": fnlwgt,
        "EducationNum": education_num,
        "Marital Status": marital_status,
        "Occupation": occupation,
        "Relationship": relationship,
        "Race": race,
        "Gender": gender,
        "Capital Gain": capital_gain,
        "capital loss": capital_loss,
        "Hours per Week": hours_per_week,
        "Native Country": native_country
    }])

    hasil_pred = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    prob_persen = float(probabilities[hasil_pred]) * 100

    st.markdown("---")
    st.markdown("## 📊 Hasil Prediksi")

    if hasil_pred == 1:
        st.success("### Pendapatan diprediksi >50K")
    else:
        st.info("### Pendapatan diprediksi <=50K")

    st.markdown(
        f"<div class='result'>"
        f"<h2>{prob_persen:.2f}%</h2>"
        f"<p>Probabilitas untuk kategori hasil prediksi</p>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.progress(prob_persen / 100)

    st.caption(
        "Catatan: Hasil merupakan prediksi model berdasarkan pola "
        "pada dataset, bukan penentuan pendapatan aktual individu."
    )

# ================= FOOTER =================
st.markdown("---")
st.caption(
    "Income Prediction System | Random Forest Machine Learning"
)
