
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ================= CONFIG =================
st.set_page_config(
    page_title="Income Prediction",
    page_icon="💼",
    layout="wide"
)

# ================= STYLE =================
st.markdown("""
<style>
.stApp {
    background: #f5f7fb;
}

.main .block-container {
    max-width: 1180px;
    padding: 1rem 1.5rem 2rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.hero {
    background: linear-gradient(110deg, #142c4d, #2469a5);
    padding: 22px 30px;
    border-radius: 15px;
    color: white;
    margin-bottom: 18px;
}

.hero h1 {
    color: white;
    font-size: 27px;
    margin: 0 0 7px 0;
}

.hero p {
    color: #e1edfa;
    font-size: 13px;
    margin: 0;
}

.section-title {
    font-size: 20px;
    font-weight: 750;
    color: #193453;
    margin: 12px 0 2px;
}

.section-desc {
    color: #75849a;
    font-size: 12px;
    margin-bottom: 12px;
}

.card-title {
    background: white;
    padding: 12px 14px;
    border: 1px solid #e1e8f0;
    border-radius: 10px;
    color: #1c4269;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 10px;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.45rem;
}

div[data-testid="stWidgetLabel"] p {
    font-size: 12px;
    font-weight: 600;
    color: #34465e;
}

div[data-baseweb="select"] > div {
    border-radius: 8px;
}

div[data-testid="stNumberInput"] input {
    border-radius: 8px;
}

div[data-testid="stFormSubmitButton"] button {
    background: #1769ad;
    color: white;
    border: none;
    border-radius: 9px;
    height: 46px;
    font-weight: 700;
    font-size: 14px;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: #104c80;
    color: white;
}

.result-card {
    background: linear-gradient(120deg, #142c4d, #2469a5);
    color: white;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    margin: 12px 0;
}

.result-card h2 {
    color: white;
    margin: 7px 0;
    font-size: 27px;
}

.result-card p {
    color: #e1edfa;
    font-size: 13px;
}

.metric {
    background: white;
    border: 1px solid #e1e8f0;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 8px;
}

.metric-label {
    color: #75849a;
    font-size: 12px;
}

.metric-value {
    color: #193453;
    font-size: 22px;
    font-weight: 750;
}

.footer {
    text-align: center;
    color: #8492a6;
    font-size: 11px;
    border-top: 1px solid #e1e8f0;
    padding-top: 15px;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)


# ================= HEADER =================
st.markdown("""
<div class="hero">
    <h1>💼 Income Prediction System</h1>
    <p>
        Prediksi kategori pendapatan individu menggunakan
        Machine Learning dengan algoritma Random Forest.
    </p>
</div>
""", unsafe_allow_html=True)


# ================= TRAIN MODEL =================
@st.cache_resource
def train_model():
    df = pd.read_csv("adult.csv", skipinitialspace=True)

    df.replace("?", np.nan, inplace=True)
    df = df.drop_duplicates().reset_index(drop=True)

    if "Education" in df.columns:
        df.drop(columns=["Education"], inplace=True)

    X = df.drop(columns=["Income"])
    y = df["Income"].map({"<=50K": 0, ">50K": 1})

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

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


try:
    with st.spinner("Menyiapkan model prediksi..."):
        model = train_model()

except Exception as e:
    st.error(f"Gagal memuat model atau dataset: {e}")
    st.stop()


# ================= INPUT FORM =================
st.markdown(
    '<div class="section-title">Data Individu</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-desc">Lengkapi data berikut untuk melakukan prediksi pendapatan.</div>',
    unsafe_allow_html=True
)

with st.form("income_form"):

    col1, col2, col3 = st.columns(3, gap="medium")

    # PERSONAL
    with col1:
        st.markdown(
            '<div class="card-title">👤 Personal</div>',
            unsafe_allow_html=True
        )

        age = st.number_input(
            "Usia", 17, 90, 35
        )

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
            ["Married-civ-spouse", "Never-married",
             "Divorced", "Separated", "Widowed",
             "Married-spouse-absent", "Married-AF-spouse"]
        )

        relationship = st.selectbox(
            "Hubungan Keluarga",
            ["Husband", "Not-in-family", "Own-child",
             "Unmarried", "Wife", "Other-relative"]
        )

    # PEKERJAAN
    with col2:
        st.markdown(
            '<div class="card-title">💼 Pekerjaan & Pendidikan</div>',
            unsafe_allow_html=True
        )

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
            "Lama Pendidikan", 1, 16, 10
        )

        hours_per_week = st.slider(
            "Jam Kerja per Minggu", 1, 99, 40
        )

        native_country = st.selectbox(
            "Negara Asal",
            ["United-States", "Mexico", "Philippines",
             "Germany", "Canada", "Puerto-Rico",
             "El-Salvador", "India", "Cuba", "England", "Other"]
        )

    # KEUANGAN
    with col3:
        st.markdown(
            '<div class="card-title">💰 Informasi Keuangan</div>',
            unsafe_allow_html=True
        )

        fnlwgt = st.number_input(
            "Final Weight",
            10000, 1500000, 180000
        )

        capital_gain = st.number_input(
            "Capital Gain",
            0, 100000, 0
        )

        capital_loss = st.number_input(
            "Capital Loss",
            0, 10000, 0
        )

        st.markdown("---")
        st.caption(
            "Pastikan seluruh informasi telah diisi "
            "dengan benar sebelum melakukan prediksi."
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "🔍 PREDIKSI PENDAPATAN",
        use_container_width=True
    )


# ================= PREDICTION RESULT =================
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

    prob_low = float(probabilities[0]) * 100
    prob_high = float(probabilities[1]) * 100

    confidence = float(probabilities[hasil_pred]) * 100

    kategori = ">50K" if hasil_pred == 1 else "<=50K"

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Hasil Prediksi</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div class="result-card">
        <p>HASIL KLASIFIKASI RANDOM FOREST</p>
        <h2>💼 {kategori}</h2>
        <p>Probabilitas kategori hasil prediksi</p>
        <h2>{confidence:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

    result1, result2 = st.columns(2)

    with result1:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-label">Pendapatan ≤50K</div>
            <div class="metric-value">{prob_low:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(prob_low / 100)

    with result2:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-label">Pendapatan >50K</div>
            <div class="metric-value">{prob_high:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(prob_high / 100)

    with st.expander("Lihat Data yang Dianalisis"):
        st.dataframe(input_data, use_container_width=True)

    st.caption(
        "Catatan: Hasil merupakan estimasi model berdasarkan "
        "pola dalam dataset, bukan pendapatan aktual individu."
    )


# ================= FOOTER =================
st.markdown("""
<div class="footer">
    <b>INCOME PREDICTION SYSTEM</b><br>
    Machine Learning Project | Random Forest Classifier
</div>
""", unsafe_allow_html=True)
