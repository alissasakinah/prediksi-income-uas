
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Income AI | Prediction System",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= CUSTOM DESIGN =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f3f6fc;
}

.main .block-container {
    max-width: 1250px;
    padding: 2rem 2rem 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    background: linear-gradient(120deg, #101b35, #173e70 65%, #176db1);
    padding: 42px;
    border-radius: 24px;
    color: white;
    position: relative;
    overflow: hidden;
    margin-bottom: 28px;
    box-shadow: 0 12px 35px rgba(20, 48, 90, .15);
}

.hero:after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    background: rgba(255,255,255,.07);
    right: -60px;
    top: -100px;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,.13);
    padding: 7px 14px;
    border-radius: 30px;
    font-size: 12px;
    letter-spacing: 1px;
    font-weight: 700;
    color: #cce7ff;
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin: 20px 0 10px;
}

.hero p {
    color: #d4e4f7;
    font-size: 15px;
    line-height: 1.7;
    max-width: 650px;
}

.section-title {
    color: #142746;
    font-size: 23px;
    font-weight: 800;
    margin: 20px 0 5px;
}

.section-desc {
    color: #74839a;
    font-size: 13px;
    margin-bottom: 20px;
}

.info-card {
    background: white;
    padding: 19px 22px;
    border: 1px solid #e4eaf3;
    border-radius: 16px;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(30,55,90,.035);
}

.card-heading {
    font-size: 16px;
    font-weight: 800;
    color: #193456;
    margin-bottom: 5px;
}

.card-sub {
    color: #8492a6;
    font-size: 12px;
}

div[data-baseweb="select"] > div {
    border-radius: 10px;
    border-color: #dce4ef;
    background: #ffffff;
}

div[data-testid="stNumberInput"] input {
    border-radius: 10px;
}

div[data-testid="stSlider"] {
    padding-top: 8px;
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(100deg, #1769b0, #2454a5);
    color: white;
    border: none;
    border-radius: 12px;
    height: 52px;
    font-size: 15px;
    font-weight: 800;
    box-shadow: 0 6px 18px rgba(25,91,165,.2);
    transition: .2s;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: #123f78;
    color: white;
    transform: translateY(-2px);
}

.result-card {
    background: linear-gradient(125deg, #102440, #1d4777);
    border-radius: 20px;
    padding: 30px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 25px rgba(20,50,90,.12);
    margin: 12px 0 20px;
}

.result-card h2 {
    color: white;
    font-size: 32px;
    margin: 10px 0;
}

.result-card p {
    color: #d2e4f8;
    font-size: 14px;
}

.result-label {
    color: #a9d5ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.metric-box {
    background: white;
    border: 1px solid #e4eaf3;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}

.metric-name {
    color: #8492a6;
    font-size: 12px;
}

.metric-value {
    color: #193456;
    font-size: 23px;
    font-weight: 800;
    margin-top: 5px;
}

.footer {
    text-align: center;
    color: #94a2b5;
    font-size: 12px;
    padding: 25px 0 5px;
    border-top: 1px solid #e2e8f0;
    margin-top: 40px;
}

.stProgress > div > div > div > div {
    background: #2777c8;
}
</style>
""", unsafe_allow_html=True)


# ================= HEADER =================
st.markdown("""
<div class="hero">
    <span class="hero-tag">✦ MACHINE LEARNING APPLICATION</span>
    <h1>Income Prediction System</h1>
    <p>
        Sistem prediksi kategori pendapatan individu berbasis
        Artificial Intelligence menggunakan algoritma
        <b>Random Forest Classifier</b>.
        Masukkan karakteristik individu untuk memperoleh hasil prediksi.
    </p>
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


try:
    with st.spinner("✦ Menyiapkan sistem prediksi..."):
        model = train_model()
except Exception as e:
    st.error(f"Gagal memuat model atau dataset: {e}")
    st.stop()


# ================= FORM =================
st.markdown(
    '<div class="section-title">Input Data Individu</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-desc">Lengkapi seluruh informasi untuk memulai analisis prediksi.</div>',
    unsafe_allow_html=True
)

with st.form("income_form"):

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-heading">👤 Informasi Personal</div>
            <div class="card-sub">Karakteristik dasar individu</div>
        </div>
        """, unsafe_allow_html=True)

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
            ["Married-civ-spouse", "Never-married",
             "Divorced", "Separated", "Widowed",
             "Married-spouse-absent", "Married-AF-spouse"]
        )

        relationship = st.selectbox(
            "Hubungan Keluarga",
            ["Husband", "Not-in-family", "Own-child",
             "Unmarried", "Wife", "Other-relative"]
        )

    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="card-heading">💼 Informasi Pekerjaan</div>
            <div class="card-sub">Pendidikan dan aktivitas pekerjaan</div>
        </div>
        """, unsafe_allow_html=True)

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
            "Lama Pendidikan (EducationNum)",
            1, 16, 10
        )

        hours_per_week = st.slider(
            "Jam Kerja per Minggu",
            1, 99, 40
        )

    st.markdown("""
    <div class="info-card">
        <div class="card-heading">💰 Informasi Keuangan</div>
        <div class="card-sub">Informasi bobot sampel dan kondisi keuangan individu</div>
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        fnlwgt = st.number_input(
            "Final Weight",
            10000, 1500000, 180000
        )

        capital_gain = st.number_input(
            "Capital Gain",
            0, 100000, 0
        )

    with col4:
        capital_loss = st.number_input(
            "Capital Loss",
            0, 10000, 0
        )

        native_country = st.selectbox(
            "Negara Asal",
            ["United-States", "Mexico", "Philippines",
             "Germany", "Canada", "Puerto-Rico",
             "El-Salvador", "India", "Cuba", "England", "Other"]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "✦ ANALISIS & PREDIKSI PENDAPATAN",
        use_container_width=True
    )


# ================= PREDICTION =================
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

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Hasil Analisis</div>',
        unsafe_allow_html=True
    )

    kategori = ">50K" if hasil_pred == 1 else "<=50K"

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">PREDICTION RESULT</div>
        <h2>💼 {kategori}</h2>
        <p>Kategori pendapatan yang diprediksi oleh model Random Forest</p>
        <h2>{confidence:.2f}%</h2>
        <p>Probabilitas kategori hasil prediksi</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-name">Pendapatan ≤50K</div>
            <div class="metric-value">{prob_low:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(prob_low / 100)

    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-name">Pendapatan >50K</div>
            <div class="metric-value">{prob_high:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(prob_high / 100)

    with st.expander("Lihat data yang dianalisis"):
        st.dataframe(input_data, use_container_width=True)

    st.caption(
        "Catatan: Hasil merupakan estimasi model berdasarkan pola "
        "dalam dataset, bukan pendapatan aktual individu."
    )


# ================= FOOTER =================
st.markdown("""
<div class="footer">
    <b>INCOME PREDICTION SYSTEM</b><br>
    Machine Learning Project | Random Forest Classifier<br>
    Adult Income Dataset
</div>
""", unsafe_allow_html=True)
