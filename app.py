
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="INCOME AI | Smart Prediction",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# PREMIUM UI DESIGN
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 5% 5%, rgba(67,97,238,.08), transparent 25%),
        radial-gradient(circle at 95% 20%, rgba(126,87,255,.07), transparent 25%),
        #f4f6fc;
}

.main .block-container {
    max-width: 1250px;
    padding: 1.5rem 2rem 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* HERO */

.hero {
    position: relative;
    overflow: hidden;
    padding: 32px 36px;
    border-radius: 22px;
    background: linear-gradient(115deg, #101b36 0%, #183b70 55%, #4935a5 100%);
    box-shadow: 0 14px 35px rgba(28,43,91,.16);
    margin-bottom: 20px;
}

.hero:before {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    background: rgba(255,255,255,.07);
    right: 7%;
    top: -160px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 190px;
    height: 190px;
    border-radius: 50%;
    border: 35px solid rgba(255,255,255,.055);
    right: -35px;
    bottom: -120px;
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero-badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 30px;
    background: rgba(255,255,255,.13);
    color: #d9e7ff;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 35px;
    font-weight: 900;
    letter-spacing: -1.2px;
    margin: 15px 0 7px;
}

.hero p {
    color: #d3def4 !important;
    font-size: 13px;
    line-height: 1.7;
    margin: 0;
    max-width: 650px;
}

/* TOP METRICS */

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 13px;
    margin-bottom: 25px;
}

.stat-card {
    background: rgba(255,255,255,.9);
    border: 1px solid #e7eaf3;
    padding: 15px 18px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(30,45,80,.035);
}

.stat-label {
    color: #8792a7;
    font-size: 11px;
    font-weight: 600;
}

.stat-value {
    color: #182c4c;
    font-size: 19px;
    font-weight: 800;
    margin-top: 5px;
}

.stat-icon {
    float: right;
    font-size: 20px;
}

/* SECTION */

.section-heading {
    color: #172b4d;
    font-size: 21px;
    font-weight: 800;
    margin: 5px 0 3px;
}

.section-subtitle {
    color: #8490a4;
    font-size: 12px;
    margin-bottom: 15px;
}

/* INPUT CARDS */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1px solid #e6eaf2;
    border-radius: 17px;
    box-shadow: 0 5px 20px rgba(30,45,80,.045);
    padding: 13px 15px;
}

.card-heading {
    color: #263c61;
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 3px;
}

.card-description {
    color: #95a0b2;
    font-size: 10px;
    margin-bottom: 13px;
}

div[data-testid="stWidgetLabel"] p {
    color: #44536b;
    font-size: 11px;
    font-weight: 600;
}

div[data-baseweb="select"] > div {
    border-radius: 9px;
    border-color: #e0e5ef;
    background: #fbfcff;
}

div[data-testid="stNumberInput"] input {
    border-radius: 9px;
    background: #fbfcff;
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(100deg, #315eea, #7049dc);
    color: white;
    border: none;
    height: 49px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: .5px;
    box-shadow: 0 7px 17px rgba(75,82,205,.22);
    transition: all .2s ease;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(100deg, #234bc8, #5836be);
    color: white;
    transform: translateY(-2px);
}

/* RESULT */

.result-card {
    background: linear-gradient(115deg, #101c38, #234a83 65%, #5140a4);
    padding: 28px;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 12px 30px rgba(30,48,100,.14);
    margin: 12px 0 18px;
}

.result-label {
    color: #b7cbf7;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
}

.result-category {
    color: #ffffff;
    font-size: 34px;
    font-weight: 900;
    margin: 10px 0;
}

.result-confidence {
    color: #ffffff;
    font-size: 27px;
    font-weight: 800;
}

.result-caption {
    color: #d1ddf5;
    font-size: 11px;
}

.prob-card {
    background: white;
    border: 1px solid #e6eaf2;
    border-radius: 15px;
    padding: 17px;
}

.prob-label {
    color: #7f8ca2;
    font-size: 11px;
    font-weight: 600;
}

.prob-value {
    color: #20375b;
    font-size: 24px;
    font-weight: 800;
    margin: 5px 0 10px;
}

.bar-bg {
    width: 100%;
    height: 8px;
    background: #edf0f7;
    border-radius: 20px;
    overflow: hidden;
}

.bar-fill {
    height: 8px;
    border-radius: 20px;
    background: linear-gradient(90deg, #3969ee, #8756e8);
}

.footer {
    text-align: center;
    padding-top: 22px;
    margin-top: 35px;
    border-top: 1px solid #e3e8f1;
    color: #8995a9;
    font-size: 10px;
}

@media (max-width: 700px) {
    .main .block-container {
        padding: 1rem;
    }

    .hero {
        padding: 24px;
    }

    .hero h1 {
        font-size: 26px;
    }

    .stats-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <div class="hero-content">
        <span class="hero-badge">✦ ARTIFICIAL INTELLIGENCE • MACHINE LEARNING</span>
        <h1>INCOME AI.</h1>
        <p>
            Smart Income Prediction Dashboard<br>
            Analisis dan prediksi kategori pendapatan individu
            menggunakan algoritma Random Forest.
        </p>
    </div>
</div>

<div class="stats-grid">
    <div class="stat-card">
        <span class="stat-icon">🤖</span>
        <div class="stat-label">ALGORITHM</div>
        <div class="stat-value">Random Forest</div>
    </div>
    <div class="stat-card">
        <span class="stat-icon">📊</span>
        <div class="stat-label">INPUT FEATURES</div>
        <div class="stat-value">13 Features</div>
    </div>
    <div class="stat-card">
        <span class="stat-icon">🎯</span>
        <div class="stat-label">PREDICTION TYPE</div>
        <div class="stat-value">Binary Classification</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# MODEL
# ==================================================

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
    with st.spinner("Memuat model AI..."):
        model = train_model()

except Exception as e:
    st.error(f"Gagal memuat model atau dataset: {e}")
    st.stop()


# ==================================================
# INPUT FORM
# ==================================================

st.markdown("""
<div class="section-heading">Prediction Workspace</div>
<div class="section-subtitle">
    Masukkan informasi individu untuk mendapatkan hasil prediksi pendapatan.
</div>
""", unsafe_allow_html=True)


with st.form("income_form"):

    col1, col2, col3 = st.columns(3, gap="medium")

    # PERSONAL INFORMATION
    with col1:

        with st.container(border=True):

            st.markdown("""
            <div class="card-heading">01 / Personal Profile</div>
            <div class="card-description">Informasi karakteristik individu</div>
            """, unsafe_allow_html=True)

            age = st.number_input(
                "Usia",
                min_value=17,
                max_value=90,
                value=35
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

            native_country = st.selectbox(
                "Negara Asal",
                ["United-States", "Mexico", "Philippines",
                 "Germany", "Canada", "Puerto-Rico",
                 "El-Salvador", "India", "Cuba", "England", "Other"]
            )

    # WORK INFORMATION
    with col2:

        with st.container(border=True):

            st.markdown("""
            <div class="card-heading">02 / Career & Education</div>
            <div class="card-description">Informasi pekerjaan dan pendidikan</div>
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
                "Lama Pendidikan",
                min_value=1,
                max_value=16,
                value=10
            )

            hours_per_week = st.slider(
                "Jam Kerja per Minggu",
                min_value=1,
                max_value=99,
                value=40
            )

    # FINANCIAL INFORMATION
    with col3:

        with st.container(border=True):

            st.markdown("""
            <div class="card-heading">03 / Financial Profile</div>
            <div class="card-description">Informasi kondisi keuangan</div>
            """, unsafe_allow_html=True)

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

            capital_loss = st.number_input(
                "Capital Loss",
                min_value=0,
                max_value=10000,
                value=0
            )

            st.markdown("---")

            st.caption(
                "Pastikan seluruh data sudah sesuai sebelum melakukan prediksi."
            )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "✦  GENERATE INCOME PREDICTION",
        use_container_width=True
    )


# ==================================================
# PREDICTION
# ==================================================

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

    st.markdown("""
    <div class="section-heading">Prediction Result</div>
    <div class="section-subtitle">
        Hasil analisis berdasarkan model Random Forest.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">✦ AI PREDICTION COMPLETE</div>
        <div class="result-category">{kategori}</div>
        <div class="result-caption">PREDICTED INCOME CATEGORY</div>
        <br>
        <div class="result-confidence">{confidence:.2f}%</div>
        <div class="result-caption">Probabilitas kategori hasil prediksi</div>
    </div>
    """, unsafe_allow_html=True)

    res1, res2 = st.columns(2, gap="medium")

    with res1:
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-label">INCOME ≤50K</div>
            <div class="prob-value">{prob_low:.2f}%</div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{prob_low:.2f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-label">INCOME >50K</div>
            <div class="prob-value">{prob_high:.2f}%</div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{prob_high:.2f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Lihat Data yang Dianalisis"):
        st.dataframe(input_data, use_container_width=True)

    st.info(
        "Catatan: Hasil merupakan estimasi berdasarkan pola dalam dataset, "
        "bukan pendapatan aktual individu."
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown("""
<div class="footer">
    <b>INCOME AI</b> &nbsp; | &nbsp; SMART PREDICTION SYSTEM<br>
    Machine Learning Project • Random Forest Classifier • Adult Income Dataset
</div>
""", unsafe_allow_html=True)
