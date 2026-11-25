import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Prediksi Kategori Pendapatan", page_icon="💼")

st.title("💼 Prediksi Kategori Pendapatan (Income)")
st.write("Aplikasi klasifikasi Machine Learning (**Random Forest**) untuk memprediksi kategori pendapatan **<=50K** atau **>50K**.")

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
            n_estimators=50, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X, y)
    return model

with st.spinner("Menyiapkan model Random Forest..."):
    model = train_model()

st.subheader("Masukkan Data Individu")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia (Age)", min_value=17, max_value=90, value=35)
    workclass = st.selectbox("Jenis Pekerjaan (Workclass)", [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ])
    fnlwgt = st.number_input("Final Weight", min_value=10000, max_value=1500000, value=180000)
    education_num = st.slider("Lama Pendidikan / EducationNum (1-16)", min_value=1, max_value=16, value=10)
    marital_status = st.selectbox("Status Pernikahan (Marital Status)", [
        "Married-civ-spouse", "Never-married", "Divorced",
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])
    occupation = st.selectbox("Bidang Pekerjaan (Occupation)", [
        "Exec-managerial", "Prof-specialty", "Craft-repair", "Adm-clerical",
        "Sales", "Other-service", "Machine-op-inspct", "Transport-moving",
        "Handlers-cleaners", "Farming-fishing", "Tech-support",
        "Protective-serv", "Priv-house-serv", "Armed-Forces"
    ])

with col2:
    relationship = st.selectbox("Hubungan Keluarga (Relationship)", [
        "Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"
    ])
    race = st.selectbox("Ras (Race)", [
        "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
    ])
    gender = st.selectbox("Jenis Kelamin (Gender)", ["Male", "Female"])
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=10000, value=0)
    hours_per_week = st.slider("Jam Kerja per Minggu (Hours per Week)", min_value=1, max_value=99, value=40)
    native_country = st.selectbox("Negara Asal (Native Country)", [
        "United-States", "Mexico", "Philippines", "Germany", "Canada",
        "Puerto-Rico", "El-Salvador", "India", "Cuba", "England", "Other"
    ])

if st.button("Prediksi Pendapatan", type="primary"):
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

    hasil_pred = int(model.predict(input_data).item(0))
    prob_array = model.predict_proba(input_data)
    prob_persen = float(prob_array.max()) * 100

    st.divider()
    if hasil_pred == 1:
        st.success(f"### Hasil Prediksi: Pendapatan **>50K** (Probabilitas: {prob_persen:.2f}%)")
    else:
        st.info(f"### Hasil Prediksi: Pendapatan **<=50K** (Probabilitas: {prob_persen:.2f}%)")

