import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Kategori Pendapatan", page_icon="💼")

st.title("💼 Prediksi Kategori Pendapatan (Income)")
st.write("Aplikasi klasifikasi Machine Learning (**Random Forest**) untuk memprediksi kategori pendapatan **<=50K** atau **>50K**.")

@st.cache_resource
def load_model():
    return joblib.load("best_income_pipeline.joblib")

model = load_model()

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

    prediksi = model.predict(input_data)
    probabilitas = model.predict_proba(input_data)

    st.divider()
    if prediksi == 1:
        st.success(f"### Hasil Prediksi: Pendapatan **>50K** (Probabilitas: {probabilitas*100:.2f}%)")
    else:
        st.info(f"### Hasil Prediksi: Pendapatan **<=50K** (Probabilitas: {probabilitas*100:.2f}%)")
