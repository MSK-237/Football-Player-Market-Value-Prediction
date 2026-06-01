import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.title("Futbolcu Piyasa Değeri Tahmini")

st.write(
    "Bu uygulama futbolcunun temel bilgileri ve performans istatistiklerine göre "
    "tahmini piyasa değerini hesaplar."
)

model = joblib.load("football_market_value_model.pkl")
model_columns = joblib.load("model_columns.pkl")


st.sidebar.header("Oyuncu Bilgileri")

height_in_cm = st.sidebar.number_input("Boy (cm)", 150, 220, 180)
international_caps = st.sidebar.number_input("Milli Maç Sayısı", 0, 250, 0)
international_goals = st.sidebar.number_input("Milli Gol Sayısı", 0, 150, 0)

goals = st.sidebar.number_input("Toplam Gol", 0, 500, 0)
assists = st.sidebar.number_input("Toplam Asist", 0, 500, 0)
minutes_played = st.sidebar.number_input("Oynanan Dakika", 0, 100000, 1000)
yellow_cards = st.sidebar.number_input("Sarı Kart", 0, 300, 0)
red_cards = st.sidebar.number_input("Kırmızı Kart", 0, 50, 0)

position = st.sidebar.selectbox(
    "Pozisyon",
    ["Attack", "Midfield", "Defender", "Goalkeeper"]
)

sub_position = st.sidebar.selectbox(
    "Alt Pozisyon",
    [
        "Centre-Forward",
        "Left Winger",
        "Right Winger",
        "Attacking Midfield",
        "Central Midfield",
        "Defensive Midfield",
        "Centre-Back",
        "Left-Back",
        "Right-Back",
        "Goalkeeper"
    ]
)

foot = st.sidebar.selectbox(
    "Kullandığı Ayak",
    ["right", "left", "both"]
)

competition = st.sidebar.selectbox(
    "Lig / Organizasyon",
    ["GB1", "ES1", "IT1", "L1", "FR1", "TR1", "NL1", "PO1", "BE1", "Unknown"]
)


input_data = pd.DataFrame({
    "height_in_cm": [height_in_cm],
    "international_caps": [international_caps],
    "international_goals": [international_goals],
    "goals": [goals],
    "assists": [assists],
    "minutes_played": [minutes_played],
    "yellow_cards": [yellow_cards],
    "red_cards": [red_cards],
    "position": [position],
    "sub_position": [sub_position],
    "foot": [foot],
    "current_club_domestic_competition_id": [competition]
})


input_data = pd.get_dummies(input_data)

input_data = input_data.reindex(
    columns=model_columns,
    fill_value=0
)


if st.button("Piyasa Değerini Tahmin Et"):

    tahmin_log = model.predict(input_data)

    tahmin_euro = np.expm1(tahmin_log)[0]

    st.success(
        f"Tahmini Piyasa Değeri: €{tahmin_euro:,.0f}"
    )

    st.write(
        "Bu tahmin makine öğrenmesi modeli tarafından üretilmiştir. "
        "Gerçek piyasa değeri sözleşme durumu, sakatlık geçmişi, oyuncu popülaritesi "
        "ve transfer piyasası koşullarına göre değişebilir."
    )