import streamlit as st
import pandas as pd
import datetime
import pickle
import matplotlib.pyplot as plt

with open('./rain_prediction_model.pkl', 'rb') as file:
    model, columns = pickle.load(file)

st.set_page_config(page_title="Deszcz w Australii", page_icon="🌧️", layout="wide")
st.markdown(
    """
    <style>
    .toolbar {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        text-align: left;
        font-size: 24px;
        line-height: 60px;
        padding: 0 20px;
        margin-bottom: 20px;
        height: 60px;
    }
    .scrollable {
        max-height: 300px;
        overflow-y: scroll;
        border: 1px solid #ccc;
        padding: 10px;
        background-color: #f9f9f9;
    }
    </style>
    <div class="toolbar">Deszcz w Australii Projekt SUML</div>
    """,
    unsafe_allow_html=True
)

st.subheader("Wybierz parametry")

regions = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
           'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
           'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
           'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
           'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
           'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
           'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
           'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
           'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
           'AliceSprings', 'Darwin', 'Katherine', 'Uluru']

selected_region = st.selectbox("Wybierz region", regions)

start_date = st.date_input("Wybierz początkową datę", datetime.date.today(), min_value=datetime.date.today())

end_date = st.date_input("Wybierz końcową datę", start_date, min_value=start_date)

if st.button("Submit"):
    date_range = pd.date_range(start_date, end_date)

    predictions = []
    dates = []
    for single_date in date_range:
        year = single_date.year
        month = single_date.month
        day = single_date.day

        sample_data = {
            'Year': year,
            'Month': month,
            'Day': day
        }

        for region in regions:
            sample_data[f'Location_{region}'] = 1 if region == selected_region else 0

        input_data = pd.DataFrame([sample_data])
        input_data = input_data[columns]
        prediction_proba = model.predict_proba(input_data)[0, 1]
        predictions.append(prediction_proba * 100)
        dates.append(single_date)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Szansa na deszcz w poszczególnych dniach")
        fig, ax = plt.subplots(figsize=(20, 3))
        ax.bar([date.strftime('%Y-%m-%d') for date in dates], predictions, color='blue', alpha=0.7)
        ax.set_title('Szansa na deszcz w poszczególnych dniach')
        ax.set_xlabel('Data')
        ax.set_ylabel('Szansa na deszcz (%)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("Prognoza szansy na deszcz")
        scrollable_content = "<div class='scrollable'>"
        for date, proba in zip(dates, predictions):
            scrollable_content += f"<div class='item'>{date.strftime('%Y-%m-%d')}: {proba:.2f}% szansy na deszcz</div>"
        scrollable_content += "</div>"
        st.markdown(scrollable_content, unsafe_allow_html=True)