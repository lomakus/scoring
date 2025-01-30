import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict
import numpy as np


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/score.png')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Scoring",
        page_icon=image,

    )

    st.write(
        """
        # Классификация заемщиков. Определяем кто вернет кредит, а кто нет.
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    train_df = open_data()
    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_input_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():

    age = st.sidebar.slider("Возраст", min_value=1, max_value=100, value=40,
                            step=1)
    rev = st.sidebar.slider("Кредитный баланс деленный на лимит кредита", min_value=0.0, max_value=5.0, value=0.2,
                            step=0.1)
    NumberOfTime30 = st.sidebar.slider("Количество просрочек более месяца и менее двух", min_value=1, max_value=100, value=0,
                            step=1)
    NumberOfTime60 = st.sidebar.slider("Количество просрочек от двух до трех месяцев", min_value=0, max_value=60, value=2,
                            step=1)
    NumberOfTimes90 = st.sidebar.slider("Количество просрочек от трех месяцев", min_value=0, max_value=60, value=2,
                            step=1)
    income = st.sidebar.slider("Месячный доход", min_value=0, max_value=12000, value=5400,
                            step=100)
    DebtRatio = st.sidebar.slider("Расход деленный на доход", min_value=0.0, max_value=1.0, value=0.2,
                            step=0.1)
    NCredits = st.sidebar.slider("Количество кредитов", min_value=0, max_value=60, value=4,
                            step=1)
    NumberOfDependents = st.sidebar.slider("Количество иждивенцев", min_value=0, max_value=20, value=2,
                            step=1)
    # sex = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    # embarked = st.sidebar.selectbox("Порт посадки", (
    # "Шербур-Октевиль", "Квинстаун", "Саутгемптон"))
    # pclass = st.sidebar.selectbox("Класс", ("Первый", "Второй", "Третий"))
    #
    # age = st.sidebar.slider("Возраст", min_value=1, max_value=80, value=20,
    #                         step=1)
    #
    # sib_sp = st.sidebar.slider(
    #     "Количетсво ваших братьев / сестер / супругов на борту",
    #     min_value=0, max_value=10, value=0, step=1)
    #
    # par_ch = st.sidebar.slider("Количетсво ваших детей / родителей на борту",
    #                            min_value=0, max_value=10, value=0, step=1)

    # translatetion = {
    #     "Мужской": "male",
    #     "Женский": "female",
    #     "Шербур-Октевиль": "C",
    #     "Квинстаун": "Q",
    #     "Саутгемптон": "S",
    #     "Первый": 1,
    #     "Второй": 2,
    #     "Третий": 3,
    # }

    data = {
        "RevolvingUtilizationOfUnsecuredLines": rev,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": NumberOfTime30,
        "DebtRatio": DebtRatio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": NCredits,
        "NumberOfTimes90DaysLate": NumberOfTimes90,
        "NumberOfTime60-89DaysPastDueNotWorse": NumberOfTime60,
        "NumberOfDependents": NumberOfDependents
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()