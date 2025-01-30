from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load
import pandas as pd
import numpy as np

aim = ['SeriousDlqin2yrs']
numerical = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']


def split_data(df):

    y = df[aim[0]]
    X = df[numerical]

    return X, y


def open_data(path="https://raw.githubusercontent.com/evgpat/stepik_from_idea_to_mvp/main/datasets/credit_scoring.csv"):
    df = pd.read_csv(path)
    df = df[aim+numerical]

    return df


def preprocess_data(df: pd.DataFrame, test=True):

    feature = numerical[0]
    prep_df = df.copy()
    prep_df.loc[prep_df[feature] > 3, feature] = df[feature].mean()

    feature = numerical[1]
    prep_df.loc[prep_df[feature] < 19, feature] = df[feature].mean()
    prep_df.fillna(df[feature].mean(), inplace=True)

    for feature in numerical:
        prep_df[feature] = np.log1p(prep_df[feature])

    feature = numerical[4]
    prep_df.fillna(prep_df[feature].mean(), inplace=True)

    prep_df.dropna(inplace=True)

    scaler = MinMaxScaler()
    prep_df[numerical] = scaler.fit_transform(prep_df[numerical])

    if test:
        X_df, y_df = split_data(prep_df)
    else:
        X_df = prep_df

    # to_encode = ['Sex', 'Embarked']
    # for col in to_encode:
    #     dummy = pd.get_dummies(X_df[col], prefix=col)
    #     X_df = pd.concat([X_df, dummy], axis=1)
    #     X_df.drop(col, axis=1, inplace=True)

    if test:
        return X_df, y_df
    else:
        return X_df


def fit_and_save_model(X_df, y_df, path="data/model_weights.mw"):
    model = LogisticRegression()
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    accuracy = accuracy_score(test_prediction, y_df)
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    prediction = int(prediction_proba[1] > 0.06)
    # prediction = np.squeeze(prediction)


    encode_prediction_proba = {
        0: "Вероятность выдачи кредита",
        1: "Вероятность отказа"
    }

    encode_prediction = {
        0: "Ура!!! Вы можете взять кредт!",
        1: "Сожалеем, Вам не выдадут кредт!"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)