import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv('train.csv')

# Data preprocessing
data.drop(['Cabin', 'Ticket', 'Name', 'Embarked'], axis=1, inplace=True)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

data['Age'] = data[['Age', 'Pclass']].apply(impute_age, axis=1)
data.dropna(inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

X = data.drop(['Survived', 'PassengerId'], axis=1)
y = data['Survived']

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Streamlit app layout
st.title("Titanic Survival Prediction")
st.sidebar.header("Passenger Information")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 100, 30)
sibsp = st.sidebar.slider("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.slider("Number of Parents/Children Aboard", 0, 10, 0)

input_data = np.array([[pclass, 0 if sex == "male" else 1, age, sibsp, parch]])
input_df = pd.DataFrame(input_data, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("The passenger is predicted to survive.")
    else:
        st.error("The passenger is predicted not to survive.")
