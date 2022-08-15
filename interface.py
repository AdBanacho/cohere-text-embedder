import streamlit as st
from co import CoHere
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.header("Co:here Text Classifier vs Random Forest")

api_key = st.text_input("API Key:", type="password")

cohere = CoHere(api_key)
cohere.list_of_examples(50)
cohere.embed(1000)

forest = RandomForestClassifier(max_depth=10, random_state=0)

col1, col2 = st.columns(2)

if col1.button("Classify"):
    forest.fit(cohere.X_train_embeded, cohere.y_train)
    predict = forest.predict_proba(np.array(cohere.X_test_embeded[0]).reshape(1, -1))[0]
    here = cohere.classify([cohere.X_test[0]])[0]
    col2.success(f"Correct prediction: {cohere.y_test[0]}")

    col1, col2 = st.columns(2)
    col1.header("Co:here classify")
    for con in here.confidence:
        col1.write(f"{con.label}: {np.round(con.confidence*100, 2)}%")
        col1.progress(con.confidence)

    col2.header("Random Forest")
    for con, pred in zip(here.confidence, predict):
        col2.write(f"{con.label}: {np.round(pred*100, 2)}%")
        col2.progress(pred)
