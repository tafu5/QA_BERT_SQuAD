import streamlit as st
import requests

###### PRINCIPAL #####
st.title("**BERT SQUAD MODEL**")
st.write("------------------------------")

st.header("**Predictions**")

API_URL = "http://localhost:9200/model/predict"

context = st.text_input("Context:")
question = st.text_input("Question:")

if st.button("Get Answer"):
    if context and question:
        # Prepare the payload for the POST request
        payload = {"context": context, "question": question}

        # Send the request to the FastAPI server
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                answer = response.text
                st.write("**Answer:**", answer)
            else:
                st.write("Error:", response.status_code)
        except requests.exceptions.RequestException as e:
            st.write("Error: Could not connect to the API. Make sure it's running.")

    else:
        st.write("Please provide both context and question.")

# Styling for the chat
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5dc;
    }
    </style>
    """, unsafe_allow_html=True)
