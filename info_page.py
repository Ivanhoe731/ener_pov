import streamlit as st

# Page: Info
def info_page():
    st.title("Info Page")
    st.header("About the Model")
    st.write("The model proposed on this page is an exploratory model, trying to explore how energy poverty under technological progress could be simulated. Agents are households with some disposable income and some characteristics, most notably 'inability', which dentoes wheather the households have an inability to keep home adequately warm. The agents are subject to prices of two fuels - yellow fuel and brown fuel and if activated, then some govenrmental policies / programs. ")

    st.header("How It Works")
    st.write("Explanation of how the model works goes here.")

    st.header("How to Use the Model")
    st.write("The model is operated through the listed widgets. Currently after every adjustment the model recalculates.")

