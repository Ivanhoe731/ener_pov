import streamlit as st

# Page: Info
def info_page():
    st.title("Info Page")
    st.header("About the Model")
    st.write("The model proposed on this page is an exploratory model, trying to explore how energy poverty under technological progress could be simulated. Agents are households with some disposable income and some characteristics, most notably 'inability', which dentoes wheather the households have an inability to keep home adequately warm. The agents are subject to prices of two fuels - yellow fuel and brown fuel and if activated, then some govenrmental policies / programs. ")
    st.write("The code itself is explained directly in the source code on the Code page.")

    st.header("How It Works")
    st.write(" 1. The model firts creates number of agents and assigns to them disposable income based on gini coefficient and min + median disposable income variables. \n" 
             + "2. Agents are assigned the 'inability' status, bsaed on their disposable income and treshold picked for the inability level")

    st.header("How to Use the Model")
    st.write("The model is operated through the listed widgets. Currently after every adjustment the model is recalculated. The mdoel is fundamentally unitless, making it great for customization of observed periods and metrics universaly")



