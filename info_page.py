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

    st.header("Things to Notice")
    st.write("Important things to notice about the model go here.")

    st.header("Things to Try")
    st.write("Suggested experiments or variations to try with the model go here.")

    st.header("Extending the Model")
    st.write("Tips or guidance on how to extend or modify the model go here.")

    st.header("Related Models")
    st.write("Information about related models or resources go here.")

    st.header("Credits and References")
    st.write("Credits for the model and references go here.")

