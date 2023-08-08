import streamlit as st

def code_page():
    # Add code for displaying the source code
    st.title("Source Code")


        # Display the source code of the model classes and functions
    st.subheader("Model Source Code")
    st.code(get_model_source_code(), language="python")

def get_model_source_code():
    # Paste the code directly here as a string
    source_code = """

SOURCE CODE HERE

    """
    return source_code