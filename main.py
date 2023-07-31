import streamlit as st
from abm import model_page
from info_page import info_page
from code_page import code_page

def main():
    st.set_page_config(layout="wide", page_title="ABM App")

    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("", ("Introduction", "Model", "Info", "Code"), index=0)

    if selected_page == "Model":
        model_page()
    elif selected_page == "Info":
        info_page()
    elif selected_page == "Code":
        code_page()
    else:
        st.title("ABM - Energy Poverty & Technological Progress")

        st.header("Introduction")
        st.write("Welcome to the ABM - Energy Poverty & Technological Progress app. This app showcases an Agent-Based Model for analyzing energy poverty and technological progress. It consists of three main pages: Model, Info, and Code. Select a page from the sidebar to explore different aspects of the app. The application was developed as part of diploma thesis by Ivan Novák.")

        st.header("Contents")

        st.markdown("""
        <style>
        .css-1r6slb0 esravye1 {
            background-color: white;
            border-radius: 15px;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Model")
            st.write("The Model page allows you to interact with the Agent-Based Model. You can configure parameters, run simulations, and visualize the results.")

        with col2:
            st.subheader("Info")
            st.write("The Info page provides additional information about the app, including how it works, how to use it, things to notice, and related models.")

        with col3:
            st.subheader("Code")
            st.write("The Code page contains the source code for the ABM app. You can explore the code and understand the implementation details.")

if __name__ == "__main__":
    main()
