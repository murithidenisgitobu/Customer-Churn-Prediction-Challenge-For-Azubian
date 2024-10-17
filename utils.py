# Streamlit Multipage App with Logout on Every Page

# utils.py (New file for shared functionality)
import streamlit as st

def add_logout_button():
    _, _, logout_col = st.columns([1, 1, 1])
    with logout_col:
        if st.button("Logout", key="logout"):
            st.session_state['authentication_status'] = False
            st.session_state['logged_in'] = False
            st.rerun()