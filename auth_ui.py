# auth_ui.py

import streamlit as st
from auth import authenticate, signup

def login_signup_page():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        auth_mode = st.radio("Choose:", ["Login", "Sign Up"], horizontal=True)

        st.title("🔐 Sign Language Detection - Authentication")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if auth_mode == "Login":
            if st.button("Login"):
                if authenticate(username, password):
                    st.session_state.authenticated = True
                    st.success("✅ Logged in successfully!")
                    st.rerun()

                else:
                    st.error("❌ Invalid username or password")

        elif auth_mode == "Sign Up":
            if st.button("Sign Up"):
                if signup(username, password):
                    st.success("🎉 Account created! You can now log in.")
                else:
                    st.warning("⚠️ Username already exists!")

        st.stop()  # Stop app execution until login
 
 
def logout_user():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.rerun()
  # Refresh to show login screen