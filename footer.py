# footer.py

import streamlit as st

def add_footer():
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #000000;
        color: #ffffff;
        text-align: center;
        padding: 12px;
        font-size: 13px;
        z-index: 9999;
        box-shadow: 0 -1px 5px rgba(255,255,255,0.1);
    }
    </style>

    <div class="footer">
        © 2025 ASL Detection | Built using Streamlit | 📞 Contact: 03185447359
    </div>
    """, unsafe_allow_html=True)
