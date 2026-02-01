# eda.py

import streamlit as st
import pandas as pd
import os
from PIL import Image
import plotly.express as px

def run_eda(data_dir="data/train"):
    st.title("📊 EDA Dashboard ")

    st.markdown("### 🔍 Select an EDA Feature to Display")
    col1, col2, col3 = st.columns(3)
    selected = None

    with col1:
        if st.button("📊 Class Distribution"):
            selected = "class_dist"
        if st.button("🖼️ Sample Images"):
            selected = "samples"

    with col2:
        if st.button("🥇 Top & Least Classes"):
            selected = "top_bottom"
        if st.button("📐 Image Size Distribution"):
            selected = "size_dist"

    with col3:
        if st.button("📏 Aspect Ratio Distribution"):
            selected = "aspect_ratio"
       

    # --- Load class counts once ---
    class_counts = {folder: len(os.listdir(os.path.join(data_dir, folder)))
                    for folder in os.listdir(data_dir)}
    df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Count'])
    df = df.sort_values(by="Count", ascending=False)

    # === 1. Class Distribution ===
    if selected == "class_dist":
        st.subheader("📊 Class Distribution")
        fig = px.bar(df, x=df.index, y="Count", color="Count", title="Class Count per Label")
        st.plotly_chart(fig, use_container_width=True)

    # === 2. Top & Bottom Classes ===
    elif selected == "top_bottom":
        st.subheader("🥇 Top & Least Populated Classes")
        col1, col2 = st.columns(2)

        top_5 = df.head(5)
        bottom_5 = df.tail(5)

        with col1:
            st.markdown("##### 🔝 Top 5 Classes")
            fig = px.bar(top_5, x=top_5.index, y="Count", color="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("##### 🧮 Least 5 Classes")
            fig = px.bar(bottom_5, x=bottom_5.index, y="Count", color="Count")
            st.plotly_chart(fig, use_container_width=True)

    # === 3. Sample Images ===
    elif selected == "samples":
        st.subheader("🖼️ Sample Images per Class")
        cols = st.columns(5)
        for i, cls in enumerate(df.head(5).index):
            image_path = os.path.join(data_dir, cls, os.listdir(os.path.join(data_dir, cls))[0])
            image = Image.open(image_path)
            cols[i % 5].image(image, caption=cls, use_container_width=True)

    # === 4. Image Size Distribution ===
    elif selected == "size_dist":
        st.subheader("📐 Image Size Distribution")
        all_sizes = []
        for cls in os.listdir(data_dir):
            for img_file in os.listdir(os.path.join(data_dir, cls))[:50]:
                path = os.path.join(data_dir, cls, img_file)
                try:
                    img = Image.open(path)
                    all_sizes.append(img.size)
                except:
                    continue
        size_df = pd.DataFrame(all_sizes, columns=["Width", "Height"])
        fig = px.scatter(size_df, x="Width", y="Height", title="Image Width vs Height")
        st.plotly_chart(fig, use_container_width=True)

    # === 5. Aspect Ratio Distribution ===
    elif selected == "aspect_ratio":
        st.subheader("📏 Aspect Ratio Distribution")
        aspect_ratios = []
        for cls in os.listdir(data_dir):
            for img_file in os.listdir(os.path.join(data_dir, cls))[:50]:
                path = os.path.join(data_dir, cls, img_file)
                try:
                    img = Image.open(path)
                    w, h = img.size
                    ratio = round(w / h, 2) if h != 0 else 0
                    aspect_ratios.append(ratio)
                except:
                    continue
        ratio_df = pd.DataFrame(aspect_ratios, columns=["Aspect Ratio"])
        fig = px.histogram(ratio_df, x="Aspect Ratio", nbins=30, title="Aspect Ratio Histogram")
        st.plotly_chart(fig, use_container_width=True)
      
  
 
    if not selected:
        st.info("👆 Click a button above to explore the dataset visually.")

