# game.py
import streamlit as st
import random
import os
from PIL import Image

def run_game(data_dir="data/test"):
    st.title("🎮 Guess the Sign - ASL Game")
    st.markdown("Test your ASL knowledge! A random sign will be shown, and you must guess the correct label.")

    # Load class names
    class_labels = sorted(os.listdir(data_dir))

    # Initialize session state
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "round" not in st.session_state:
        st.session_state.round = 1
    if "game_img_path" not in st.session_state:
        st.session_state.game_img_path = None
    if "correct_label" not in st.session_state:
        st.session_state.correct_label = None
    if "options" not in st.session_state:
        st.session_state.options = []

    # On new round, select new image & options
    if st.session_state.game_img_path is None:
        correct_label = random.choice(class_labels)
        img_dir = os.path.join(data_dir, correct_label)
        img_file = random.choice(os.listdir(img_dir))
        img_path = os.path.join(img_dir, img_file)

        st.session_state.correct_label = correct_label
        st.session_state.game_img_path = img_path

        # Pick 3 random options and insert correct one if missing
        options = random.sample(class_labels, 3)
        if correct_label not in options:
            options[random.randint(0, 2)] = correct_label
        random.shuffle(options)

        st.session_state.options = options

    # Show the image
    st.image(st.session_state.game_img_path, caption=f"Round {st.session_state.round}", use_container_width=True)

    # User guess with unique key
    guess = st.radio("What sign is this?", st.session_state.options, key=f"radio_{st.session_state.round}")

    # Submit answer button with unique key
    if st.button("Submit Answer", key=f"submit_{st.session_state.round}"):
        if guess == st.session_state.correct_label:
            st.success("✅ Correct!")
            st.session_state.score += 1
        else:
            st.error(f"❌ Wrong! It was '{st.session_state.correct_label}'")

        # Reset for next round
        st.session_state.round += 1
        st.session_state.game_img_path = None
        st.session_state.correct_label = None
        st.session_state.options = []

        # Re-render page with updated state
        st.rerun()

    # Sidebar score
    st.sidebar.success(f"🏆 Score: {st.session_state.score}")
    st.sidebar.info(f"🕐 Round: {st.session_state.round}")
