"""
Handwriting Collection App
===========================
A Streamlit web app that collects handwriting samples (A–Z) from users
and saves them as a labeled image dataset for ML training.

HOW TO INSTALL DEPENDENCIES:
    pip install streamlit streamlit-drawable-canvas opencv-python numpy pillow

HOW TO RUN:
    streamlit run app.py
"""

import os
import uuid
import numpy as np
import streamlit as st
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
CANVAS_SIZE = 200          # Drawing canvas width & height (pixels)
SAVE_SIZE   = 64           # Final saved image size (64x64 grayscale)
DATASET_DIR = "dataset"    # Root folder for all user data
LETTERS     = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ──────────────────────────────────────────────
# SESSION STATE INITIALISATION
# Session state persists values across reruns
# ──────────────────────────────────────────────
if "user_id" not in st.session_state:
    # Give each new visitor a unique folder name
    st.session_state.user_id = "user_" + uuid.uuid4().hex[:6]

if "letter_index" not in st.session_state:
    st.session_state.letter_index = 0   # Start at letter A (index 0)

if "finished" not in st.session_state:
    st.session_state.finished = False   # Tracks whether all 26 letters are done

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0     # Changing this key resets the canvas widget

# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def get_user_folder() -> str:
    """Return (and create if needed) the folder for the current user's images."""
    folder = os.path.join(DATASET_DIR, st.session_state.user_id)
    os.makedirs(folder, exist_ok=True)
    return folder


def save_image(canvas_image_data: np.ndarray, letter: str) -> str:
    """
    Process and save a single letter image.

    Steps:
      1. Convert RGBA canvas data → grayscale
      2. Invert colours (canvas is white background, we want black background)
      3. Resize to SAVE_SIZE × SAVE_SIZE
      4. Save as PNG in the user's folder

    Returns the full path of the saved file.
    """
    # canvas_image_data is RGBA (height, width, 4)
    img_rgba = Image.fromarray(canvas_image_data.astype("uint8"), "RGBA")

    # Convert to grayscale using PIL
    img_gray = img_rgba.convert("L")

    # Convert to numpy for OpenCV processing
    img_np = np.array(img_gray)

    # Invert: drawing strokes are dark on a white canvas;
    # ML datasets typically expect white strokes on black background.
    img_inverted = cv2.bitwise_not(img_np)

    # Resize to 64×64
    img_resized = cv2.resize(img_inverted, (SAVE_SIZE, SAVE_SIZE),
                             interpolation=cv2.INTER_AREA)

    # Build the file path, e.g. dataset/user_abc123/A.png
    save_path = os.path.join(get_user_folder(), f"{letter}.png")
    cv2.imwrite(save_path, img_resized)

    return save_path


def clear_canvas():
    """Reset the canvas by incrementing its key (forces Streamlit to recreate it)."""
    st.session_state.canvas_key += 1


def next_letter(canvas_result):
    """
    Save the current drawing, then advance to the next letter.
    If all 26 letters are done, mark the session as finished.
    """
    current_letter = LETTERS[st.session_state.letter_index]

    # Only save if the user actually drew something
    if canvas_result.image_data is not None:
        save_image(canvas_result.image_data, current_letter)

    # Move to the next letter
    st.session_state.letter_index += 1

    # Check if all letters are complete
    if st.session_state.letter_index >= len(LETTERS):
        st.session_state.finished = True

    # Clear the canvas for the next letter
    clear_canvas()


# ──────────────────────────────────────────────
# UI LAYOUT
# ──────────────────────────────────────────────

st.set_page_config(page_title="Handwriting Collector", page_icon="✏️")

st.title("✏️ Handwriting Collection Tool")
st.caption(f"Your session ID: `{st.session_state.user_id}`")

# ── FINISHED SCREEN ──────────────────────────
if st.session_state.finished:
    st.success("🎉 Thank you! Your handwriting sample has been saved.")
    st.balloons()
    st.info(
        f"All 26 letters have been saved to:\n\n"
        f"`{os.path.join(DATASET_DIR, st.session_state.user_id)}/`"
    )
    st.stop()   # Don't render anything else

# ── ACTIVE COLLECTION SCREEN ─────────────────
current_letter = LETTERS[st.session_state.letter_index]
progress = st.session_state.letter_index / len(LETTERS)

# Progress bar
st.progress(progress, text=f"Letter {st.session_state.letter_index + 1} of {len(LETTERS)}")

# Big instruction prompt
st.markdown(
    f"<h2 style='text-align:center;'>Write the letter &nbsp;"
    f"<span style='color:#4CAF50; font-size:2em;'>{current_letter}</span></h2>",
    unsafe_allow_html=True,
)

# ── DRAWING CANVAS ────────────────────────────
# streamlit-drawable-canvas handles both mouse AND touch input automatically,
# making this work on mobile browsers out of the box.
canvas_result = st_canvas(
    fill_color   = "rgba(255,255,255,0)",   # Transparent fill
    stroke_width = 12,                       # Thick enough to be visible
    stroke_color = "#000000",               # Black ink
    background_color = "#FFFFFF",           # White background
    height = CANVAS_SIZE,
    width  = CANVAS_SIZE,
    drawing_mode = "freedraw",              # Freehand drawing mode
    key = f"canvas_{st.session_state.canvas_key}",  # Changing key resets canvas
    display_toolbar = False,                # Hide the built-in toolbar
)

# ── BUTTONS ───────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    if st.button("🗑️ Clear", use_container_width=True):
        clear_canvas()
        st.rerun()

with col2:
    if st.button("➡️ Next Letter", use_container_width=True, type="primary"):
        next_letter(canvas_result)
        st.rerun()

# ── TIPS ──────────────────────────────────────
st.divider()
st.markdown(
    """
    **Tips:**
    - Write large and fill most of the canvas.
    - Use your finger on mobile or a mouse on desktop.
    - Click **Clear** to redo the letter before saving.
    - Click **Next Letter** when you're happy with your writing.
    """
)
