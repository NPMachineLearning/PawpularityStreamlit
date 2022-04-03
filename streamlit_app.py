from distutils.command.upload import upload
from turtle import width
import streamlit as st

st.header("Pawpularity")

with st.container():
    uploaded_image = st.file_uploader("Upload image file", 
                                    type=["jpeg", "jpg"], 
                                    help="Upload an image of pet")

    if uploaded_image is not None:
        left_margin, col2, right_margin = st.columns(3)
        with left_margin:
            pass
        with col2:
            st.image(uploaded_image, caption=uploaded_image.name)
        with right_margin:
            pass