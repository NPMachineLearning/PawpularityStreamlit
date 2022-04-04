from distutils.command.upload import upload
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from os import path

IMAGE_SIZE = 224

@st.cache
def load_lite_model(model_path):
    """
    Load tensorflow lite model
    """
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_pawpularity(lite_model, 
                        image_buffer, 
                        rescale=False,
                        return_image=False):
    """
    Make pawpularity prediction of a tensor image

    Args:
        * lite_model: Tensorflow lite model
        * image_buffer: streamlit uploaded image
        * rescale: Whether to rescale image pixel to 0~1 or not
    
    Return:
        score in float as percentage
    """

    # preprocessing image
    image = Image.open(image_buffer)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    if rescale:
        image = image / 255
    
    # prepare image for input
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)

    # make prediction
    output = lite_model.get_output_details()[0]
    input = lite_model.get_input_details()[0]

    lite_model.set_tensor(input['index'], image)
    lite_model.invoke()
    pred = lite_model.get_tensor(output['index'])
    pred = tf.squeeze(pred)
    pred = tf.round(pred * 100.)

    if return_image:
        if tf.rank(image) == 4:
            img_arr = tf.squeeze(image).numpy().astype(np.uint8)
        return pred.numpy(), Image.fromarray(img_arr)

    return pred.numpy()

# load pawnet model
model_path = path.join("models", "pawnet_224.tflite")
model = load_lite_model(model_path)

# store uploaded image
uploaded_image = None

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

with st.container():
    if uploaded_image is not None:
        analyze = st.button("Analyze pawpularity")

        if analyze:
            prediction, img = predict_pawpularity(model, uploaded_image, return_image=True)
            st.text(f"Pawpularity: {prediction}%")
            st.image(img)
    