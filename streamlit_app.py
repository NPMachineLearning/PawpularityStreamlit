import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from os import path

IMAGE_SIZE = 240

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
    if image.width<240 or image.height<240:
        raise Exception("The minimum size of an image is 240 x 240")
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
model_path = path.join("models", "pawnet_240.tflite")
model = load_lite_model(model_path)

# store uploaded image
uploaded_image = None


left_margin, col2, right_margin = st.columns(3)
with left_margin:
    pass
with col2:
    st.header("Pawpularity")
with right_margin:
    pass

st.markdown("""
**Pawpularity is an app mimicing human's perception about an image.
In addition, it generalize popularity of an image and rank it in score.**

**The motivation of pawpularity is to help photographer to take photo of an animal which
is pleasant to general public and hopefully to establish a bounding between the human
and animal at first sight. As a result, this can potentially increase adopting rate in 
animal homeless shelter and curiosity from human to an animal.**  
""")

st.markdown("""
Pawpularity adopt machine learning to analyze an image such as dog and cat
then rank image in score between 0% ~ 100%.

⚠️ Pawpularity was trained with thousand of image of dog and cat therefore, it
is ideal to given an image of dog or cat.

⚠️ Using variety of photography and editing skills such as brightness, saturation, 
cropping and angling may lead to improve or lower the score.

⚠️ Human's perception is dependent in individual and subjective. Therefore, score
from pawpularity is probabilistic and is not ideal for absolute score.
""")

left_margin, col2, right_margin = st.columns([1,3,1])
with left_margin:
    pass
with col2:
    st.subheader("Let's upload an image")
    st.markdown("""
    ⚠️ The minimun size of an image is 240 x 240
    """)
with right_margin:
    pass

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
        left_margin, col2, right_margin = st.columns([5,2,5])
        with left_margin:
            pass
        with col2:
            analyze = st.button("Pawpularity")
        with right_margin:
            pass
        

        if analyze:
            with st.spinner("Analyzing image ...."):
                try:
                    prediction, img = predict_pawpularity(model, uploaded_image, return_image=True)
                    left_margin, col2, right_margin = st.columns([2,3,1])
                    with left_margin:
                        pass
                    with col2:
                        st.markdown(f"""
                        ### Pawpularity: {prediction}%
                        """)
                        if prediction <= 60.:
                            st.markdown("""
                            ⚠️ Try different photography skills or filters can improve the overall score.

                            **Tips:**
                            * Brightness, constrast, saturation
                            * Take photo with indoor or outdoor background
                            * Make sure photo is not blurry
                            * Center the animal in photo
                            * Full animal body in photo
                            * Don't blend animal in background
                            """)
                        elif prediction > 60. and prediction < 90.:
                            st.markdown("""
                            The photo is decent and I believe people like it.
                            """)
                        else:
                            st.markdown("""
                            Well done the photo is perfect and I believe people love it
                            """)
                    with right_margin:
                        pass
                except Exception as e:
                    left_margin, col2, right_margin = st.columns([2,3,1])
                    with left_margin:
                        pass
                    with col2:
                        st.error(e)
                    with right_margin:
                        pass

                
    