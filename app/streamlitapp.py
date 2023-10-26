# Import all of the dependencies
import streamlit as st # use protobuf-3.20.0 not higher that everything works fine
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-02.png')
    st.title("LippReadingAI")
    st.info("This is originally developed from LipNet deeplearning model.")

# Make a title for the options
st.title('LipNet Full Stack App')

# Generating a list of options or videos
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)
 
if options:

    with col1:
        # Video Column
        st.info('The Video below is converted in mp4 format (for streamlit)')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info("This is all the machine learning model sees when making a prediction")
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=500)

        st.info("This is the output of the machine learning model: tokens")
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        # here learn more about the ctc decoder
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)
        
        st.info("Decode the raw tokens into words")
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        