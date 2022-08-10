import streamlit as st
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
from pathlib import Path
import base64
import time


class_names =['Fruit', 'Package', 'Vegetable']

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes("Supermarket.PNG")
)
st.markdown(
    header_html, unsafe_allow_html=True
)

st.markdown(""" <span style='color:red;'> **Visit us at www.ValuePriceSupermarket.com** </span>""", unsafe_allow_html= True)


@st.cache(allow_output_mutation=True)

def load_model():
  model=tf.keras.models.load_model('C:/Users/yeong/Desktop/LabWork/Mini Projects_CapStone/CAPSTONE/mobilemodel3.hdf5')
  return model

with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Image Classification
         """
         )

st.caption(""" **Marketing Auntie Interactive Dashboard (MAID)**""")
st.markdown(""" <span style='color:black;'> **Categories :     Fruit, Package, Vegetable** </span>""", unsafe_allow_html= True)

image2 = Image.open('C:/Users/yeong/Desktop/LabWork/Mini Projects_CapStone/CAPSTONE/project MAID.png')
st.image(image2, caption='Download now from google playstore',output_format="auto", width=250)


file = st.file_uploader("Please upload an image file ", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=80)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    #st.write(predictions)
    #st.write(score)
    #st.write( 
    #"This image most likely belongs to {} with a {:.2f} percent confidence."
    #.format(class_names[np.argmax(score)], 100 * np.max(score))
    cl = class_names[np.argmax(score)]
    display = f"This image belongs most likely to **{cl}**."
    st.markdown(display)
    
    #st.write( "This image most likely belongs to {}.".format(class_names[np.argmax(score)]))

    
   
#time.sleep(1.5)
    
#st.markdown(""" <span style='color:blue;'> This is </span>   **cool :)** """, unsafe_allow_html= True)
    
    
st.markdown(""" <span style='color:blue;'> ValuePrice - Transforming Tomorrow's Supermarket TODAY </span>""", unsafe_allow_html= True)




