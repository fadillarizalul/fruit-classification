import tensorflow as tf
import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow_hub as hub
from PIL import Image


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model1.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Fruit Quality Classification
         """
         )

file = st.file_uploader("Please upload an image of fruit", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction

if __name__ == '__main__':
    st.write('This is a demo of sBuah app which can classify whether a fruit is Fresh or Rotten')
    st.write('Please upload an image of fruit to classify.')
    st.write('The classification result will be displayed.')

if file is None:
    st.text("Upload an image of fruit")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(prediction)
    st.write(score)
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)