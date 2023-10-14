import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
import numpy as np
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Input, Cropping2D, Flatten
from keras.models import Model
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
model=tf.keras.models.load_model("cnn_semantics.h5")

st.title("""
Contrail Detection Using U-Net Architecture
""")
a=None
file = st.file_uploader("Choose an Input file")

if file is not None:
    print(file.name)
    a=np.load(file)
    x=[]
    y=[]
    xa=a[:,:,:3]
    ya=a[:,:,3]
    x.append(xa.astype('float32'))
    y.append(ya.astype(bool))
    x=np.array(x)
    y=np.array(y)
    prediction = model.predict(x,verbose=1)
    ypredt=prediction>0.5
    










if st.button("Click to predict"):
   fig, axes = plt.subplots(1, 3, figsize=(100, 100))

    # Ash Colored Image
   axes[0].imshow(x[0])
   axes[0].set_title("Ash Colored Image", fontsize=55)

# Human Pixel Mask
   axes[1].imshow(y[0])
   axes[1].set_title("Human Pixel Mask", fontsize=55)

# Leave the third subplot empty

# Predicted Mask
   axes[2].imshow(ypredt[0])
   axes[2].set_title("Predicted Mask", fontsize=55)

# Hide the empty subplot
   st.pyplot(fig)